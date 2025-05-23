# ==============================================================================
#  KratosShapeOptimizationApplication
#
#  License:         BSD License
#                   license: ShapeOptimizationApplication/license.txt
#
#  Main authors:    Baumgaertner Daniel, https://github.com/dbaumgaertner
#                   Geiser Armin, https://github.com/armingeiser
#
# ==============================================================================


# Kratos Core and Apps
import KratosMultiphysics as KM
import KratosMultiphysics.ShapeOptimizationApplication as KSO

# Additional imports
from KratosMultiphysics.ShapeOptimizationApplication.algorithms.algorithm_base import OptimizationAlgorithm
from KratosMultiphysics.ShapeOptimizationApplication import mapper_factory
from KratosMultiphysics.ShapeOptimizationApplication.loggers import data_logger_factory
from KratosMultiphysics.ShapeOptimizationApplication.utilities.custom_timer import Timer
from KratosMultiphysics.ShapeOptimizationApplication.utilities.custom_variable_utilities import WriteDictionaryDataOnNodalVariable
from KratosMultiphysics.ShapeOptimizationApplication import model_part_controller_factory

# ==============================================================================
class AlgorithmSteepestDescentInitialFiltering(OptimizationAlgorithm):
    # --------------------------------------------------------------------------
    def __init__(self, optimization_settings, analyzer, communicator, model_part_controller):
        default_algorithm_settings = KM.Parameters("""
        {
            "name"               : "steepest_descent_initial_filtering",
            "max_iterations"     : 100,
            "relative_tolerance" : 1e-3,
            "gradient_tolerance" : 1e-5,
            "line_search" : {
                "line_search_type"           : "manual_stepping",
                "normalize_search_direction" : true,
                "step_size"                  : 1.0,
                "parameters"                 : {
                                                "mu1": 1e-4,
                                                "mu2": 0.9,
                                                "sigma": 2,   
                                                "max_iterations": 5,
                                                "min_alpha": 1e-1,
                                                "max_alpha": 50.0                                                     
                                                },                                  
                "estimation_tolerance"       : 0.1,
                "increase_factor"            : 1.1,
                "max_increase_factor"        : 10.0
            }
        }""")
        self.algorithm_settings =  optimization_settings["optimization_algorithm"]
        self.algorithm_settings.RecursivelyValidateAndAssignDefaults(default_algorithm_settings)

        self.optimization_settings = optimization_settings
        self.mapper_settings = optimization_settings["design_variables"]["filter"]

        self.analyzer = analyzer
        self.communicator = communicator
        self.model_part_controller = model_part_controller

        self.design_surface = None
        self.mapper = None
        self.data_logger = None
        self.optimization_utilities = None
        self.variable_utils = None

        self.objectives = optimization_settings["objectives"]
        self.constraints = optimization_settings["constraints"]

        self.previos_objective_value = None

        self.max_iterations = self.algorithm_settings["max_iterations"].GetInt() + 1
        self.relative_tolerance = self.algorithm_settings["relative_tolerance"].GetDouble()
        self.gradient_tolerance = self.algorithm_settings["gradient_tolerance"].GetDouble()
        self.line_search_type = self.algorithm_settings["line_search"]["line_search_type"].GetString()
        self.estimation_tolerance = self.algorithm_settings["line_search"]["estimation_tolerance"].GetDouble()
        self.step_size = self.algorithm_settings["line_search"]["step_size"].GetDouble()
        self.increase_factor = self.algorithm_settings["line_search"]["increase_factor"].GetDouble()
        self.max_step_size = self.step_size*self.algorithm_settings["line_search"]["max_increase_factor"].GetDouble()
        self.mu1 = self.algorithm_settings["line_search"]["parameters"]["mu1"].GetDouble()
        self.mu2 = self.algorithm_settings["line_search"]["parameters"]["mu2"].GetDouble()
        self.sigma = self.algorithm_settings["line_search"]["parameters"]["sigma"].GetDouble()
        self.max_iterations_line_search = self.algorithm_settings["line_search"]["parameters"]["max_iterations"].GetInt()
        self.min_alpha = self.algorithm_settings["line_search"]["parameters"]["min_alpha"].GetDouble()
        self.max_alpha = self.algorithm_settings["line_search"]["parameters"]["max_alpha"].GetDouble()

        self.optimization_model_part = model_part_controller.GetOptimizationModelPart()
        self.optimization_model_part.AddNodalSolutionStepVariable(KSO.SEARCH_DIRECTION)

        # Create initial model and related components (model settings, controller and model part)
        initial_model = KM.Model()
        initial_model_settings = optimization_settings["model_settings"].Clone()
        initial_model_settings["model_part_name"].SetString(optimization_settings["model_settings"]["model_part_name"].GetString() + "_initial")
       
        self.initial_model_part_controller = model_part_controller_factory.CreateController(initial_model_settings, initial_model)       
        initial_model_part = self.initial_model_part_controller.GetOptimizationModelPart()

        # Add nodal solution step variables to initial model part
        nodal_variable = KM.KratosGlobals.GetVariable("DF1DX")
        initial_model_part.AddNodalSolutionStepVariable(nodal_variable)
        nodal_variable = KM.KratosGlobals.GetVariable("DF1DX_MAPPED")
        initial_model_part.AddNodalSolutionStepVariable(nodal_variable)

        nodal_variable = KM.KratosGlobals.GetVariable("DF1DX_LINE_SEARCH")
        initial_model_part.AddNodalSolutionStepVariable(nodal_variable)
        nodal_variable = KM.KratosGlobals.GetVariable("DF1DX_LINE_SEARCH_MAPPED")
        initial_model_part.AddNodalSolutionStepVariable(nodal_variable)

        initial_model_part.AddNodalSolutionStepVariable(KSO.CONTROL_POINT_UPDATE)
        initial_model_part.AddNodalSolutionStepVariable(KSO.CONTROL_POINT_CHANGE)
        initial_model_part.AddNodalSolutionStepVariable(KSO.SHAPE_CHANGE)
        
        # Temporary storage for gradient data during line search computations
        nodal_variable = KM.KratosGlobals.GetVariable("DF1DX_LINE_SEARCH")
        self.optimization_model_part.AddNodalSolutionStepVariable(nodal_variable)
        nodal_variable = KM.KratosGlobals.GetVariable("DF1DX_LINE_SEARCH_MAPPED")
        self.optimization_model_part.AddNodalSolutionStepVariable(nodal_variable)
        nodal_variable = KM.KratosGlobals.GetVariable("SEARCH_DIRECTION_LINE_SEARCH")
        self.optimization_model_part.AddNodalSolutionStepVariable(nodal_variable)

        # Counter for object value and gradient calculations during line search computations
        self.line_search_f_df_evaluation_count_per_iteration = 0

    # --------------------------------------------------------------------------
    def CheckApplicability(self):
        if self.objectives.size() > 1:
            raise RuntimeError("Steepest descent algorithm only supports one objective function!")
        if self.constraints.size() > 0:
            raise RuntimeError("Steepest descent algorithm does not allow for any constraints!")

    # --------------------------------------------------------------------------
    def InitializeOptimizationLoop(self):
        self.model_part_controller.Initialize()

        self.analyzer.InitializeBeforeOptimizationLoop()
       
        self.design_surface = self.model_part_controller.GetDesignSurface()
        
        self.model_part_controller.InitializeDamping()
        
        # Initialize initial model components
        self.initial_model_part_controller.Initialize()
        self.initial_design_surface = self.initial_model_part_controller.GetDesignSurface()
        
        self.initial_mapper = mapper_factory.CreateMapper(self.initial_design_surface, self.initial_design_surface, self.mapper_settings)
        self.initial_mapper.Initialize()
        
        self.initial_model_part_controller.InitializeDamping()
        
        self.variable_utils = KM.VariableUtils()

        self.data_logger = data_logger_factory.CreateDataLogger(self.model_part_controller, self.communicator, self.optimization_settings)
        self.data_logger.InitializeDataLogging()

        self.optimization_utilities = KSO.OptimizationUtilities

    # --------------------------------------------------------------------------
    def RunOptimizationLoop(self):
        timer = Timer()
        timer.StartTimer()

        for self.optimization_iteration in range(1,self.max_iterations):
            KM.Logger.Print("")
            KM.Logger.Print("===============================================================================")
            KM.Logger.PrintInfo("ShapeOpt - Initial fltering", "",timer.GetTimeStamp(), ": Starting optimization iteration ",self.optimization_iteration)
            KM.Logger.Print("===============================================================================\n")

            timer.StartNewLap()

            self.__initializeNewShape()

            self.__analyzeShape()

            # Timing for search direction and mapped gradient computation differs by line search method:
            # For adaptive_stepping: Uses search direction and mapped gradient from previous iteration (k-1)
            # For armijo/strong_wolfe: Computes fresh search direction and mapped gradient at current iteration (k)     
            if self.line_search_type in ["armijo", "strong_wolfe"]:
                self.__computeGradientMappingAndSearchDirection()  

            # Execute appropriate line search method
            if self.line_search_type == "adaptive_stepping" and self.optimization_iteration > 1:
                self.__adjustStepSize()
            elif self.line_search_type == "armijo" and self.optimization_iteration > 1:
                self.__ArmijoLineSearch()
            elif self.line_search_type == "strong_wolfe" and self.optimization_iteration > 1:
                self.__strongWolfeLineSearch()

            self.__computeShapeUpdate()

            self.__logCurrentOptimizationStep()

            KM.Logger.Print("")
            KM.Logger.PrintInfo("ShapeOpt", "Time needed for current optimization step = ", timer.GetLapTime(), "s")
            KM.Logger.PrintInfo("ShapeOpt", "Time needed for total optimization so far = ", timer.GetTotalTime(), "s")

            if self.__isAlgorithmConverged():
                break
            else:
                self.__determineAbsoluteChanges()

    # --------------------------------------------------------------------------
    def FinalizeOptimizationLoop(self):
        self.data_logger.FinalizeDataLogging()
        self.analyzer.FinalizeAfterOptimizationLoop()

    # --------------------------------------------------------------------------
    def __initializeNewShape(self):
        self.model_part_controller.UpdateTimeStep(self.optimization_iteration)
        # Update the mesh geometry of the model based on the values of the nodal variable KSO.SHAPE_UPDATE
        self.model_part_controller.UpdateMeshAccordingInputVariable(KSO.SHAPE_UPDATE)
        self.model_part_controller.SetReferenceMeshToMesh()

    # --------------------------------------------------------------------------
    def __analyzeShape(self):
        # Calculate value and gradient(DF1DX) on current model part
        self.communicator.initializeCommunication()
        self.communicator.requestValueOf(self.objectives[0]["identifier"].GetString())
        self.communicator.requestGradientOf(self.objectives[0]["identifier"].GetString())

        self.analyzer.AnalyzeDesignAndReportToCommunicator(self.optimization_model_part, self.optimization_iteration, self.communicator)

        objGradientDict = self.communicator.getStandardizedGradient(self.objectives[0]["identifier"].GetString())
        WriteDictionaryDataOnNodalVariable(objGradientDict, self.optimization_model_part, KSO.DF1DX)

        if self.objectives[0]["project_gradient_on_surface_normals"].GetBool():
            self.model_part_controller.ComputeUnitSurfaceNormals()
            self.model_part_controller.ProjectNodalVariableOnUnitSurfaceNormals(KSO.DF1DX)
      
        self.model_part_controller.DampNodalSensitivityVariableIfSpecified(KSO.DF1DX)
        
        # Copy gradient to initial model part
        self.variable_utils.CopyModelPartNodalVar(KSO.DF1DX, self.model_part_controller.GetOptimizationModelPart(),
                                                  self.initial_model_part_controller.GetOptimizationModelPart(), 0)
    
    # --------------------------------------------------------------------------
    def __adjustStepSize(self):
        current_a = self.step_size

        # Compare actual and estimated improvement using linear information from the previos step
        dfda1 = 0.0
        for node in self.design_surface.Nodes:
            # The following variables are not yet updated and therefore contain the information from the previos step
            s1 = node.GetSolutionStepValue(KSO.SEARCH_DIRECTION)
            dfds1 = node.GetSolutionStepValue(KSO.DF1DX_MAPPED)
            dfda1 += s1[0]*dfds1[0] + s1[1]*dfds1[1] + s1[2]*dfds1[2]

        f2 = self.communicator.getStandardizedValue(self.objectives[0]["identifier"].GetString())
        f1 = self.previos_objective_value

        df_actual = f2 - f1
        df_estimated = current_a*dfda1

        # Adjust step size if necessary
        if f2 < f1:
            estimation_error = (df_actual-df_estimated)/df_actual

            # Increase step size if estimation based on linear extrapolation matches the actual improvement within a specified tolerance
            if estimation_error < self.estimation_tolerance:
                new_a = min(current_a*self.increase_factor, self.max_step_size)

            # Leave step size unchanged if a nonliner change in the objective is observed but still a descent direction is obtained
            else:
                new_a = current_a
        else:
            # Search approximation of optimal step using interpolation
            a = current_a
            corrected_step_size = - 0.5 * dfda1 * a**2 / (f2 - f1 - dfda1 * a )

            # Starting from the new design, and assuming an opposite gradient direction, the step size to the approximated optimum behaves reciprocal
            new_a = current_a-corrected_step_size

        self.step_size = new_a

   # --------------------------------------------------------------------------
    def __ArmijoLineSearch(self):
        """
        Adjust the step size using Armijo's condition: φ(α) ≤ φ(0) + μ₁αφ'(0)
        """
        KM.Logger.PrintInfo("ShapeOpt", "ARMIJO LINE SEARCH PROCEDURE")
        
        # Reset per-iteration counter
        self.line_search_f_df_evaluation_count_per_iteration = 0

        # Get current objective value φ(0) from current k step
        phi_0 = self.communicator.getStandardizedValue(self.objectives[0]["identifier"].GetString())  

        # Compute the dot product of gradient and search direction (φ'(0)) using the data from current k step
        phi_0_prime = 0.0
        for node in self.design_surface.Nodes:
            search_direction = node.GetSolutionStepValue(KSO.SEARCH_DIRECTION)
            gradient = node.GetSolutionStepValue(KSO.DF1DX_MAPPED)
            phi_0_prime += gradient[0] * search_direction[0] + gradient[1] * search_direction[1] + gradient[2] * search_direction[2]      

        KM.Logger.Print("Initial values - phi_0:", phi_0, "phi_0_prime:", phi_0_prime)

        # Use initial step size from settings
        initial_step_size = self.algorithm_settings["line_search"]["step_size"].GetDouble()
        alpha = initial_step_size   
        
        # Armijo loop
        for iteration in range(self.max_iterations_line_search):

            phi_2 = self.__evaluateObjectiveValueandGradientAtNewPoint(alpha, False)

            # Check Armijo condition
            if phi_2 <=  phi_0 +  self.mu1 * alpha * phi_0_prime:
                KM.Logger.Print(f"Backtracking: Accepted alpha={alpha}")
                self.step_size = alpha
                return
            else:
                alpha = max(alpha / self.sigma, self.min_alpha)
                KM.Logger.Print(f"Backtracking: Reducing alpha")           
        
        self.step_size = min(max(alpha, self.min_alpha), self.max_alpha)
        KM.Logger.PrintWarning("ShapeOpt", f"Line search max iterations reached! Final alpha={self.step_size}")
    
    # --------------------------------------------------------------------------
    def __strongWolfeLineSearch(self):
        """
        Line search algorithm that satisfies the strong Wolfe conditions:
        1. Sufficient decrease condition: φ(α) ≤ φ(0) + μ₁αφ'(0)
        2. sufficient curvature condition: |φ'(α)| ≤ μ₂|φ'(0)|
        """
        KM.Logger.PrintInfo("ShapeOpt", "STRONG WOLFE LINE SEARCH PROCEDURE")

        # Reset per-iteration counter
        self.line_search_f_df_evaluation_count_per_iteration = 0

        # Get current objective value φ(0) from current k step
        phi_0 = self.communicator.getStandardizedValue(self.objectives[0]["identifier"].GetString())  

        # Compute the dot product of gradient and search direction (φ'(0)) using the data from current k step
        phi_0_prime = 0.0
        for node in self.design_surface.Nodes:
            search_direction = node.GetSolutionStepValue(KSO.SEARCH_DIRECTION)
            gradient = node.GetSolutionStepValue(KSO.DF1DX_MAPPED)
            phi_0_prime += gradient[0] * search_direction[0] + gradient[1] * search_direction[1] + gradient[2] * search_direction[2]      

        KM.Logger.Print("Initial values - phi_0:", phi_0, "phi_0_prime:", phi_0_prime)

        # Initialize bracketing phase variables
        alpha_1 = 0.0
        phi_1 = phi_0
        phi_1_prime = phi_0_prime
        
        # Use initial step size from settings
        initial_step_size = self.algorithm_settings["line_search"]["step_size"].GetDouble()
        alpha_2 = initial_step_size  

        # Bracketing phase:
        # Identifies an interval that is guaranteed to contain a valid step size satisfying the strong Wolfe conditions
        for iteration in range(self.max_iterations_line_search):

            phi_2, phi_2_prime = self.__evaluateObjectiveValueandGradientAtNewPoint(alpha_2, True)           

            # Check sufficient decrease condition
            if (phi_2 > phi_1 + self.mu1 * alpha_2 * phi_1_prime) or (iteration > 1 and phi_2 > phi_1):
                KM.Logger.Print(f"Bracketing: Found interval [{alpha_1}, {alpha_2}]")
                # Found the current interval containing an acceptable point - switch to pinpoint phase
                # Here, alpha_low = alpha_1, alpha_high = alpha_2
                alpha_opt = self.__pinpointPhase(alpha_1, alpha_2, phi_1, phi_1_prime, 
                                                  phi_2, self.mu1, self.mu2)
                self.step_size = min(max(alpha_opt, self.min_alpha), self.max_alpha)
                return    

            # Check sufficient curvature condition
            if abs(phi_2_prime) <= self.mu2 * abs(phi_1_prime):
                KM.Logger.Print(f"Sufficient curvature condition satisfied at alpha={alpha_2}")
                self.step_size = alpha_2
                return           
            elif phi_2_prime >= 0:
                KM.Logger.Print(f"Bracketing: Found interval [{alpha_1}, {alpha_2}]")
                # Found the current interval containing an acceptable point - switch to pinpoint phase
                # Here, alpha_low = alpha_2, alpha_high = alpha_1
                alpha_opt = self.__pinpointPhase(alpha_2, alpha_1, phi_2, phi_2_prime,
                                                  phi_1, self.mu1, self.mu2)
                self.step_size = min(max(alpha_opt, self.min_alpha), self.max_alpha)
                return
            else:
                # If none of the above conditions are met, expand the interval
                alpha_1 = alpha_2
                alpha_2 = self.sigma * alpha_2
                phi_1 = phi_2
                phi_1_prime = phi_2_prime          

        self.step_size = min(max(alpha_2, self.min_alpha), self.max_alpha)
        KM.Logger.PrintWarning("ShapeOpt", f"Line search max iterations reached! Final alpha={self.step_size}")
    
    # --------------------------------------------------------------------------
    def __pinpointPhase(self, alpha_low, alpha_high, phi_low, phi_low_prime, phi_high, mu1, mu2):
        """
        Pinpointing phase: find a step size satisfying strong Wolfe conditions within the bracketed interval [alpha_low, alpha_high]
        The quadratic interpolation formula is:
                        2α_low(φ_high - φ_low) + φ'_low(α_low² - α_high²)
            α* = ----------------------------------------------------------
                        2[φ_high - φ_low + φ'_low(α_low - α_high)]
        """
        KM.Logger.Print(f"Pinpoint: Initial interval [{alpha_low}, {alpha_high}]")
        
        # Threshold for switching to bisection (when the interval becomes too small)
        min_interval_width = 1e-2
        
        for iteration in range(self.max_iterations_line_search):         
            current_width = abs(alpha_high - alpha_low)

            # Switch to bisection if interval is too small
            if current_width < min_interval_width:
                alpha_p = 0.5 * (alpha_low + alpha_high)
            else:
                # Apply quadratic interpolation to find alpha_p                
                delta_phi = phi_high - phi_low
                alpha_sum = alpha_low + alpha_high
                alpha_diff = alpha_low - alpha_high

                numerator = 2 * alpha_low * delta_phi + phi_low_prime * alpha_diff * alpha_sum
                denominator = 2 *(delta_phi + phi_low_prime * alpha_diff)
                if abs(denominator) < 1e-5:
                    alpha_p = 0.5 * (alpha_low + alpha_high)
                else:
                    alpha_p = numerator / denominator
            
            # Evaluate function at trial point alpha_p
            phi_p, phi_p_prime = self.__evaluateObjectiveValueandGradientAtNewPoint(alpha_p, True)
            
            # Check sufficient decrease condition
            if (phi_p > phi_low + mu1 * alpha_p * phi_low_prime) or (phi_p > phi_low):
                alpha_high = alpha_p
                phi_high = phi_p
            else:              
                # Check sufficient curvature condition
                if abs(phi_p_prime) <= mu2 * abs(phi_low_prime):
                    KM.Logger.Print(f"Pinpoint: Found alpha={alpha_p} satisfying strong Wolfe conditions")
                    return alpha_p
                
                # Check if we need to update intervals
                elif phi_p_prime * (alpha_high - alpha_low) >= 0:
                    # previous low point is high point now
                    alpha_high = alpha_low 
                    phi_high = phi_low

                # p is the lower point 
                alpha_low = alpha_p
                phi_low = phi_p
                phi_low_prime = phi_p_prime
                
        KM.Logger.PrintWarning("ShapeOpt", "Pinpoint phase max iterations reached!")
        return 0.5*(alpha_low + alpha_high)

    # --------------------------------------------------------------------------
    def __evaluateObjectiveValueandGradientAtNewPoint(self, alpha, calculate_gradient):
        """
        This function temporarily updates the mesh, computes the objective function value and gradient at the new point: x_k + alpha * s_k,
        and restores the original mesh state.
        """
        # Increment evaluation counters
        self.line_search_f_df_evaluation_count_per_iteration += 1       

        # Calculate control point update using the input step size
        normalize = self.algorithm_settings["line_search"]["normalize_search_direction"].GetBool()
        self.optimization_utilities.ComputeControlPointUpdate(self.design_surface, alpha, normalize)
        self.optimization_utilities.AddFirstVariableToSecondVariable(self.design_surface, KSO.CONTROL_POINT_UPDATE, KSO.CONTROL_POINT_CHANGE)

        # Copy the control point change from current model to the initial model part
        self.variable_utils.CopyModelPartNodalVar(KSO.CONTROL_POINT_CHANGE, self.model_part_controller.GetOptimizationModelPart(),
                                                  self.initial_model_part_controller.GetOptimizationModelPart(), 0)
        
        # Save previous shape change into a vector
        prev_shape_change = KM.Vector() 
        self.optimization_utilities.AssembleVector(self.design_surface, prev_shape_change, KSO.SHAPE_CHANGE)

        # Forward filtering using initial mapper
        # Map the control point change (control field Xs) to the shape change variable (physical design surface X)
        self.initial_mapper.Map(KSO.CONTROL_POINT_CHANGE, KSO.SHAPE_CHANGE)
        self.initial_model_part_controller.DampNodalUpdateVariableIfSpecified(KSO.SHAPE_CHANGE)
        
        # Copy the shape change from initial model part to current model part
        self.variable_utils.CopyModelPartNodalVar(KSO.SHAPE_CHANGE, self.initial_model_part_controller.GetOptimizationModelPart(),
                                                  self.model_part_controller.GetOptimizationModelPart(), 0)
        
        # Compute SHAPE_UPDATE from SHAPE_CHANGE
        # Save current shape change into a vector
        shape_change = KM.Vector() 
        self.optimization_utilities.AssembleVector(self.design_surface, shape_change, KSO.SHAPE_CHANGE)
        
        # Compute the difference (shape update) between the current and previous shape changes
        shape_update = KM.Vector() 
        shape_update = shape_change - prev_shape_change
        self.optimization_utilities.AssignVectorToVariable(self.design_surface, shape_update, KSO.SHAPE_UPDATE)


        # Update the mesh temporarily
        mesh_utilities = KSO.MeshControllerUtilities(self.design_surface)
        mesh_utilities.UpdateMeshAccordingInputVariable(KSO.SHAPE_UPDATE)
        self.model_part_controller.SetReferenceMeshToMesh()
        
        # Calculate objectivate value and gradient corresponding to the input step size
        new_objective_value, new_gradient_dict = self.analyzer.AnalyzeDesignGetResultDirectly(self.optimization_model_part, self.optimization_iteration, self.communicator, calculate_gradient)
        
        if not calculate_gradient:
            # Restore the original mesh state
            mesh_utilities.RevertMeshUpdateAccordingInputVariable(KSO.SHAPE_UPDATE)
            mesh_utilities.SetReferenceMeshToMesh() 
            return new_objective_value
        else:
            # Save gradient data into KSO.DF1DX_LINE_SEARCH
            WriteDictionaryDataOnNodalVariable(new_gradient_dict, self.optimization_model_part, KSO.DF1DX_LINE_SEARCH)

            if self.objectives[0]["project_gradient_on_surface_normals"].GetBool():
                self.model_part_controller.ComputeUnitSurfaceNormals()
                self.model_part_controller.ProjectNodalVariableOnUnitSurfaceNormals(KSO.DF1DX_LINE_SEARCH)

            self.model_part_controller.DampNodalSensitivityVariableIfSpecified(KSO.DF1DX_LINE_SEARCH)
            
            # Copy gradient to initial model part
            self.variable_utils.CopyModelPartNodalVar(KSO.DF1DX_LINE_SEARCH, self.model_part_controller.GetOptimizationModelPart(),
                                                    self.initial_model_part_controller.GetOptimizationModelPart(), 0)
            
            self.initial_mapper.Update()
            self.initial_mapper.InverseMap(KSO.DF1DX_LINE_SEARCH, KSO.DF1DX_LINE_SEARCH_MAPPED)

            # Copy DF1DX_MAPPED(control field Xs) to current model
            self.variable_utils.CopyModelPartNodalVar(KSO.DF1DX_LINE_SEARCH_MAPPED, self.initial_model_part_controller.GetOptimizationModelPart(),
                                                    self.model_part_controller.GetOptimizationModelPart(), 0)

            self.optimization_utilities.ComputeSearchDirectionSteepestDescentLineSearch(self.design_surface)
        
            if normalize:
                self.optimization_utilities.NormalizeSearchDirectionLineSearch(self.design_surface)

            # Calculate rate of change of objective w.r.t step size: φ'(α)
            new_phi_prime = 0
            for node in self.design_surface.Nodes:
                search_direction = node.GetSolutionStepValue(KSO.SEARCH_DIRECTION_LINE_SEARCH)
                gradient = node.GetSolutionStepValue(KSO.DF1DX_LINE_SEARCH_MAPPED)
                new_phi_prime += gradient[0] * search_direction[0] + gradient[1] * search_direction[1] + gradient[2] * search_direction[2] 

            # Restore the original mesh state
            mesh_utilities.RevertMeshUpdateAccordingInputVariable(KSO.SHAPE_UPDATE)
            mesh_utilities.SetReferenceMeshToMesh() 
       
            return new_objective_value, new_phi_prime    
       
    # --------------------------------------------------------------------------     
    def __computeGradientMappingAndSearchDirection(self):
        # Backward filtering using initial mapper
        # Perform inverse mapping, get the nodal variable DF1DX_MAPPED(control field Xs) from DF1DX(physical design surface X)
        self.initial_mapper.Update()
        self.initial_mapper.InverseMap(KSO.DF1DX, KSO.DF1DX_MAPPED)

        # Copy DF1DX_MAPPED(control field Xs) to current model
        self.variable_utils.CopyModelPartNodalVar(KSO.DF1DX_MAPPED, self.initial_model_part_controller.GetOptimizationModelPart(),
                                                 self.model_part_controller.GetOptimizationModelPart(), 0)
        
        # Compute the search direction
        self.optimization_utilities.ComputeSearchDirectionSteepestDescent(self.design_surface) 
        if self.algorithm_settings["line_search"]["normalize_search_direction"].GetBool():
            self.optimization_utilities.NormalizeSearchDirection(self.design_surface)  
    
    # --------------------------------------------------------------------------
    def __computeShapeUpdate(self):     
        # For constant step size and all line search types except armijo/strong_wolfe:
        # - Computes gradient mapping and search direction here
        
        # For armijo/strong_wolfe:
        # - Uses the current step (k) values (including DF1DX_MAPPED and the search direction), which are precomputed before calling the line search methods
        if self.line_search_type not in ["armijo", "strong_wolfe"]:
            self.__computeGradientMappingAndSearchDirection()    

        # Compute the control point update on current model
        # Normalization is now done in the function __computeGradientMappingAndSearchDirection()
        normalizeF = False
        self.optimization_utilities.ComputeControlPointUpdate(self.design_surface, self.step_size, normalizeF)     
          
        # Add the control point update to the control point change(collect update from each iteration)
        self.optimization_utilities.AddFirstVariableToSecondVariable(self.design_surface, KSO.CONTROL_POINT_UPDATE, KSO.CONTROL_POINT_CHANGE)

        # Copy the control point change from current model to the initial model part
        self.variable_utils.CopyModelPartNodalVar(KSO.CONTROL_POINT_CHANGE, self.model_part_controller.GetOptimizationModelPart(),
                                                  self.initial_model_part_controller.GetOptimizationModelPart(), 0)
        
        # Save previous shape change into a vector
        prev_shape_change = KM.Vector() 
        self.optimization_utilities.AssembleVector(self.design_surface, prev_shape_change, KSO.SHAPE_CHANGE)

        # Forward filtering using initial mapper
        # Map the control point change (control field Xs) to the shape change variable (physical design surface X)
        self.initial_mapper.Map(KSO.CONTROL_POINT_CHANGE, KSO.SHAPE_CHANGE)
        self.initial_model_part_controller.DampNodalUpdateVariableIfSpecified(KSO.SHAPE_CHANGE)
        
        # Copy the shape change from initial model part to current model part
        self.variable_utils.CopyModelPartNodalVar(KSO.SHAPE_CHANGE, self.initial_model_part_controller.GetOptimizationModelPart(),
                                                  self.model_part_controller.GetOptimizationModelPart(), 0)
        
        # Compute SHAPE_UPDATE from SHAPE_CHANGE
        # Save current shape change into a vector 
        shape_change = KM.Vector() 
        self.optimization_utilities.AssembleVector(self.design_surface, shape_change, KSO.SHAPE_CHANGE)
        
        # Compute the difference (shape update) between the current and previous shape changes
        shape_update = KM.Vector() 
        shape_update = shape_change - prev_shape_change
        self.optimization_utilities.AssignVectorToVariable(self.design_surface, shape_update, KSO.SHAPE_UPDATE)

    # --------------------------------------------------------------------------
    def __logCurrentOptimizationStep(self):

        self.previos_objective_value = self.communicator.getStandardizedValue(self.objectives[0]["identifier"].GetString())
        self.norm_objective_gradient = self.optimization_utilities.ComputeL2NormOfNodalVariable(self.design_surface, KSO.DF1DX_MAPPED)

        additional_values_to_log = {}
        additional_values_to_log["step_size"] = self.step_size
        additional_values_to_log["norm_objective_gradient"] = self.norm_objective_gradient
        additional_values_to_log["line_search_evaluations_count"] = self.line_search_f_df_evaluation_count_per_iteration
        self.data_logger.LogSensitivityHeatmap(self.optimization_iteration, self.mapper)
        self.data_logger.LogCurrentValues(self.optimization_iteration, additional_values_to_log)
        self.data_logger.LogCurrentDesign(self.optimization_iteration)

    # --------------------------------------------------------------------------
    def __isAlgorithmConverged(self):

        if self.optimization_iteration > 1 :
            # Check if maximum iterations were reached
            if self.optimization_iteration == self.max_iterations:
                KM.Logger.Print("")
                KM.Logger.PrintInfo("ShapeOpt", "Maximal iterations of optimization problem reached!")
                return True

            # Check gradient norm
            if self.optimization_iteration == 2:
                self.initial_norm_objective_gradient = self.norm_objective_gradient
            else:
                if self.norm_objective_gradient < self.gradient_tolerance*self.initial_norm_objective_gradient:
                    KM.Logger.Print("")
                    KM.Logger.PrintInfo("ShapeOpt", "Optimization problem converged as gradient norm reached specified tolerance of ",self.gradient_tolerance)
                    return True

            # Check for relative tolerance
            relative_change_of_objective_value = self.data_logger.GetValues("rel_change_objective")[self.optimization_iteration]
            if abs(relative_change_of_objective_value) < self.relative_tolerance:
                KM.Logger.Print("")
                KM.Logger.PrintInfo("ShapeOpt", "Optimization problem converged within a relative objective tolerance of ",self.relative_tolerance,"%.")
                return True

    # --------------------------------------------------------------------------
    def __determineAbsoluteChanges(self):
        self.optimization_utilities.AddFirstVariableToSecondVariable(self.design_surface, KSO.CONTROL_POINT_UPDATE, KSO.CONTROL_POINT_CHANGE)
        self.optimization_utilities.AddFirstVariableToSecondVariable(self.design_surface, KSO.SHAPE_UPDATE, KSO.SHAPE_CHANGE)

# ==============================================================================
