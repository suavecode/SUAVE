import numpy as np
import copy

class Trust_Region_Optimization():
    
    def __defaults__(self):
        
        self.tag                      = 'TR_Opt'
        self.max_iterations           = 50
        self.max_function_evaluations = 1000
        self.convergence_tolerance    = 1e-6
        self.constraint_tolerance     = 1e-6
        self.difference_interval      = 1e-6
        
    def initialize(self):
        
        self.iteration_index              = 0
        self.trust_region_center_index    = 0
        self.trust_region_center          = None
        self.shared_data_index            = 0 
        self.truth_history                = None # history for truth function evaluations
        self.surrogate_history            = None # history for evaluation of surrogate models (all fidelity levels)
        self.trust_region_history         = None
        self.number_thrust_evals          = 0
        self.number_duplicate_truth_evals = 0
        self.number_surrogate_evals       = 0
        self.user_data                    = None