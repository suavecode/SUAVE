import numpy as np
import copy

class Trust_Region():
    
    def __defaults__(self):
        
        self.initial_size = 0.5
        self.size = 0.5
        self.minimum_size = 1e-15
        self.contract_threshold = 0.25
        self.expand_threshold = 0.75
        self.contraction_factor = 0.25
        self.expansion_factor = 2.
    
        self.max_iterations = 50
        self.soft_convergence_limit = 5
        self.convergence_tolerance = 1e-6
        self.constraint_tolerance = 1e-6
    
        self.approx_subproblem = 'direct'
        self.merit_function = 'penalty'
        self.acceptance_test = 'ratio'
    
        self.correction_type = 'additive'
        self.correction_order = 1