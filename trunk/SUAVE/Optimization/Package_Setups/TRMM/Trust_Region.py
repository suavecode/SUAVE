'''
Define the trust region class.

Rick Fenrich 8/9/16
'''
import SUAVE
from SUAVE.Core import Data
import numpy as np
import copy

class Trust_Region(Data):
    def __init__(self):
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
        
    def set_center(self,trc):
        self.center = copy.copy(trc)
            
    def evaluate_function(self,f,gviol,*args):
        # Previously calculated values of objective function, inequality constraints, and 
        # equality constraints at some x are used as inputs. Only active inequality constraints
        # are handed to this function through gviol which is the Euclidean 2-norm of the 
        # nonlinear constraint violation when the constraints are in the from g <= 0
        if( self.merit_function == 'penalty' ):
            k = args[0]
            offset = args[1]
            rp = np.exp((k+offset)/10)
            phi = f + gviol**2
            return phi
        elif( self.merit_function == 'augmented-lagrangian' ):
            raise NotImplementedError('Augmented-Lagrangian merit function not implemented')
        else:
            raise NotImplementedError('%s merit function not implemented' % self.merit_function)
        
