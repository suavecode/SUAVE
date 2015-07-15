
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from VyPy.optimize.drivers import Driver
import numpy as np
from time import time

try:
    import scipy
    import scipy.optimize  
except ImportError:
    pass


# ----------------------------------------------------------------------
#   Broyden-Fletcher-Goldfarb-Shanno Algorithm
# ----------------------------------------------------------------------

class BFGS(Driver):
    def __init__(self):
        
        # test import
        import scipy.optimize
        
        Driver.__init__(self)
        
        self.max_iterations = 1000
        
    
    def run(self,problem):
        
        # store the problem
        self.problem = problem
        
        # single objective
        assert len(problem.objectives) == 1 , 'too many objectives'
        
        # optimizer
        import scipy.optimize
        optimizer = scipy.optimize.fmin_bfgs
        
        # inputs
        func   = self.func
        fprime = None
        x0     = problem.variables.scaled.initials_array()
        n_iter = self.max_iterations
        disp   = self.verbose
        
        # start timing
        tic = time()
        
        # run the optimizer
        result = optimizer( 
            f           = func   ,
            x0          = x0     ,
            fprime      = fprime ,
            full_output = True   ,
            disp        = disp   ,
            maxiter     = n_iter ,
            **self.other_options.to_dict() 
        )
        
        # stop timing
        toc = time() - tic        
        
        # get final variables
        x_min = result[0]        
        vars_min = self.problem.variables.scaled.unpack_array(x_min)
        
        # pack outputs
        outputs = self.pack_outputs(vars_min)
        outputs.success               = result[6] == 0
        outputs.messages.exit_flag    = result[6]
        outputs.messages.evaluations  = result[4]
        outputs.messages.run_time     = toc
        
        # done!
        return outputs

    def func(self,x):
        
        obj = self.objective(x)[0,0]
        cons = self.constraints(x)
        
        # penalty for constraints
        result = obj + sum( cons**2 ) * 100000.0
        
        return result
            
    def objective(self,x):
        objective = self.problem.objectives[0]
        result = objective.function(x)
        return result
        
    def constraints(self,x):
        equalities   = self.problem.equalities
        inequalities = self.problem.inequalities
        
        result = []
        
        for inequality in inequalities:
            res = inequality.function(x)
            res[res<0.0] = 0.0
            result.append(res)
        for equality in equalities:
            res = equality.function(x)
            result.append(res)
            
        if result:
            result = np.vstack(result)
            result = np.squeeze(result)
            
        return result
    

