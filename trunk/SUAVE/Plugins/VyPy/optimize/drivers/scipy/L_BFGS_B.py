
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

class L_BFGS_B(Driver):
    def __init__(self):
        
        # test import
        import scipy.optimize
        
        Driver.__init__(self)
        
        self.max_iterations = 1000
        self.max_evaluations = 1000
    
    def run(self,problem):
        
        # store the problem
        self.problem = problem
        
        # single objective
        assert len(problem.objectives) == 1 , 'too many objectives'
        
        # optimizer
        import scipy.optimize
        optimizer = scipy.optimize.fmin_l_bfgs_b
        
        # inputs
        func   = self.func
        fprime = self.fprime
        approx_grad = False
        x0     = problem.variables.scaled.initials_array()
        bounds = problem.variables.scaled.bounds_array()
        n_func = self.max_evaluations
        n_iter = self.max_iterations
        iprint = 0
        
        # gradients
        # gradients?
        dobj,dineq,deq = problem.has_gradients()
        if not (dobj and dineq and deq) : fprime = None
        if fprime is None: approx_grad = True
        
        # printing
        if not self.verbose: iprint = -1  
        
        # start timing
        tic = time()
        
        # run the optimizer
        result = optimizer( 
            func        = func        ,
            x0          = x0          ,
            fprime      = fprime      ,
            approx_grad = approx_grad ,
            bounds      = bounds      ,
            maxfun      = n_func      ,
            maxiter     = n_iter      ,
            iprint      = iprint      ,
            **self.other_options.to_dict() 
        )
        
        # stop timing
        toc = time() - tic        
        
        # get final variables
        x_min = result[0]        
        vars_min = self.problem.variables.scaled.unpack_array(x_min)
        
        # pack outputs
        outputs = self.pack_outputs(vars_min)
        outputs.success               = result[2]['warnflag'] == 0
        outputs.messages.exit_flag    = result[2]['warnflag']
        outputs.messages.evaluations  = result[2]['funcalls']
        outputs.messages.iterations   = result[2]['nit']
        outputs.messages.run_time     = toc
        
        # done!
        return outputs

    def func(self,x):
        
        obj = self.objective(x)[0,0]
        cons = self.constraints(x)
        
        # penalty for constraints
        result = obj + np.sum( cons**2. ) * 100000.0
        
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
            
        return result
    
    
    def fprime(self,x):
        
        dobj  = self.grad_objective(x)
        cons  = self.constraints(x)
        dcons = self.grad_constraints(x)
        
        # penalty for constraints
        result = dobj + np.sum( (2. * cons * dcons) , axis=0) * 100000.0        
        
        result = np.squeeze(result)
        
        return result
    
    def grad_objective(self,x):
        objective = self.problem.objectives[0]
        result = objective.gradient(x)
        return result
        
    def grad_constraints(self,x):
        equalities   = self.problem.equalities
        inequalities = self.problem.inequalities
        
        result = []
        
        for inequality in inequalities:
            res = inequality.function(x)
            i_feas = res<0.0
            res = inequality.gradient(x)
            res[i_feas] = 0.0
            result.append(res)
        for equality in equalities:
            res = equality.gradient(x)
            result.append(res)
            
        if result:
            result = np.vstack(result)
            
        return result
    
    

