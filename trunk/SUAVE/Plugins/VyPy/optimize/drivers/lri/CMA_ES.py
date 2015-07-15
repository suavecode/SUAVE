
from VyPy.optimize.drivers import Driver
import numpy as np
from time import time

try:
    import cma
except ImportError:
    pass
    
# ----------------------------------------------------------------------
#   Covariance Matrix Adaptation - Evolutionary Strategy
# ----------------------------------------------------------------------
class CMA_ES(Driver):
    def __init__(self):
        
        Driver.__init__(self)
        
        self.verbose                  = True
        self.print_iterations         = 1
        self.standard_deviation_ratio = 0.10
        self.max_evaluations          = np.inf
    
    def run(self,problem):
        
        # store the problem
        self.problem = problem
        
        # single objective
        assert len(problem.objectives) == 1 , 'too many objectives'
        
        # optimizer
        from VyPy.plugins import cma
        optimizer = cma.fmin
        
        # inputs
        func   = self.func
        x0     = problem.variables.scaled.initials_array()
        x0     = np.ravel(x0)
        sigma0 = self.sigma0()
        bounds = problem.variables.scaled.bounds_array()
        bounds = [ bounds[:,0] , bounds[:,1] ]
        evals  = self.max_evaluations
        iprint = self.print_iterations
        
        # printing
        if not self.verbose: iprint = 0
        
        options = {
            'bounds'    : bounds      ,
            'verb_disp' : iprint      ,
            'verb_log'  : 0           ,
            'verb_time' : 0           ,
            'maxfevals' : evals       ,
        }
        options.update(self.other_options.to_dict())
        
        # start timing
        tic = time()
        
        # run the optimizer
        result = optimizer( 
            objective_function = func    ,
            x0                 = x0      ,
            sigma0             = sigma0  ,
            options            = options ,
        )
        
        # stop timing
        toc = time() - tic
        
        # pull minimizing variables
        x_min = result[0]
        vars_min = self.problem.variables.scaled.unpack_array(x_min)
        
        # stringify message keys
        messages = { str(k):v for k,v in result[7].items() }
        
        # success criteria
        success = False
        for k in ['ftarget','tolx','tolfun']:
            if messages.has_key(k):
                success = True
                break
        
        # pack outputs
        outputs = self.pack_outputs(vars_min)
        outputs.success               = success
        outputs.messages.exit_message = messages
        outputs.messages.evaluations  = result[3]
        outputs.messages.iterations   = result[4]
        outputs.messages.run_time     = toc
        
        return outputs
    
    def func(self,x):
        
        obj  = self.objective(x)[0,0]
        cons = self.constraints(x)
        
        # penalty for constraints
        result = obj + np.sum( cons**2 ) * 100000.0
        
        return result
            
    def objective(self,x):
        objective = self.problem.objectives[0]
        result = objective.function(x)
        return result
        
    def constraints(self,x):
        equalities   = self.problem.equalities
        inequalities = self.problem.inequalities
        
        result = np.empty([0,1])
        for inequality in inequalities:
            res = inequality.function(x)
            res[res < 0.0] = 0.0
            result = np.vstack([result, res])
        for equality in equalities:
            res = equality.function(x)
            result = np.vstack([result, res])      
            
        return result
    
    def sigma0(self):
        bounds = self.problem.variables.scaled.bounds_array()
        sig0 = np.mean( np.diff(bounds) ) * self.standard_deviation_ratio
        return sig0
        

    
"""

        `sigma0`
            initial standard deviation.  The problem variables should
            have been scaled, such that a single standard deviation
            on all variables is useful and the optimum is expected to
            lie within about `x0` +- ``3*sigma0``. See also options
            `scaling_of_variables`. Often one wants to check for
            solutions close to the initial point. This allows,
            for example, for an easier check of consistency of the
            objective function and its interfacing with the optimizer.
            In this case, a much smaller `sigma0` is advisable.


    
ftarget
    -inf  
    target function value, minimization
maxfevals
    inf  
    maximum number of function evaluations
maxiter
    100 + 50 * (N+3)**2 // popsize**0.5  
    maximum number of iterations
tolx
    1e-11
    termination criterion: tolerance in x-changes
tolfacupx
    1e3
    termination when step-size increases by tolfacupx 
    (diverges). That is, the initial step-size was chosen 
    far too small and better solutions were found far away 
    from the initial solution x0',
tolfun
    1e-11  
    tolerance in function value
tolfunhist
    1e-12  
    tolerance in function value history, minimum movement 
    after first 9 iterations 
tolstagnation
    int(100 + 100 * N**1.5 / popsize)
    termination if no improvement over tolstagnation iterations
tolupsigma
    1e20
    tolerance on "creeping behavior"
    sigma/sigma0 > tolupsigma * max(sqrt(eivenvals(C))) 
    indicates "creeping behavior" with usually minor 
    improvements

"""