
from VyPy.optimize.drivers import Driver
from VyPy.parallel import Remember
import numpy as np
from time import time

try:
    import scipy
    import scipy.optimize  
except ImportError:
    pass

# ----------------------------------------------------------------------
#   Constrained Optimization BY Linear Approximation
# ----------------------------------------------------------------------
class COBYLA(Driver):
    def __init__(self):
        
        # test import
        import scipy.optimize        
        
        Driver.__init__(self)
        
        self.max_evaluations    = 1000
        self.trust_region_scale_start = 1.0
        self.trust_region_scale_end   = 1e-4
        
        # this cache is a special requirement for scipy's 
        # COBYLA constraints
        self._cache = {'x':None, 'c':None}
    
    def run(self,problem):
        
        # store the problem
        self.problem = problem
        
        # single objective
        assert len(problem.objectives) == 1 , 'too many objectives'
        
        # optimizer        
        import scipy.optimize        
        optimizer = scipy.optimize.fmin_cobyla
        
        # inputs
        func   = self.func
        x0     = self.problem.variables.scaled.initials_array()
        cons   = self.setup_cons()
        rhobeg = self.rhobeg()
        rhoend = [ r*self.trust_region_scale_end for r in rhobeg ]
        iprint = 1
        n_eval = self.max_evaluations
        
        # verbosity
        if not self.verbose: iprint = 0
        
        # start timing
        tic = time()
        
        x_min = optimizer( 
            func      = func   ,
            x0        = x0     ,
            cons      = cons   ,
            rhobeg    = rhobeg ,
            rhoend    = rhoend ,
            iprint    = iprint ,
            maxfun    = n_eval ,
        )
        
        # stop timing
        toc = time() - tic
        
        # pull minimizing variables
        vars_min = self.problem.variables.scaled.unpack_array(x_min)
        
        # pack outputs
        outputs = self.pack_outputs(vars_min)
        outputs.success           = True
        outputs.messages.run_time = toc
        
        # done!
        return outputs
        
    def func(self,x):
        objective = self.problem.objectives[0]
        result = objective.function(x)
        result = np.squeeze(result)
        return result
    
    def cons(self,x):
        # check cache
        if np.all( self._cache['x'] == x ):
            return self._cache['c']
            
        # otherwise...
        
        equalities   = self.problem.equalities
        inequalities = self.problem.inequalities
        
        result = []
        
        for inequality in inequalities:
            res = -inequality.function(x)
            result.append(res)
            
        for equality in equalities:
            # build equality constraint with two inequality constraints
            res = equality.function(x)
            result.append(res)
            result.append(-res)
            
        # todo - design space bounds
            
        if result:
            result = np.vstack(result)
            result = np.squeeze(result)
            
        # store to cache
        self._cache['x'] = x + 0 
        self._cache['c'] = result + 0
            
        return result
            
        
    def setup_cons(self):        
        
        # find out number of constraints
        x0 = self.problem.variables.scaled.initials_array()[:,0]
        c0 = self.cons(x0)
        n_c0 = len(c0)
        
        # build a list of constraint function handles
        result = [ _Constraint( self.cons, i ) for i in range(n_c0) ]
        
        return result
    
    def rhobeg(self):
        bounds = self.problem.variables.scaled.bounds()
        rho = []
        for b in bounds:
            lo,hi = b
            r = (hi-lo)*self.trust_region_scale_start
            rho.append(r)
        return rho
    
    
class _Constraint(object):
    def __init__(self,con,i):
        self.con = con
        self.i   = i
    def __call__(self,x):
        return self.con(x)[self.i]