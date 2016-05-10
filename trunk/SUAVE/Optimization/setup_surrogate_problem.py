

def setup_surrogate_problem(surrogate_function, inputs, constraints):
    #taken from initial optimization problem that you run
    names            = inputs[:,0] # Names
    ini              = inputs[:,1] # values
    bnd              = inputs[:,2] # Bounds
    scl              = inputs[:,3] # Scaling
    
    constraint_names = constraints[:,0]
    constraint_scale = constraints[:,3]
    opt_problem      = pyOpt.Optimization('surrogate', surrogate_function)
    
    #constraints
    bnd_constraints    = helper_functions.scale_const_bnds(constraints)
    scaled_constraints = helper_functions.scale_const_values(constraints,bnd_constraints)
       
    x = ini#/scl
    
    
    for j in range(len(inputs[:,1])):
        print 'len(names)=',len(names)
        name = names[j]
        lbd = bnd[j][0]#/scl[j]
        ubd = bnd[j][1]#/scl[j]
        print 'name=', name
        print 'lbd=', lbd
        print 'ubd=', ubd
        print 'value=',x[j]
        
        opt_problem.addVar(name, 'c', lower = lbd, upper = ubd, value = x[j])
    
    for j in range(len(constraints[:,0])):
        #only using inequality constraints, where we want everything >0
        name = constraint_names[j]
        edge = scaled_constraints[j]
        opt_problem.addCon(name, type ='i', lower=edge, upper=np.inf)
    
    opt_problem.addObj('f')
    
    return opt_problem, surrogate_function
 
class run_surrogate_problem(Data):
    def __defaults__(self):
        self.obj_surrogate = None
        self.constraints_surrogates = None
    
    def compute(self, x):
        f = self.obj_surrogate.predict(x)
        
        g = []
        for j in range(len(self.constraints_surrogates)):
            g.append(self.constraints_surrogates[j].predict(x))
        
        fail = 0
        print 'f=', f
        print 'g=', g
        return f, g, fail
        
    __call__ = compute