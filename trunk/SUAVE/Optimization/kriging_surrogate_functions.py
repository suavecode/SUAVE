from SUAVE.Core import Data
import pyKriging
import pyOpt  
from pyKriging.krige import kriging  
from pyKriging.samplingplan import samplingplan
import numpy as np
import time


def build_kriging_models(filename, inputs):
    #need vector of initial inputs to determine where to separate 
    #inputs from constraints in text file
    file_in = open(filename)
    data = file_in.readlines()
    file_in.close()
    data = format_input_data(data)
    
    #unpack data
    iterations  = data[:,0]
    obj_values  = data[:,1]
    inp_end_idx = len(inputs)+1
    inputs      = data[:,2:inp_end_idx]
    constraints = data[:,inp_end_idx+1:-1]
    
    print 'obj_values = ', obj_values
    print 'inputs=', inputs
    print 'constraints=', constraints
    
    #now build surrogates based on these
    t1=time.time()
  
    
    obj_surrogate = kriging(inputs, obj_values , name='simple')
    obj_surrogate.train()
    constraints_surrogates = []
   
    
    
    for j in range(len(constraints[0,:])):
        print 'j=', j
        constraint_surrogate = kriging(inputs, constraints[:,j] , name='simple')
        constraint_surrogate.train()
        constraints_surrogates.append(constraint_surrogate)
    t2=time.time()
    print 'time to set up = ', t2-t1
    surrogate_function    = run_surrogate_problem()
    surrogate_function.obj_surrogate          = obj_surrogate
    surrogate_function.constraints_surrogates = constraints_surrogates
    
    return obj_surrogate, constraints_surrogates, surrogate_function    
    
    
def setup_surrogate_problem(surrogate_function, inputs, constraints):
    #taken from initial optimization problem that you run
    names            = inputs[:,0] # Names
    bnd              = inputs[:,2] # Bounds
    scl              = inputs[:,3] # Scaling
    
    constraint_names = constraints[:,0]
    constraint_scale = constraints[:,3]
    opt_problem      = pyOpt.optimization('surrogate',run_surrogate_problem)
    
    #constraints
    bnd_constraints    = helper_functions.scale_const_bnds(constraints)
    scaled_constraints = helper_functions.scale_const_values(constraints,bnd_constraints)
       
    
    
    for j in range(len(inputs[:,0])):
        name 
        lbd = bnd[j][0]/scl[j]
        ubd = bnd[j][1]/scl[j]
        opt_problem.addVar(name[j], 'c', lower = lbd, upper = ubd, value = inputs[0][j])
    
    for j in range(len(constraints[:,0])):
        #only using inequality constraints, where we want everything >0
        name = constraint_names[j]
        edge = scaled_constraints[j]
        opt_problem.addCon(name, type ='i', lower=edge, upper=np.inf)
    
    opt_prob.addObj('f')
    return opt_problem 
 
class run_surrogate_problem(Data):
    def __defaults__(self):
        self.obj_surrogate = None
        self.constraints_surrogates = None
    
    def compute(self, x):
        f = self.obj_surrogate.predict(x)
        g = []
        for j in range(len(self.constraints_surrogates)):
            g.append(self.constraint_surrogates[i].predict(x))
        
        fail = 0
        return f, g, fail
        
    __call__ = compute
 
'''
def run_surrogate_problem(self,x):
    f = self.obj_surrogate.predict(x)
    g = []
    for j in range(len(self.constraints_surrogates)):
        g.append(self.constraint_surrogates[i].predict(x))
    
    fail = 0
    return f, g, fail
'''    
def format_input_data(data):

    data_out=[]
  
    for line in data:
       
        line = line.replace('[','')
        line = line.replace(']','')
        line = line.replace(',','')
        line = line.replace('iteration = ' , '')
        line = line.replace(' objective = ' , '')
        line = line.replace('inputs = ' , '')
        line = line.replace('constraints = ' , '')
        numbers = line.split(' ')
        
        numbers_out=[]
        for number in numbers:
            if number != ' ' or number != '\t':
                
                numbers_out.append(float(number))
            
        data_out.append(numbers_out)
    data_out=np.array(data_out)  #change into numpy array to work with later
    
    return data_out
