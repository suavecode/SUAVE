#Surrogate_Optimization.py
#Created: Jul 2016, M. Vegh

from SUAVE.Core import Data
from SUAVE.Surrogate.svr_surrogate_functions import build_svr_models
from SUAVE.Optimization.Package_Setups.pyopt_surrogate_setup import pyopt_surrogate_setup
from SUAVE.Optimization.Package_Setups.pyopt_surrogate_setup import pyopt_surrogate_setup
from read_optimization_outputs import read_optimization_outputs
import numpy as np
import time

# ----------------------------------------------------------------------
#  Surrogate_Optimization
# ----------------------------------------------------------------------

'''
Takes a SUAVE Optimization problem, builds a surrogate around it, 
and iteratively optimizes the surrogate, sampling the SUAVE problem at
the optimum determined by the surrogate

(currently only uses SVR, plan to add other surrogate options later)

'''
class Surrogate_Optimization(Data):
    def defaults(self):
        self.sample_plan           = None #VyPy.sampling.lhc_uniform
        self.problem               = None  #SUAVE nexus object
        self.optimizer             = pyOpt.pySNOPT.SNOPT()
        self.optimization_filename = None  #where you keep track of results
        self.number_of_points      = 0.
        
    def build_surrogate(self):
        #unpack
        npoints           = self.number_of_points
        problem           = self.problem
        opt_prob          = self.problem.optimization_problem

        base_inputs       = opt_prob.inputs
        names             = base_inputs[:,0] # Names
        bnd               = base_inputs[:,2] # Bounds
        scl               = base_inputs[:,3] # Scaling
        base_units        = base_inputs[:,-1]*1.0
        base_inputs[:,-1] = base_units #keeps it from overwriting 
                        

       
        bounds        = []
        scaled_bounds = []
        for i in range(len(bnd)):
            lb = bnd[i][0] *base_units[i]/scl[i]
            ub = bnd[i][1] *base_units[i]/scl[i] 
    
            scaled_bounds.append([lb,ub])
            
        #now handle constraints    
        scaled_bounds      = np.array(scaled_bounds)
        
        #now create a sample
        npoints = self.number_of_points
        Xsample = self.sampling_plan(scaled_bounds,npoints)
        
        #now run; results will be written to file, which can be read later
        for i in range(0,npoints):
            opt_prob.inputs[:,1] = Xsample[i,:]*scl
            problem.objective()
        return 
        
        
        #now set up optimization problem on surrogate
        
    def iterative_optimization(self):
        filename  = self.optimization_filename
        problem   = self.problem
        optimizer = self.optimizer
        opt_prob  = problem.optimization_problem
        
        
        base_inputs       = opt_prob.inputs
        scl               = base_inputs[:,3] # Scaling
        base_constraints  = opt_prob.constraints
        base_units        = base_inputs[:,-1]*1.0
        base_inputs[:,-1] = base_units #keeps it from overwriting 
        
        #constraint_names  = base_constraints[:,0]
        #constraint_scale  = base_constraints[:,3]
        
        for j in range(0,300):
            surr_iterations, surr_obj_values, surr_inputs, surr_constraints = read_optimization_outputs(filename, base_inputs, base_constraints)
            obj_surrogate, constraints_surrogates ,surrogate_function = build_svr_models(surr_obj_values, surr_inputs ,surr_constraints, C = 1E5, epsilon=.01 )
            surrogate_problem = pyopt_surrogate_setup(surrogate_function, base_inputs, base_constraints)
        
        
            t3 = time.time()
        
            surrogate_outputs = optimizer(surrogate_problem) 
        
            t4 = time.time()
            print 'surrogate optimization time=', t4-t3
            print surrogate_outputs
            
            print 'surrogate_outputs[0]=',surrogate_outputs[0]
            print 'surrogate_outputs[1]=',surrogate_outputs[1]
        
        
            f_out, g_out, fail_out = surrogate_function(np.array(surrogate_outputs[1]))
            print 'f_out=', f_out
            print 'g_out=', g_out
            print 'fail_out=', fail_out
            opt_prob.inputs[:,1] = surrogate_outputs[1]*scl/base_units
            output_real = problem.objective(surrogate_outputs[1])
            print 'opt_prob.inputs[:,1]=', opt_prob.inputs[:,1]
            print 'output_real=', output_real
            print 'constraints_out=', problem.all_constraints()
            
        return output_real, surrogate_problem