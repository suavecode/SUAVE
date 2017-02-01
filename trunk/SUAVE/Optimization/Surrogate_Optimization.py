#Surrogate_Optimization.py
#Created: Jul 2016, M. Vegh

from SUAVE.Core import Data
from SUAVE.Surrogate.svr_surrogate_functions import build_svr_models
from SUAVE.Surrogate.kriging_surrogate_functions import build_kriging_models
from SUAVE.Surrogate.vypy_surrogate_functions import build_gpr_models
from SUAVE.Surrogate.scikit_surrogate_functions import build_scikit_models

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
    def __defaults__(self):
        self.sample_plan           = None #VyPy.sampling.lhc_uniform
        self.problem               = None  #SUAVE nexus object
        self.optimizer             = None #pyOpt.pySNOPT.SNOPT()
        self.surrogate_model       = None #Kriging, SVR, GPR, or any scikit learn regression  #used for different options for 
        self.optimization_filename = None  #where you keep track of results
        self.number_of_points      = 0.
        self.max_iterations        = 100
        
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
                        
        if npoints > 0: #use 0 points to utilize an existing dataset
            bounds        = []
            scaled_bounds = []
            
            for i in range(len(bnd)):
                lb = bnd[i][0] /scl[i]
                ub = bnd[i][1] /scl[i] 
        
                scaled_bounds.append([lb,ub])
                
            #now handle constraints    
            scaled_bounds      = np.array(scaled_bounds)
    
            #now create a sample
            npoints = self.number_of_points
            Xsample = self.sample_plan(scaled_bounds,npoints)
    
            #now run; results will be written to file, which can be read later
            for i in range(0,npoints):
        
                opt_prob.inputs[:,1] = Xsample[i,:]*scl#/base_units
            
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
        
        for j in range(0,self.max_iterations):
            if j ==0 or self.surrogate_model != 'Kriging':
                surr_iterations, surr_obj_values, surr_inputs, surr_constraints = read_optimization_outputs(filename, base_inputs, base_constraints)
            if self.surrogate_model == 'SVR':
                obj_surrogate, constraints_surrogates ,surrogate_function = build_svr_models(surr_obj_values, surr_inputs ,surr_constraints, C = 1E5, epsilon=.01 )
            elif self.surrogate_model == 'Kriging':
                #obj_surrogate, constraints_surrogates ,surrogate_function = build_kriging_models(surr_obj_values, surr_inputs ,surr_constraints)
                
                if j==0:
                    obj_surrogate, constraints_surrogates ,surrogate_function = build_kriging_models(surr_obj_values, surr_inputs ,surr_constraints)
                    
                else:       #add to existing surrogate to improve code speed
                    xt1= time.time()
                    obj_surrogate.addPoint(x_out, output_real[0])
                    obj_surrogate.train()
                    for k in range(len(constraints_surrogates)):
                        constraints_surrogates[k].addPoint(x_out,problem.all_constraints()[k])
                        constraints_surrogates[k].train()
                    xt2= time.time()
                    #reassign to surrogate_function
                    surrogate_function.obj_surrogate  = obj_surrogate
                    surrogate_function.constraints_surrogates =constraints_surrogates
                    print 'time to train model=', xt2-xt1
            elif self.surrogate_model == 'GPR':
                obj_surrogate, constraints_surrogates ,surrogate_function = build_gpr_models(surr_obj_values, surr_inputs ,surr_constraints, base_inputs)
            
            else: #directly call scikit learn models
                obj_surrogate, constraints_surrogates ,surrogate_function = build_scikit_models(self, surr_obj_values, surr_inputs ,surr_constraints)
            surrogate_problem = pyopt_surrogate_setup(surrogate_function, base_inputs, base_constraints)
        
            t3 = time.time()
        
            surrogate_outputs = optimizer(surrogate_problem) 
            print 'j=', j
            print 'surrogate_outputs[0]=',surrogate_outputs[0]
            print 'x_out=', surrogate_outputs[1]
            
            
            if j>1:
                x_diff = surrogate_outputs[1]-x_out
                print 'x_diff=', x_diff 
                if np.linalg.norm(x_diff)<.0001:  #exit for loop if surrogate optimization converges
                    print 'surrogate optimization terminated successfully'
                    break
            x_out = surrogate_outputs[1]*1.
            t4 = time.time()
            print 'surrogate optimization time=', t4-t3
            print surrogate_outputs
            
            
            
        
            f_out, g_out, fail_out = surrogate_function(np.array(x_out))
          
            print 'f_out=', f_out
            print 'g_out=', g_out
            print 'fail_out=', fail_out
            opt_prob.inputs[:,1] = surrogate_outputs[1]*scl/base_units
            
            output_real = problem.objective(surrogate_outputs[1])
            print 'opt_prob.inputs[:,1]=', opt_prob.inputs[:,1]
            print 'output_real=', output_real
            print 'constraints_out=', problem.all_constraints()
           
            
        return output_real, surrogate_problem