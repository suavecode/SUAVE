# Nexus.py
# 
# Created:  Jul 2015, E. Botero 
# Modified: Feb 2015, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE 
from SUAVE.Core import Data, DataOrdered
from SUAVE.Analyses import Process
from copy import deepcopy
import helper_functions as help_fun
import numpy as np
from SUAVE.Optimization.read_gradient_outputs import read_gradient_outputs
from SUAVE.Optimization.write_gradient_outputs import write_gradient_outputs
# ----------------------------------------------------------------------
#  Nexus Class
# ----------------------------------------------------------------------
    
class Nexus(Data):
    
    def __defaults__(self):
        self.vehicle_configurations      = SUAVE.Components.Configs.Config.Container()
        self.analyses                    = SUAVE.Analyses.Analysis.Container()
        self.missions                    = None
        self.procedure                   = Process()
        self.results                     = SUAVE.Analyses.Results()
        self.summary                     = Data()
        self.optimization_problem        = None
        self.last_inputs                 = None
        self.evaluation_count            = 0
        self.finite_difference_step      = 1E-8
        self.gradient_values             = None
        self.write_gradients             = False
        self.minimum_gradient_write_step = 1E-4
        self.gradient_filename           = 'gradient_results.txt'
        
        
    def evaluate(self,x = None):
        
        self.unpack_inputs(x)
        # This function calls really_evaluate
        if np.all(self.optimization_problem.inputs==self.last_inputs):
            pass
        else:
            self._really_evaluate()
        
    
    def _really_evaluate(self):
        
        nexus = self
        
        self.evaluation_count += 1
        
        for key,step in nexus.procedure.items():
            if hasattr(step,'evaluate'):
                self = step.evaluate(nexus)
            else:
                nexus = step(nexus)
            self = nexus
                
        # Store to cache
        self.last_inputs = deepcopy(self.optimization_problem.inputs)
          
    
    def objective(self,x = None):
        
        self.evaluate(x)
        
        aliases     = self.optimization_problem.aliases
        objective   = self.optimization_problem.objective
        results     = self.results
    
        objective_value  = help_fun.get_values(self,objective,aliases)  
        scaled_objective = help_fun.scale_obj_values(objective,objective_value)
        
        return scaled_objective
    
    def inequality_constraint(self,x = None):
        
        self.evaluate(x)
        
        aliases     = self.optimization_problem.aliases
        constraints = self.optimization_problem.constraints
        results     = self.results
        
        # Setup constraints  
        indices = []
        for ii in xrange(0,len(constraints)):
            if constraints[ii][1]==('='):
                indices.append(ii)        
        iqconstraints = np.delete(constraints,indices,axis=0)
    
        if iqconstraints == []:
            scaled_constraints = []
        else:
            constraint_values = help_fun.get_values(self,iqconstraints,aliases)
            constraint_values[iqconstraints[:,1]=='<'] = -constraint_values[iqconstraints[:,1]=='<']
            bnd_constraints   = constraint_values - help_fun.scale_const_bnds(iqconstraints)
            scaled_constraints = help_fun.scale_const_values(iqconstraints,constraint_values)

        return scaled_constraints      
    
    def equality_constraint(self,x = None):
        
        self.evaluate(x)

        aliases     = self.optimization_problem.aliases
        constraints = self.optimization_problem.constraints
        results     = self.results
        
        # Setup constraints  
        indices = []
        for ii in xrange(0,len(constraints)):
            if constraints[ii][1]=='>':
                indices.append(ii)
            elif constraints[ii][1]=='<':
                indices.append(ii)
        eqconstraints = np.delete(constraints,indices,axis=0)
    
        if eqconstraints == []:
            scaled_constraints = []
        else:
            constraint_values = help_fun.get_values(self,eqconstraints,aliases) - help_fun.scale_const_bnds(eqconstraints)
            scaled_constraints = help_fun.scale_const_values(eqconstraints,constraint_values)

        return scaled_constraints   
    
    def all_constraints(self,x = None):
        
        self.evaluate(x)
        
        aliases     = self.optimization_problem.aliases
        constraints = self.optimization_problem.constraints
        results     = self.results
    
        constraint_values = help_fun.get_values(self,constraints,aliases) 
        scaled_constraints = help_fun.scale_const_values(constraints,constraint_values)
    
        return scaled_constraints     
    
    
    def unpack_inputs(self,x = None):
        
        # Scale the inputs if given
        inputs = self.optimization_problem.inputs
        if x is not None:
            inputs = help_fun.scale_input_values(inputs,x)
            
        # Convert units
        converted_values = help_fun.convert_values(inputs)
        
        # Set the dictionary
        aliases = self.optimization_problem.aliases
        vehicle = self.vehicle_configurations
        
        self = help_fun.set_values(self,inputs,converted_values,aliases)     
    
    def constraints_individual(self,x = None):
        pass     

    def finite_difference(self,x = None, f= None, g = None): #f and g not used; just used to allow common interface for pyOpt
        inputs = self.optimization_problem.inputs
        if x == None:
            x = inputs[:,1] / inputs[:,3] #unscale input values
            
        fd_step = self.finite_difference_step
        obj     = self.objective(x)
        con     = self.all_constraints(x)
        
        inpu  = self.optimization_problem.inputs
        const = self.optimization_problem.constraints
        
        inplen = len(inpu)
        conlen = len(const)
        
        grad_obj = np.zeros(inplen)
        jac_con  = np.zeros((inplen,conlen))
        
        con2 = (con*np.ones_like(jac_con))
        
        for ii in xrange(0,inplen):
            newx     = np.asarray(x)*1.0
            newx[ii] = newx[ii]+ fd_step
            
            grad_obj[ii]  = self.objective(newx)
            jac_con[ii,:] = self.all_constraints(newx)
        
        grad_obj = (grad_obj - obj)/(fd_step)
        
        jac_con = (jac_con - con2).T/(fd_step)
        
        grad_obj             = grad_obj.astype(float)
        jac_con              = jac_con.astype(float)
        self.gradient_values = [grad_obj, jac_con]
        
        if self.write_gradients == True:
            filename = self.gradient_filename
            data_inputs, obj_grads, con_grads, read_success = read_gradient_outputs(filename, x, con)
            #determine the initial step
            min_norm = 1000.
            diff = np.subtract(x, data_inputs)
            imin_dist = -1 
            if read_success:
                for k in range(len(diff[:,-1])):
                    row = diff[k,:]
                    row_norm = np.linalg.norm(row)
                    if row_norm < min_norm:
                        min_norm = row_norm
                        imin_dist = k*1 

            if read_success == 0 or min_norm > self.minimum_gradient_write_step:
                write_gradient_outputs(x, grad_obj, jac_con, filename)
            '''
            grad_filename = 'gradient_results.txt'
            file = open(grad_filename, 'ab')
            file.write('inputs = ')
            file.write(str(x.tolist()))
            file.write(', objective = ')
            file.write(str(grad_obj.tolist()))
            file.write(', constraints = ')
            file.write(str(jac_con.tolist()))
            file.write('\n') 
            file.close()    
            '''
     
     
        fail                 = 0
        print 'grad_obj = ', grad_obj
        print 'jac_con = ', jac_con
        grad_obj = grad_obj.tolist()
        jac_con  = jac_con.tolist()
        return grad_obj, jac_con, fail
    
    
    def translate(self,x = None):
        
        # Run the problem just in case
        self.evaluate(x)
        
        # Pull out the inputs and print them
        inpu       = self.optimization_problem.inputs
        print('Design Variable Table:\n')
        print inpu
        
        # Pull out the constraints
        const       = self.optimization_problem.constraints
        const_vals  = self.all_constraints(x)
        const_scale = help_fun.unscale_const_values(const,const_vals)
        
        # Make a new table
        const_table = np.array(const)
        const_table = np.insert(const_table,1,const_scale,axis=1)

        print('\nConstraint Table:\n')
        print const_table
        
        return inpu,const_table
                               
        
    
 
