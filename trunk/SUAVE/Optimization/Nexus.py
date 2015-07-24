# Nexus.py
# 
# Created:  Jul 2015, E. Botero 
# Modified:  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE
from SUAVE.Core import Data
from copy import deepcopy
import helper_functions as help_fun
import numpy as np

# ----------------------------------------------------------------------
#  Nexus Class
# ----------------------------------------------------------------------
    
class Nexus(Data):
    
    def __defaults__(self):
        self.vehicle_configurations       = SUAVE.Components.Configs.Config.Container()
        self.analyses                     = SUAVE.Analyses.Analysis.Container()
        self.missions                     = None
        self.procedure                    = None
        self.results                      = SUAVE.Analyses.Results()
        self.optimization_problem         = None
        self.last_inputs                  = None
        self.evaluation_count             = 0
    
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
                
        ## Store to cache
        self.last_inputs = deepcopy(self.optimization_problem.inputs)
          
    
    def objective(self,x = None):
        
        self.evaluate(x)
        
        aliases     = self.optimization_problem.aliases
        objective   = self.optimization_problem.objective
        results     = self.results
    
        objective_value  = help_fun.get_values(results,objective,aliases)  
        scaled_objective = help_fun.scale_obj_values(objective,objective_value)
        
        return scaled_objective
    
    def inequality_constraint(self,x = None):
        
        self.evaluate(x)
        
        aliases     = self.optimization_problem.aliases
        constraints = self.optimization_problem.constraints
        results     = self.results
    
        constraint_values = help_fun.get_values(results,constraints,aliases) 
        scaled_constraints = help_fun.scale_const_values(constraints,constraint_values)
    
        return scaled_constraints  
    
    def equality_constraint(self,x = None):
        
        self.evaluate(x)
        
        aliases     = self.optimization_problem.aliases
        constraints = self.optimization_problem.constraints
        results     = self.results

        constraint_values = help_fun.get_values(results,constraints,aliases)  
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
        
        vehicle = help_fun.set_values(vehicle,inputs,converted_values,aliases)     
    
    def constraints_individual(self,x):
        pass     

    
 
