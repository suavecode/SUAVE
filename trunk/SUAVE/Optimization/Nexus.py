## @ingroup Optimization
# Nexus.py
# 
# Created:  Jul 2015, E. Botero 
# Modified: Feb 2016, M. Vegh
#           Apr 2017, T. MacDonald
#           Jul 2020, M. Clarke
#           May 2021, E. Botero 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE 
from SUAVE.Core import Data, DataOrdered
from SUAVE.Analyses import Process
from copy import deepcopy
from . import helper_functions as help_fun
import numpy as np

# ----------------------------------------------------------------------
#  Nexus Class
# ----------------------------------------------------------------------

## @ingroup Optimization
class Nexus(Data):
    """noun (plural same or nexuses)
        -a connection or series of connections linking two or more things
        -a connected group or series: a nexus of ideas.
        -the central and most important point or place
        
        This is the class that makes optimization possible. We put all the data and functions together to make
        your future dreams come true.
        
        Assumptions:
        You like SUAVE
        
        Source:
        Oxford English Dictionary
    """    
    
    def __defaults__(self):
        """This sets the default values.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """          
        self.vehicle_configurations = SUAVE.Components.Configs.Config.Container()
        self.analyses               = SUAVE.Analyses.Analysis.Container()
        self.missions               = None
        self.procedure              = Process()
        self.results                = Data()
        self.summary                = Data()
        self.optimization_problem   = None
        self.fidelity_level         = 1
        self.last_inputs            = None
        self.last_fidelity          = None
        self.evaluation_count       = 0
        self.force_evaluate         = False
        self.hard_bounded_inputs    = False
    
    def evaluate(self,x = None):
        """This function runs the problem you setup in SUAVE.
            If the last time you ran this the inputs were the same, a cache is used.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            x       [vector]
    
            Outputs:
            None
    
            Properties Used:
            None
        """          
        
        self.unpack_inputs(x)
        
        # Check if last call was the same
        if np.all(self.optimization_problem.inputs==self.last_inputs) \
           and self.last_fidelity == self.fidelity_level \
           and self.force_evaluate == False:
            pass
        else:
            self._really_evaluate()
        
    
    def _really_evaluate(self):
        """Tricky little function you're not supposed to use. Doesn't check if the last inputs were already run.
            This steps through like a process through the nexus, and stores the results.
    
            Assumptions:
            Doesn't set values!
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """          
        
        nexus = self
        
        self.evaluation_count += 1
        
        for key,step in nexus.procedure.items():
            if hasattr(step,'evaluate'):
                self = step.evaluate(nexus)
            else:
                nexus = step(nexus)
            self = nexus
                
        # Store to cache
        self.last_inputs   = deepcopy(self.optimization_problem.inputs)
        self.last_fidelity = self.fidelity_level
          
    
    def objective(self,x = None):
        """Retrieve the objective value for your function
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            x       [vector]
    
            Outputs:
            scaled_objective [float]
    
            Properties Used:
            None
        """           
    
        self.evaluate(x)
        
        aliases     = self.optimization_problem.aliases
        objective   = self.optimization_problem.objective
    
        objective_value  = help_fun.get_values(self,objective,aliases)  
        scaled_objective = help_fun.scale_obj_values(objective,objective_value)
        
        return scaled_objective.astype(np.double) 
    
    def inequality_constraint(self,x = None):
        """Retrieve the inequality constraint values for your function
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            x                  [vector]
    
            Outputs:
            scaled_constraints [vector]
    
            Properties Used:
            None
            """           
        
        self.evaluate(x)
        
        aliases     = self.optimization_problem.aliases
        constraints = self.optimization_problem.constraints
        results     = self.results
        
        # Setup constraints  
        indices = []
        for ii in range(0,len(constraints)):
            if constraints[ii][1]==('='):
                indices.append(ii)        
        iqconstraints = np.delete(constraints,indices,axis=0)
    
        if iqconstraints == []:
            scaled_constraints = []
        else:

            # get constaint values 
            constraint_values = help_fun.get_values(self,iqconstraints,aliases)          
            
            # scale bounds 
            scaled_bnd_constraints  = help_fun.scale_const_bnds(iqconstraints)
            
            # scale constaits 
            scaled_constraints = help_fun.scale_const_values(iqconstraints,constraint_values)
            
            # determine difference between bounds and constaints 
            constraint_evaluations = scaled_constraints  - scaled_bnd_constraints
            
            # coorect constaints based on sign 
            constraint_evaluations[iqconstraints[:,1]=='<'] = -constraint_evaluations[iqconstraints[:,1]=='<']
            
        return constraint_evaluations       
    
    def equality_constraint(self,x = None):
        """Retrieve the equality constraint values for your function
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            x                  [vector]
    
            Outputs:
            scaled_constraints [vector]
    
            Properties Used:
            None
        """         
    
        self.evaluate(x)

        aliases     = self.optimization_problem.aliases
        constraints = self.optimization_problem.constraints
        
        # Setup constraints  
        indices = []
        for ii in range(0,len(constraints)):
            if constraints[ii][1]=='>':
                indices.append(ii)
            elif constraints[ii][1]=='<':
                indices.append(ii)
        eqconstraints = np.delete(constraints,indices,axis=0)
    
        if eqconstraints == []:
            scaled_constraints = []
        else:
            constraint_values  = help_fun.get_values(self,eqconstraints,aliases)
            scaled_constraints = help_fun.scale_const_values(eqconstraints,constraint_values) - help_fun.scale_const_bnds(eqconstraints)

        return scaled_constraints   
        
    def all_constraints(self,x = None):
        """Returns both the inequality and equality constraint values for your function
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            x                  [vector]
    
            Outputs:
            scaled_constraints [vector]
    
            Properties Used:
            None
        """         
        
        self.evaluate(x)
        
        aliases     = self.optimization_problem.aliases
        constraints = self.optimization_problem.constraints
    
        constraint_values  = help_fun.get_values(self,constraints,aliases) 
        scaled_constraints = help_fun.scale_const_values(constraints,constraint_values) 

        return scaled_constraints     
    
    
    def unpack_inputs(self,x = None):
        """Put's the values of the problem in the right place.
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            x                  [vector]
    
            Outputs:
            None
    
            Properties Used:
            None
        """                 
         
        # Scale the inputs if given
        inputs = self.optimization_problem.inputs
        if x is not None:
            inputs = help_fun.scale_input_values(inputs,x)
            
        # Limit the values to the edges
        if self.hard_bounded_inputs:
            inputs = help_fun.limit_input_values(inputs)        
            
        # Convert units
        converted_values = help_fun.convert_values(inputs)

        # Set the dictionary
        aliases = self.optimization_problem.aliases
        
        self    = help_fun.set_values(self,inputs,converted_values,aliases)     

    
    def constraints_individual(self,x = None):
        """Put's the values of the problem in the right place.
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            x                  [vector]
    
            Outputs:
            None
    
            Properties Used:
            None
        """           
        pass     

    def finite_difference(self,x,diff_interval=1e-8):
        """Finite difference gradients and jacobians of the problem.
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            x                  [vector]
            diff_interval      [float]
    
            Outputs:
            grad_obj           [vector]
            jac_con            [array]
    
            Properties Used:
            None
        """           
        
        obj = self.objective(x)
        con = self.all_constraints(x)
        
        inpu  = self.optimization_problem.inputs
        const = self.optimization_problem.constraints
        
        inplen = len(inpu)
        conlen = len(const)
        
        grad_obj = np.zeros(inplen)
        jac_con  = np.zeros((inplen,conlen))
        
        con2 = (con*np.ones_like(jac_con))
        
        for ii in range(0,inplen):
            newx     = np.asarray(x)*1.0
            newx[ii] = newx[ii] + diff_interval
            
            grad_obj[ii]  = self.objective(newx)
            jac_con[ii,:] = self.all_constraints(newx)
        
        grad_obj = (grad_obj - obj)/diff_interval
        
        jac_con = (jac_con - con2).T/diff_interval
        
        grad_obj = grad_obj.astype(float)
        jac_con  = jac_con.astype(float)
        
        return grad_obj, jac_con
    
    
    def translate(self,x = None):
        """Make a pretty table view of the problem with objective and constraints at the current inputs
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            x                  [vector]
    
            Outputs:
            inpu               [array]
            const_table        [array]
    
            Properties Used:
            None
        """         
        
        # Run the problem just in case
        self.evaluate(x)
        
        # Pull out the inputs and print them
        inpu       = self.optimization_problem.inputs
        print('Design Variable Table:\n')
        print(inpu)
        
        # Print the objective value
        obj         = self.optimization_problem.objective
        obj_val     = self.objective(x)
        obj_scale   = help_fun.unscale_const_values(obj,obj_val)
        obj_table   = np.array(obj)
        obj_table   = np.insert(obj_table,1,obj_scale)
        
        print('\nObjective Table:\n')
        print(obj_table)
        
        # Pull out the constraints
        const       = self.optimization_problem.constraints
        const_vals  = self.all_constraints(x)
        const_scale = help_fun.unscale_const_values(const,const_vals)
        
        # Make a new table
        const_table = np.array(const)
        const_table = np.insert(const_table,1,const_scale,axis=1)

        print('\nConstraint Table:\n')
        print(const_table)
        
        return inpu,const_table
                               
        
    
 
