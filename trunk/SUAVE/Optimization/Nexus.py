## @ingroup Optimization
# Nexus.py
# 
# Created:  Jul 2015, E. Botero 
# Modified: Feb 2016, M. Vegh
#           Apr 2017, T. MacDonald
#           Jul 2020, M. Clarke
#           May 2021, E. Botero 
#           Jun 2022, E. Botero 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE 
from SUAVE.Core import Data
from SUAVE.Analyses import Process
from copy import deepcopy
from . import helper_functions as help_fun
import numpy as np

from jax import jacfwd, jit, grad
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
import jax
from functools import partial

# ----------------------------------------------------------------------
#  Nexus Class
# ----------------------------------------------------------------------

## @ingroup Optimization
@register_pytree_node_class
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
        self.vehicle_configurations = None
        self.analyses               = SUAVE.Analyses.Analysis.Container()
        self.missions               = None
        self.procedure              = Process()
        self.results                = Data()
        self.summary                = Data()
        self.fidelity_level         = 1.
        self.last_inputs            = Data()
        self.last_fidelity          = None
        self.last_jacobian_inputs   = Data()
        self.last_jacobians         = None
        self.evaluation_count       = 0.
        self.force_evaluate         = False
        self.hard_bounded_inputs    = False
        self.use_jax_derivatives    = False
        self.jitable                = False
        self.static_keys            = ['last_jacobians']

        self.optimization_problem             = Data()
        self.optimization_problem.inputs      = None     
        self.optimization_problem.objective   = None
        self.optimization_problem.constraints = None
        self.optimization_problem.aliases     = None
        
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
        
        self = unpack_inputs(self,x)
        self = really_evaluate(self)
            
        return self

    
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
                
        self.evaluation_count += 1.
        
        for key,step in self.procedure.items():
            if hasattr(step,'evaluate'):
                self = step.evaluate(self)
            else:
                self = step(self)
        
        # Store to cache
        self.last_inputs   = deepcopy(self.optimization_problem.inputs)
        self.last_fidelity = self.fidelity_level
        
        return self
    
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
        
        if x is None:
            input_array = self.optimization_problem.inputs.pack_array()
            x = input_array[0::5]/input_array[3::5]        
        
        if self.jitable:
            scaled_objective, problem = jit_nexus_objective_wrapper(x,self)
            self.update(problem,hard=True)
        else:
            scaled_objective, self = self._objective(x)
        

        return scaled_objective
    
    def _objective(self,x):
        """ Really retrieve the objective value for your function
    
            Assumptions:
            Your procedure must contain totally jaxable code, not all of SUAVE is jax-ed
    
            Source:
            N/A
    
            Inputs:
            x       [vector]
    
            Outputs:
            scaled_objective [float]
    
            Properties Used:
            None
        """        
        
        self  = self.evaluate(x)
        
        aliases     = self.optimization_problem.aliases
        objective   = self.optimization_problem.objective
    
        objective_value  = help_fun.get_values(self,objective,aliases)  
        scaled_objective = help_fun.scale_obj_values(objective,objective_value)
        
        return scaled_objective, self        
        

    
    def grad_objective(self,x = None):
        """Retrieve the objective gradient for your function using JAX
    
            Assumptions:
            Your procedure must contain totally jaxable code, not all of SUAVE is jax-ed
    
            Source:
            N/A
    
            Inputs:
            x       [vector]
    
            Outputs:
            scaled_objective [float]
    
            Properties Used:
            None
        """
        if x is None:
            input_array = self.optimization_problem.inputs.pack_array()
            x = input_array[0::5]/input_array[3::5]       
        
        if self.jitable:
            grad_function = jit_jac_nexus_objective_wrapper
            grad, problem = grad_function(x,self)  
            self.update(problem,hard=True)
        else:
            grad_function = jac_nexus_objective_wrapper
            grad, problem = grad_function(x,self)  
            self.update(problem,hard=True)

        return grad
        
    
    
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
    
    
    def jacobian_inequality_constraint(self,x = None):
        """Retrieve the inequality constraint jacobian for your function using JAX
    
            Assumptions:
            Your procedure must contain totally jaxable code, not all of SUAVE is jax-ed
    
            Source:
            N/A
    
            Inputs:
            x       [vector]
    
            Outputs:
            scaled_objective [float]
    
            Properties Used:
            None
        """
        
        
        raise NotImplementedError
        
    
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
    
    
    def jacobian_equality_constraint(self,x = None):
        """Retrieve the equality constraint jacobian for your function using JAX
    
            Assumptions:
            Your procedure must contain totally jaxable code, not all of SUAVE is jax-ed
    
            Source:
            N/A
    
            Inputs:
            x       [vector]
    
            Outputs:
            scaled_objective [float]
    
            Properties Used:
            None
        """
        
        
        raise NotImplementedError
        
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
        
        if x is None:
            input_array = self.optimization_problem.inputs.pack_array()
            x = input_array[0::5]/input_array[3::5]       
            
        if self.jitable:
            scaled_constraints, problem = jit_nexus_all_constraint_wrapper(x,self)
            self.update(problem,hard=True)
        else:
            scaled_constraints, self = self._all_constraints(x)


        return scaled_constraints     
    
    
    def _all_constraints(self,x):
        """Really returns both the inequality and equality constraint values for your function
    
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
        
        self = self.evaluate(x)
        
        aliases     = self.optimization_problem.aliases
        constraints = self.optimization_problem.constraints
    
        constraint_values  = help_fun.get_values(self,constraints,aliases) 
        scaled_constraints = help_fun.scale_const_values(constraints,constraint_values) 

        return scaled_constraints, self        
    
    
    def jacobian_all_constraints(self,x = None):
        """Retrieve the all constraints jacobian for your function using JAX
    
            Assumptions:
            Your procedure must contain totally jaxable code, not all of SUAVE is jax-ed
    
            Source:
            N/A
    
            Inputs:
            x       [vector]
    
            Outputs:
            scaled_objective [float]
    
            Properties Used:
            None
        """
        if x is None:
            input_array = self.optimization_problem.inputs.pack_array()
            x = input_array[0::5]/input_array[3::5]    
        
        if self.jitable:
            grad_function = jit_jac_nexus_all_constraint_wrapper
            grad, problem = grad_function(x,self)  
            self.update(problem,hard=True)
        else:
            grad_function = jac_nexus_all_constraint_wrapper
            grad, problem = grad_function(x,self)  
            self.update(problem,hard=True)
                
        return grad
    
    
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
         
        #Scale the inputs if given
        inputs = deepcopy(self.optimization_problem.inputs)
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
        
        return self
               

    
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
        raise NotImplementedError
    
    
    def convert_problem_arrays(self):
        """ Go through each part of the problem and convert to a Data structure instead of an np array
    
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

        # unpack
        inputs = self.optimization_problem.inputs
        obj    = self.optimization_problem.objective
        cons   = self.optimization_problem.constraints
        alias  = np.array(self.optimization_problem.aliases,dtype=object)
        
        # split the inputs
        in_nam  = inputs[:,0]
        in_arr  = inputs[:,1:].astype(float)
        inputs  = Data(dict(zip(in_nam,in_arr)))
        
        # do the objectives
        obj_nam = obj[:,0]
        obj_arr = obj[:,1:].astype(float)
        obj     = Data(dict(zip(obj_nam,obj_arr)))
        
        # constraints
        con_nam     = cons[:,0]
        cons[cons=='>']  =  1.
        cons[cons=='>='] =  1.
        cons[cons=='<']  = -1.
        cons[cons=='<='] = -1.
        cons[cons=='=']  =  0.
        con_arr = cons[:,1:].astype(float)
        cons        = Data(dict(zip(con_nam,con_arr)))
        
        # aliases
        ali_nam = alias[:,0]
        ali_arr = alias[:,1].flatten()
        alias   = Data(dict(zip(ali_nam,ali_arr)))
        
        # pack
        self.optimization_problem.inputs      = inputs
        self.optimization_problem.objective   = obj
        self.optimization_problem.constraints = cons 
        self.optimization_problem.aliases     = alias
        
        
        

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
    
    
    def add_array_inputs(self, full_path, lower_bound, upper_bound, scale=1., star=None):
        
        # go to the full path and figure out the array shape
        array  = eval('self.'+full_path)
        shape  = array.shape
        ndim   = array.ndim
        size   = array.size
        name   = full_path.split('.')[-1]
        
        # create an input array
        new_inputs  = np.zeros((size,6),dtype=object)
        # create an alias list
        new_aliases = []
                
        # loop over the array dimension by dimension
        for ii,val in enumerate(array.flatten()):
            # setup the inputs
            alias_name       = name+'_'+str(ii)
            new_inputs[ii,:] = np.array([alias_name,val,lower_bound,upper_bound,scale,1.],dtype=object)
            # setup the aliases            
            new_aliases.append([alias_name,full_path+'['+str(ii)+']'])

        # append the aliases and the inputs to the Nexus
        self.optimization_problem.inputs  = np.vstack((self.optimization_problem.inputs,new_inputs))
        self.optimization_problem.aliases = self.optimization_problem.aliases + new_aliases
        
        
        return self
    
    
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
        obj_scale   = help_fun.unscale_obj_values(obj,obj_val)
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
    

# Below is a list of wrappers that are used fo JAX. They don't require docstrings
        
@jit
def jit_nexus_objective_wrapper(x,nexus):
    return Nexus._objective(nexus,x)

@jit
def jit_nexus_all_constraint_wrapper(x,nexus):
    return Nexus._all_constraints(nexus,x)

@partial(jacfwd,has_aux=True)
def jac_nexus_objective_wrapper(x,nexus):
    return Nexus._objective(nexus,x)

@partial(jacfwd,has_aux=True)
def jac_nexus_all_constraint_wrapper(x,nexus):
    return Nexus._all_constraints(nexus,x)

@jit
@partial(jacfwd,has_aux=True)
def jit_jac_nexus_objective_wrapper(x,nexus):
    return Nexus._objective(nexus,x)

@jit
@partial(jacfwd,has_aux=True)
def jit_jac_nexus_all_constraint_wrapper(x,nexus):
    return Nexus._all_constraints(nexus,x)

def really_evaluate(nexus):
    return Nexus._really_evaluate(nexus)

def unpack_inputs(nexus,x):
    return Nexus.unpack_inputs(nexus,x)