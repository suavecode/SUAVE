## @ingroup Optimization
# Nexus.py
# 
# Created:  Jul 2015, E. Botero 
# Modified: Feb 2016, M. Vegh
#           Apr 2017, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE 
from SUAVE.Core import Data, DataOrdered, Units
from SUAVE.Analyses import Process
from copy import deepcopy
import helper_functions as help_fun
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
        self.optimization_problem   = Data()
        self.fidelity_level         = 1
        self.last_inputs            = None
        self.last_fidelity          = None
        self.evaluation_count       = 0
        
        opt_prob = self.optimization_problem
        opt_prob.objective      = None
        opt_prob.inputs        = None 
        opt_prob.constraints   = None
        opt_prob.aliases       = None
    
        self.optimization_problem        
    
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
        
        ## Check if last call was the same
        #if np.all(self.optimization_problem.inputs==self.last_inputs) \
           #and self.last_fidelity == self.fidelity_level:
            #pass
        #else:
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
        results     = self.results
    
        objective_value  = help_fun.get_values(self,objective,aliases)  
        scaled_objective = help_fun.scale_obj_values(objective,objective_value)
        
        return scaled_objective
    
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
        results     = self.results
    
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
            
        # Convert units
        converted_values = help_fun.convert_values(inputs)
        
        # Set the dictionary
        aliases = self.optimization_problem.aliases
        vehicle = self.vehicle_configurations
        
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
        
        for ii in xrange(0,inplen):
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
                               
                               
    def add_mission_variables(self,mission_key):
        """Make a pretty table view of the problem with objective and constraints at the current inputs
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            x                  [vector]
    
            Outputs:
            input              [array]
            const_table        [array]
    
            Properties Used:
            None
        """             
        
        # unpack
        mis = eval('self.missions.'+mission_key)
        inp = self.optimization_problem.inputs
        con = self.optimization_problem.constraints
        ali = self.optimization_problem.aliases
        
        # add the inputs
        input_count = 0
        for segment in mis.segments:
            
            # Change the mission solve to dummy solver
            print 'Overwriting the solver in ' + segment + ' segment.'
            mis.segments[segment].settings.root_finder = SUAVE.Methods.Missions.Segments.dummy_mission_solver
            
            # Start putting together the inputs
            print 'Adding in the new inputs for ' + segment + '.'
            n_points = mis.segments[segment].state.numerics.number_control_points
            unknown_keys = mis.segments[segment].state.unknowns.keys()
            unknown_keys.remove('tag')  
            len_inputs     = n_points*len(unknown_keys)
            unknown_value  = Data()
            full_unkn_vals = Data()
            for unkn in unknown_keys:
                unknown_value[unkn]  = mis.segments[segment].state.unknowns[unkn]
                full_unkn_vals[unkn] = unknown_value[unkn]*np.ones(n_points)
        
            # Basic construction
            # [Input_###, initial, (-np.inf, np.inf), initial, Units.less]
            initial_values    = full_unkn_vals.pack_array()
            input_len_strings = np.tile('Mission_Input_', len_inputs)
            input_numbers     = np.linspace(1,len_inputs,len_inputs,dtype=np.int16)
            input_names       = np.core.defchararray.add(input_len_strings,np.array(map(str,input_numbers+input_count)))
            bounds            = np.broadcast_to((-np.inf,np.inf),(len_inputs,2))
            units             = np.broadcast_to(Units.less,(len_inputs,))
            new_inputs        = np.reshape(np.tile(np.atleast_2d(np.array([None,None,(None,None),None,None])),len_inputs), (-1, 5))
            
            # Add in the inputs
            new_inputs[:,0]   = input_names 
            new_inputs[:,1]   = initial_values
            new_inputs[:,2]   = bounds.tolist()
            new_inputs[:,3]   = initial_values
            new_inputs[:,4]   = units
            inp               = np.concatenate((new_inputs,inp),axis=0)
            self.optimization_problem.inputs = inp
            
            # Create the equality constraints to the beginning of the constraints
            # all equality constraints are 0, scale 1, and unitless
            new_con = np.reshape(np.tile(np.atleast_2d(np.array([None,None,None,None,None])),len_inputs), (-1, 5))
        
            con_len_strings = np.tile('Residual_', len_inputs)
            con_names       = np.core.defchararray.add(con_len_strings,np.array(map(str,input_numbers+input_count))) 
            equals          = np.broadcast_to('=',(len_inputs,))
            zeros           = np.zeros(len_inputs)
            ones            = np.ones(len_inputs)
            
            # Add in the new constraints
            new_con[:,0]    = con_names
            new_con[:,1]    = equals
            new_con[:,2]    = zeros
            new_con[:,3]    = ones
            new_con[:,4]    = units
            con             = np.concatenate((new_con,con),axis=0)
            self.optimization_problem.constraints = con
            
            # add the corresponding aliases
            # setup the aliases for the inputs
            output_numbers = np.linspace(0,n_points-1,n_points,dtype=np.int16)
            basic_string_con = Data()
            input_string = []
            for unkn in unknown_keys:
                basic_string_con[unkn] = np.tile('missions.' + mission_key + '.segments.' + segment + '.state.unknowns.'+unkn+'[', n_points)
                input_string.append(np.core.defchararray.add(basic_string_con[unkn],np.array(map(str,output_numbers))))
            input_string  = np.ravel(input_string)
            input_string  = np.core.defchararray.add(input_string, np.tile(']',len_inputs))
            input_aliases = np.reshape(np.tile(np.atleast_2d(np.array((None,None))),len_inputs), (-1, 2))
                                          
            input_aliases[:,0] = input_names
            input_aliases[:,1] = input_string
            
            
            # setup the aliases for the residuals
            basic_string_res = np.tile('missions.' + mission_key + '.state.residuals.' + segment + '.pack_auto()[', len_inputs)
            residual_string  = np.core.defchararray.add(basic_string_res,np.array(map(str,input_numbers-1)))
            residual_string  = np.core.defchararray.add(residual_string, np.tile(']',len_inputs))
            residual_aliases = np.reshape(np.tile(np.atleast_2d(np.array((None,None))),len_inputs), (-1, 2))
            
            residual_aliases[:,0] = con_names
            residual_aliases[:,1] = residual_string
            
            # Put all the aliases in!
            for ii in xrange(len_inputs):
                ali.append(residual_aliases[ii].tolist())
                ali.append(input_aliases[ii].tolist())
                
            # The mission needs the state expanded now
            mis.segments[segment].process.initialize.expand_state(mis.segments[segment],mis.segments[segment].state)
            
            # Update the count of inputs
            input_count = input_count+input_numbers[-1]
            
        return self