## @ingroup Components-Energy-Networks
# Internal_Combustion_Propeller_Constant_Speed.py
# 
# Created:  Sep 2016, E. Botero
# Modified: Apr 2018, M. Clarke 
#           Mar 2020, M. Clarke 
#           Apr 2021, M. Clarke
#           Jul 2021, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np
from .Network import Network
from SUAVE.Components.Physical_Component import Container
from SUAVE.Core import Data, Units

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Networks
class Internal_Combustion_Propeller_Constant_Speed(Network):
    """ An internal combustion engine with a constant speed propeller.
    
        Assumptions:
        0.5 Throttle corresponds to 0 propeller pitch. Less than 0.5 throttle implies negative propeller pitch.
        
        Source:
        None
    """      
    def __defaults__(self):
        """ This sets the default values for the network to function.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            N/A
        """        
        self.engines                      = Container()
        self.propellers                   = Container()
        self.engine_length                = None
        self.number_of_engines            = None
        self.rated_speed                  = 0.0 
        self.rotor_group_indexes          = [0]
        self.motor_group_indexes          = [0]
        self.active_propulsor_groups      = [True]
        
    
    # manage process with a driver function
    def evaluate_thrust(self,state):
        """ Calculate thrust given the current state of the vehicle
    
            Assumptions:
    
            Source:
            N/A
    
            Inputs:
            state [state()]
    
            Outputs:
            results.thrust_force_vector [newtons]
            results.vehicle_mass_rate   [kg/s]
            conditions.propulsion:
                rpm                     [radians/sec]
                torque                  [N-M]
                power                   [W]
    
            Properties Used:
            Defaulted values
        """           
        # unpack
        conditions              = state.conditions
        engines                 = self.engines
        propellers              = self.propellers 
        rpm                     = conditions.propulsion.rpm 
        rotor_group_indexes     = self.rotor_group_indexes
        motor_group_indexes     = self.motor_group_indexes
        active_propulsor_groups = self.active_propulsor_groups
        
        # Unpack conditions
        a = conditions.freestream.speed_of_sound        

        # How many evaluations to do 
        unique_rotor_groups,factors = np.unique(rotor_group_indexes, return_counts=True)
        unique_motor_groups,factors = np.unique(motor_group_indexes, return_counts=True)
        if (unique_rotor_groups == unique_motor_groups).all(): # rotors and motors are paired 
            n_evals = len(unique_rotor_groups)
            rotor_indexes = unique_rotor_groups 
            factor        = factors
        else:
            n_evals = len(rotor_group_indexes)
            rotor_indexes = rotor_group_indexes 
            factor        = np.ones_like(motor_group_indexes) 
 
            
        # Setup conditions
        ones_row = conditions.ones_row
        for i in range(n_evals):         
            # Setup the conditions        
            conditions.propulsion['propulsor_group_' + str(i)].rotor.disc_loading         = ones_row(n_evals)
            conditions.propulsion['propulsor_group_' + str(i)].rotor.power_loading        = ones_row(n_evals)
            conditions.propulsion['propulsor_group_' + str(i)].rotor.torque               = ones_row(n_evals)
            conditions.propulsion['propulsor_group_' + str(i)].rotor.tip_mach             = ones_row(n_evals)
            conditions.propulsion['propulsor_group_' + str(i)].combustion_engine_throttle = ones_row(n_evals)
            
        # Setup numbers for iteration
        total_thrust        = 0. * state.ones_row(3)
        total_power         = 0.
        mdot                = 0.
        

        # Iterate over motor/rotors
        for ii in range(n_evals):
            if active_propulsor_groups[ii]:  
                engine_key = list(engines.keys())[ii]
                engine     = self.engines[engine_key]                
                rotor_key = list(propellers.keys())[rotor_indexes[ii]] 
                rot      = propellers[rotor_key] 
                       
    
                # Run the propeller to get the power
                rot.inputs.pitch_command = conditions.propulsion.throttle - 0.5
                rot.inputs.omega         = rpm
                
                # step 4
                F, Q, P, Cp, outputs, etap = rot.spin(conditions)
                
                # Run the engine to calculate the throttle setting and the fuel burn
                engine.inputs.power = P
                engine.calculate_throttle(conditions)
    
                # Create the outputs
                R                   = rot.tip_radius
                mdot                = mdot + engine.outputs.fuel_flow_rate * factor
                F_mag               = np.atleast_2d(np.linalg.norm(F, axis=1))  
                engine_throttle     = engine.outputs.throttle  
                total_thrust        = total_thrust + F * factor
                total_power         = total_power  + P * factor            
    
                # Pack the conditions 
                conditions.propulsion['propulsor_group_' + str(ii)].throttle                   = conditions.propulsion.throttle
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.torque               = Q
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.tip_mach             = (R*rpm*Units.rpm)/a
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.disc_loading         = (F_mag)/(np.pi*(R**2)) # N/m^2                  
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.power_loading        = (F_mag)/(P)      # N/W            
                conditions.propulsion['propulsor_group_' + str(ii)].combustion_engine_throttle = engine_throttle
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.efficiency           = etap
                
                
                conditions.noise.sources.rotors[rot.tag]    = outputs
            
        # Create the outputs
        conditions.propulsion.propulsor_group_0.power = total_power
        
        results = Data()
        results.thrust_force_vector       = F
        results.vehicle_mass_rate         = mdot 
        
        return results

    __call__ = evaluate_thrust
