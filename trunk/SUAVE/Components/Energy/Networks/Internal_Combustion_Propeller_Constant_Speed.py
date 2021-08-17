## @ingroup Components-Energy-Networks
# Internal_Combustion_Propeller.py
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
        None
        
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
        self.engines              = Container()
        self.propellers           = Container()
        self.engine_length        = None
        self.number_of_engines    = None
        self.rated_speed          = 0.0
        self.identical_propellers = True
        
    
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
                rpm                  [radians/sec]
                propeller_torque     [N-M]
                power                [W]
    
            Properties Used:
            Defaulted values
        """           
        # unpack
        conditions  = state.conditions
        engines     = self.engines
        propellers  = self.propellers
        num_engines = self.number_of_engines
        rpm         = conditions.propulsion.rpm 
        
        # Unpack conditions
        a = conditions.freestream.speed_of_sound        
        
        # How many evaluations to do
        if self.identical_propellers:
            n_evals = 1
            factor  = num_engines*1
        else:
            n_evals = int(num_engines)
            factor  = 1.
            
        # Setup conditions
        ones_row = conditions.ones_row
        conditions.propulsion.disc_loading               = ones_row(n_evals)
        conditions.propulsion.power_loading              = ones_row(n_evals)
        conditions.propulsion.propeller_torque           = ones_row(n_evals)
        conditions.propulsion.propeller_tip_mach         = ones_row(n_evals)
        conditions.propulsion.combustion_engine_throttle = ones_row(n_evals)
            
        # Setup numbers for iteration
        total_thrust        = 0. * state.ones_row(3)
        total_power         = 0.
        mdot                = 0.
        for ii in range(n_evals):  
            
            # Unpack the engine and props
            engine_key = list(engines.keys())[ii]
            prop_key   = list(propellers.keys())[ii]
            engine     = self.engines[engine_key]
            prop       = self.propellers[prop_key]                

            # Run the propeller to get the power
            prop.inputs.pitch_command = conditions.propulsion.throttle
            prop.inputs.omega         = rpm
            
            # step 4
            F, Q, P, Cp, outputs, etap = prop.spin(conditions)
            
            # Run the engine to calculate the throttle setting and the fuel burn
            engine.inputs.power = P
            engine.calculate_throttle(conditions)

            # Create the outputs
            R                   = prop.tip_radius
            mdot                = mdot + engine.outputs.fuel_flow_rate * factor
            F_mag               = np.atleast_2d(np.linalg.norm(F, axis=1))  
            engine_throttle     = engine.outputs.throttle
            total_thrust        = total_thrust + F * factor
            total_power         = total_power  + P * factor            

           # Pack the conditions
            conditions.propulsion.propeller_torque[:,ii]     = Q[:,0]
            conditions.propulsion.propeller_tip_mach[:,ii]   = (R*rpm[:,0]*Units.rpm)/a[:,0]
            conditions.propulsion.disc_loading[:,ii]         = (F_mag[:,0])/(np.pi*(R**2)) # N/m^2                  
            conditions.propulsion.power_loading[:,ii]        = (F_mag[:,0])/(P[:,0])      # N/W            
            conditions.propulsion.combustion_engine_throttle = engine_throttle
            conditions.propulsion.propeller_efficiency       = etap[:,0]
            
            
            conditions.noise.sources.propellers[prop.tag]    = outputs
        
        # Create the outputs
        conditions.propulsion.power = total_power
        
        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot
        
        return results

    __call__ = evaluate_thrust
