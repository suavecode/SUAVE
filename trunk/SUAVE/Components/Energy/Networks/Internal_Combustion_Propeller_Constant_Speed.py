## @ingroup Components-Energy-Networks
# Internal_Combustion_Propeller.py
# 
# Created:  Sep 2016, E. Botero
# Modified: Apr 2018, M. Clarke 
#           Mar 2020, M. Clarke 
#           Apr 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
from SUAVE.Components.Propulsors.Propulsor import Propulsor
from SUAVE.Core import Data, Units

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Networks
class Internal_Combustion_Propeller_Constant_Speed(Propulsor):
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
        self.engine            = None
        self.propeller         = None
        self.engine_length     = None
        self.number_of_engines = None
        self.thrust_angle      = 0.0
        self.rated_speed       = 0.0
    
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
        engine      = self.engine
        propeller   = self.propeller
        num_engines = self.number_of_engines
        rpm       = conditions.propulsion.rpm 

        
        # Run the propeller to get the power
        propeller.pitch_command = conditions.propulsion.throttle
        propeller.inputs.omega  = rpm
        propeller.thrust_angle  = self.thrust_angle
        
        # step 4
        F, Q, P, Cp ,  outputs  , etap  = propeller.spin(conditions)
        
        # Run the engine to calculate the throttle setting and the fuel burn
        
        # Run the engine
        engine.inputs.power = P
        engine.calculate_throttle(conditions)
        mdot            = engine.outputs.fuel_flow_rate
        engine_throttle = engine.outputs.throttle

    
        # Pack the conditions for outputs
        a                                        = conditions.freestream.speed_of_sound
        R                                        = propeller.tip_radius   
          
        conditions.propulsion.propeller_torque   = Q
        conditions.propulsion.power              = P
        conditions.propulsion.propeller_tip_mach = (R*rpm*Units.rpm)/a
        
        # noise        
        outputs.number_of_engines                = num_engines
        conditions.noise.sources.propeller       = outputs 
        
        # Create the outputs
        F                                                = num_engines* F * [np.cos(self.thrust_angle),0,-np.sin(self.thrust_angle)]  
        F_mag                                            = np.atleast_2d(np.linalg.norm(F, axis=1))   
        conditions.propulsion.disc_loading               = (F_mag.T)/ (num_engines*np.pi*(R/Units.feet)**2)   # N/m^2                      
        conditions.propulsion.power_loading              = (F_mag.T)/(P)    # N/W       
        conditions.propulsion.combustion_engine_throttle = engine_throttle
        
        
        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot
        
        return results

    __call__ = evaluate_thrust
