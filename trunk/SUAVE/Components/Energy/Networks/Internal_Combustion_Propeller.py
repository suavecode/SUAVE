## @ingroup Components-Energy-Networks
# Internal_Combustion_Propeller.py
# 
# Created:  Sep 2016, E. Botero
# Modified: Apr 2018, M. Clarke 
#           Mar 2020, M. Clarke 

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
class Internal_Combustion_Propeller(Propulsor):
    """ A simple mock up of an internal combustion propeller engine. Tis network adds an extra
        unknowns to the mission, the torque matching between motor and propeller.
    
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
        numerics    = state.numerics
        engine      = self.engine
        propeller   = self.propeller
        rated_speed = self.rated_speed
        num_engines = self.number_of_engines
        
        # Throttle the engine
        eta = conditions.propulsion.throttle[:,0,None]
        engine.speed = rated_speed * eta
        conditions.propulsion.combustion_engine_throttle = eta # keep this 'throttle' on
        
        # Run the engine
        engine.power(conditions)
        power_output = engine.outputs.power
        sfc          = engine.outputs.power_specific_fuel_consumption
        mdot         = engine.outputs.fuel_flow_rate
        torque       = engine.outputs.torque     
        
        # link
        propeller.inputs.omega = engine.speed
        propeller.thrust_angle = self.thrust_angle
        
        # step 4
        F, Q, P, Cp ,  outputs  , etap  = propeller.spin(conditions)
        
        # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
        P[eta>1.0] = P[eta>1.0]*eta[eta>1.0]
        F[eta>1.0] = F[eta>1.0]*eta[eta>1.0]   
        
        # link
        propeller.outputs = outputs
    
        # Pack the conditions for outputs
        a                                        = conditions.freestream.speed_of_sound
        R                                        = propeller.tip_radius   
        rpm                                      = engine.speed / Units.rpm
          
        conditions.propulsion.rpm                = rpm
        conditions.propulsion.propeller_torque   = Q
        conditions.propulsion.power              = P
        conditions.propulsion.propeller_tip_mach = (R*rpm*Units.rpm)/a
        
        # Create the outputs
        F                                        = num_engines* F * [np.cos(self.thrust_angle),0,-np.sin(self.thrust_angle)]  
        F_mag                                    = np.atleast_2d(np.linalg.norm(F, axis=1))   
        conditions.propulsion.disc_loading       = (F_mag.T)/ (num_engines*np.pi*(R/Units.feet)**2)   # N/m^2                      
        conditions.propulsion.power_loading      = (F_mag.T)/(P)    # N/W       
        
        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot
        
        return results
    
    
    def unpack_unknowns(self,segment,state):
        """Unpacks the unknowns set in the mission to be available for the mission.

        Assumptions:
        N/A
        
        Source:
        N/A
        
        Inputs:
        state.conditions.propulsion.propeller_power_coefficient    [Unitless] 
        
        Outputs:
        state.unknowns.propeller_power_coefficient                 [Unitless] 
        
        Properties Used:
        N/A
        """            
        
        # Here we are going to unpack the unknowns (Cp) provided for this network
        state.conditions.propulsion.propeller_power_coefficient = state.unknowns.propeller_power_coefficient
        
        return
    
    def residuals(self,segment,state):
        """ Calculates a residual based on torques 
        
        Assumptions:
        
        Inputs:
            segment.state.conditions.propulsion.
                motor_torque                       [newtom-meters]                 
                propeller_torque                   [newtom-meters] 
        
        Outputs:
            segment.state:
                residuals.network                  [newtom-meters] 
                
        Properties Used:
            N/A
                                
        """         
            
        # Here we are going to pack the residuals (torque,voltage) from the network
        
        # Unpack
        q_motor   = state.conditions.propulsion.motor_torque
        q_prop    = state.conditions.propulsion.propeller_torque
        
        # Return the residuals
        state.residuals.network[:,0] = q_motor[:,0] - q_prop[:,0]    
        
        return    
            
    __call__ = evaluate_thrust
