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
        
        # Throttle the engine
        engine.inputs.speed = state.conditions.propulsion.rpm * Units.rpm
        conditions.propulsion.combustion_engine_throttle = conditions.propulsion.throttle
        
        # Run the engine
        engine.power(conditions)
        power_output = engine.outputs.power
        sfc          = engine.outputs.power_specific_fuel_consumption
        mdot         = engine.outputs.fuel_flow_rate
        torque       = engine.outputs.torque     
        
        # link
        propeller.inputs.omega = state.conditions.propulsion.rpm * Units.rpm
        propeller.thrust_angle = self.thrust_angle
        
        # step 4
        F, Q, P, Cp ,  outputs  , etap  = propeller.spin(conditions)
        
        # Check to see if magic thrust is needed
        eta        = conditions.propulsion.throttle
        P[eta>1.0] = P[eta>1.0]*eta[eta>1.0]
        F[eta>1.0] = F[eta>1.0]*eta[eta>1.0]   
        
        # link
        propeller.outputs = outputs
    
        # Pack the conditions for outputs
        a                                            = conditions.freestream.speed_of_sound
        R                                            = propeller.tip_radius   
        rpm                                          = engine.inputs.speed / Units.rpm
          
        conditions.propulsion.rpm                    = rpm
        conditions.propulsion.propeller_torque       = Q
        conditions.propulsion.power                  = P
        conditions.propulsion.propeller_tip_mach     = (R*rpm*Units.rpm)/a
        conditions.propulsion.propeller_motor_torque = torque
         
        # Create the outputs
        F                                            = num_engines* F * [np.cos(self.thrust_angle),0,-np.sin(self.thrust_angle)]  
        F_mag                                        = np.atleast_2d(np.linalg.norm(F, axis=1))   
        conditions.propulsion.disc_loading           = (F_mag.T)/ (num_engines*np.pi*(R/Units.feet)**2)   # N/m^2                      
        conditions.propulsion.power_loading          = (F_mag.T)/(P)    # N/W       
        
        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot
        
        return results
    
    
    def unpack_unknowns(self,segment):
        """Unpacks the unknowns set in the mission to be available for the mission.

        Assumptions:
        N/A
        
        Source:
        N/A
        
        Inputs:
        state.unknowns.rpm                 [RPM] 
        
        Outputs:
        state.conditions.propulsion.rpm    [RPM] 

        
        Properties Used:
        N/A
        """            
        
        segment.state.conditions.propulsion.rpm = segment.state.unknowns.rpm
        
        return
    
    def residuals(self,segment):
        """ Calculates a residual based on torques 
        
        Assumptions:
        
        Inputs:
            segment.state.conditions.propulsion.
                propeller_motor_torque             [newtom-meters]                 
                propeller_torque                   [newtom-meters] 
        
        Outputs:
            segment.state:
                residuals.network                  [newtom-meters] 
                
        Properties Used:
            N/A
                                
        """         
            
        # Here we are going to pack the residuals (torque,voltage) from the network
        
        # Unpack
        q_motor   = segment.state.conditions.propulsion.propeller_motor_torque          
        q_prop    = segment.state.conditions.propulsion.propeller_torque
        
        # Return the residuals
        segment.state.residuals.network[:,0] = q_motor[:,0] - q_prop[:,0]    
        
        return    
            
    __call__ = evaluate_thrust
