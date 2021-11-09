## @ingroup Components-Energy-Networks
# Solar_Low_Fidelity.py
# 
# Created:  Jun 2014, E. Botero
# Modified: Feb 2016, T. MacDonald
#           Mar 2020, M. Clarke
#           Aug 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np
from .Network import Network
from SUAVE.Methods.Power.Battery.pack_battery_conditions import pack_battery_conditions

from SUAVE.Core import Data , Units

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Solar_Low_Fidelity(Network):
    """ A solar powered system with batteries and maximum power point tracking.
        
        This network adds an extra unknowns to the mission, the torque matching between motor and propeller.
    
        Assumptions:
        This model uses the low fidelity motor and propeller model to speed computation.
        This reduces accuracy as it is assuming a simple efficiency
        
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
        self.solar_flux        = None
        self.solar_panel       = None
        self.motor             = None
        self.propeller         = None
        self.esc               = None
        self.avionics          = None
        self.payload           = None
        self.solar_logic       = None
        self.battery           = None 
        self.engine_length     = None
        self.number_of_engines = None
        self.tag               = 'Solar_Low_Fidelity'
    
    # manage process with a driver function
    def evaluate_thrust(self,state):
        """ Calculate thrust given the current state of the vehicle
    
            Assumptions:
            Caps the throttle at 110% and linearly interpolates thrust off that
    
            Source:
            N/A
    
            Inputs:
            state [state()]
    
            Outputs:
            results.thrust_force_vector [newtons]
            results.vehicle_mass_rate   [kg/s]
            conditions.propulsion:
                solar_flux              [watts/m^2] 
                rpm                     [radians/sec]
                current                 [amps]
                battery_power_draw      [watts]
                battery_energy          [joules]
                
            Properties Used:
            Defaulted values
        """           
    
        # unpack
        conditions  = state.conditions
        numerics    = state.numerics        
        solar_flux  = self.solar_flux
        solar_panel = self.solar_panel
        motor       = self.motor
        propeller   = self.propeller
        esc         = self.esc
        avionics    = self.avionics
        payload     = self.payload
        solar_logic = self.solar_logic
        battery     = self.battery
        num_engines = self.number_of_engines
        
        # Set battery energy
        battery.current_energy           = conditions.propulsion.battery_energy
        battery.pack_temperature         = conditions.propulsion.battery_pack_temperature
        battery.cell_charge_throughput   = conditions.propulsion.battery_cell_charge_throughput     
        battery.age                      = conditions.propulsion.battery_cycle_day          
        battery.R_growth_factor          = conditions.propulsion.battery_resistance_growth_factor
        battery.E_growth_factor          = conditions.propulsion.battery_capacity_fade_factor  
        
        # step 1
        solar_flux.solar_radiation(conditions)
        # link
        solar_panel.inputs.flux = solar_flux.outputs.flux
        # step 2
        solar_panel.power()
        # link
        solar_logic.inputs.powerin = solar_panel.outputs.power
        # step 3
        solar_logic.voltage()
        # link
        esc.inputs.voltagein =  solar_logic.outputs.system_voltage
        # Step 4
        esc.voltageout(conditions)
        # link
        motor.inputs.voltage = esc.outputs.voltageout 
        # step 5
        motor.omega(conditions)
        # link
        propeller.inputs.omega  = motor.outputs.omega
        propeller.inputs.torque = motor.outputs.torque
        # step 6
        F, Q, P, Cplast = propeller.spin(conditions)       
            
        # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
        eta = conditions.propulsion.throttle[:,0,None]
        P[eta>1.0] = P[eta>1.0]*eta[eta>1.0]
        F[eta>1.0] = F[eta>1.0]*eta[eta>1.0]
        
        # Run the avionics
        avionics.power()
        # link
        solar_logic.inputs.pavionics =  avionics.outputs.power
        
        # Run the payload
        payload.power()
        # link
        solar_logic.inputs.ppayload = payload.outputs.power
        
        # Run the motor for current
        motor.current(conditions)
        # link
        esc.inputs.currentout =  motor.outputs.current
        
        # Run the esc
        esc.currentin(conditions)
        # link
        solar_logic.inputs.currentesc  = esc.outputs.currentin*num_engines
        solar_logic.inputs.volts_motor = esc.outputs.voltageout 
        
        # Adjust power usage for magic thrust
        solar_logic.inputs.currentesc[eta>1.0] = solar_logic.inputs.currentesc[eta>1.0]*eta[eta>1.0]
        #
        solar_logic.logic(conditions,numerics)
        # link
        battery.inputs = solar_logic.outputs
        battery.energy_calc(numerics)
        
        # Calculate avionics and payload power
        avionics_payload_power = avionics.outputs.power + payload.outputs.power        
        
        # Pack the conditions for outputs 
        a                                        = conditions.freestream.speed_of_sound
        R                                        = propeller.tip_radius        
        rpm                                      = motor.outputs.omega / Units.rpm         
        
        battery.inputs.current                   = solar_logic.inputs.currentesc
        pack_battery_conditions(conditions,battery,avionics_payload_power,P)     
        
        conditions.propulsion.solar_flux         = solar_flux.outputs.flux  
        conditions.propulsion.propeller_rpm      = rpm
        conditions.propulsion.propeller_tip_mach = (R*rpm*Units.rpm)/a
        
     
        
        #Create the outputs
        F                                        = num_engines * F * [1,0,0]      
        mdot                                     = state.conditions.ones_row(1)*0.0
        F_mag                                    = np.atleast_2d(np.linalg.norm(F, axis=1))  
        conditions.propulsion.disc_loading       = (F_mag.T)/ (num_engines*np.pi*(R)**2)   # N/m^2                 
        conditions.propulsion.power_loading      = (F_mag.T)/(P )  # N/W                        
    
        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot
    
        return results
            
    __call__ = evaluate_thrust