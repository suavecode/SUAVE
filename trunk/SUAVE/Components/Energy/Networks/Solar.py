## @ingroup Components-Energy-Networks
# Solar.py
# 
# Created:  Jun 2014, E. Botero
# Modified: Feb 2016, T. MacDonald 
#           Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
from SUAVE.Components.Propulsors.Propulsor import Propulsor

from SUAVE.Core import Data , Units

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Solar(Propulsor):
    """ A solar powered system with batteries and maximum power point tracking.
        
        This network adds an extra unknowns to the mission, the torque matching between motor and propeller.
    
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
        self.solar_flux                = None
        self.solar_panel               = None
        self.motor                     = None
        self.propeller                 = None
        self.esc                       = None
        self.avionics                  = None
        self.payload                   = None
        self.solar_logic               = None
        self.battery                   = None
        self.nacelle_diameter          = None
        self.engine_length             = None
        self.number_of_engines         = None
        self.tag                       = 'Solar'
        self.use_surrogate             = False
        self.generative_design_minimum = 0
    
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
                solar_flux           [watts/m^2] 
                rpm                  [radians/sec]
                current              [amps]
                battery_draw         [watts]
                battery_energy       [joules]
                motor_torque         [N-M]
                propeller_torque     [N-M]
    
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
        battery.current_energy = conditions.propulsion.battery_energy
        
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
        propeller.inputs.omega =  motor.outputs.omega
        
        # step 6
        F, Q, P, Cplast ,  outputs  , etap   = propeller.spin(conditions)
     
        # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
        eta = conditions.propulsion.throttle[:,0,None]
        P[eta>1.0] = P[eta>1.0]*eta[eta>1.0]
        F[eta>1.0] = F[eta>1.0]*eta[eta>1.0]
        
        # link
        propeller.outputs = outputs
        
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
        solar_logic.logic(conditions,numerics)
        
        # link
        battery.inputs = solar_logic.outputs
        battery.energy_calc(numerics)
        
        #Pack the conditions for outputs
        a                                        = conditions.freestream.speed_of_sound
        R                                        = propeller.tip_radius           
        rpm                                      = motor.outputs.omega*60./(2.*np.pi)
        current                                  = solar_logic.inputs.currentesc
        battery_draw                             = battery.inputs.power_in 
        battery_energy                           = battery.current_energy
        voltage_open_circuit                     = battery.voltage_open_circuit
        voltage_under_load                       = battery.voltage_under_load           
        
        conditions.propulsion.solar_flux           = solar_flux.outputs.flux  
        conditions.propulsion.rpm                  = rpm
        conditions.propulsion.voltage_open_circuit = voltage_open_circuit
        conditions.propulsion.voltage_under_load   = voltage_under_load  
        conditions.propulsion.current              = current
        conditions.propulsion.battery_draw         = battery_draw
        conditions.propulsion.battery_energy       = battery_energy
        conditions.propulsion.motor_torque         = motor.outputs.torque
        conditions.propulsion.propeller_torque     = Q        
        conditions.propulsion.propeller_tip_mach   = (R*rpm*Units.rpm)/a
        
        #Create the outputs
        F                                        = num_engines * F * [1,0,0]   
        F_mag                                    = np.atleast_2d(np.linalg.norm(F, axis=1))   
        conditions.propulsion.disc_loading       = (F_mag.T)/ (num_engines*np.pi*(R)**2) # N/m^2               
        conditions.propulsion.power_loading      = (F_mag.T)/(P)  # N/W                  
        
        mdot = state.ones_row(1)*0.0

        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot

        return results
    
    
    def unpack_unknowns(self,segment):
        """ This is an extra set of unknowns which are unpacked from the mission solver and send to the network.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            state.unknowns.propeller_power_coefficient [None]
    
            Outputs:
            state.conditions.propulsion.propeller_power_coefficient [None]
    
            Properties Used:
            N/A
        """       
        
        # Here we are going to unpack the unknowns (Cp) provided for this network
        segment.state.conditions.propulsion.propeller_power_coefficient = segment.state.unknowns.propeller_power_coefficient

        return
    
    def residuals(self,segment):
        """ This packs the residuals to be send to the mission solver.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            state.conditions.propulsion:
                motor_torque                          [N-m]
                propeller_torque                      [N-m]
            
            Outputs:
            None
    
            Properties Used:
            None
        """  
        
        # Here we are going to pack the residuals from the network
        
        # Unpack
        q_motor   = segment.state.conditions.propulsion.motor_torque
        q_prop    = segment.state.conditions.propulsion.propeller_torque
        
        # Return the residuals
        segment.state.residuals.network[:,0] = q_motor[:,0] - q_prop[:,0]
        
        return        
            
    __call__ = evaluate_thrust
