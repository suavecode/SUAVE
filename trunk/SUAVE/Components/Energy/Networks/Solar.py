## @ingroup Components-Energy-Networks
# Solar.py
# 
# Created:  Jun 2014, E. Botero
# Modified: Feb 2016, T. MacDonald 
#           Mar 2020, M. Clarke
#           Jul 2021, E. Botero
#           Aug 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np
from .Network import Network
from SUAVE.Components.Physical_Component import Container
from SUAVE.Methods.Power.Battery.pack_battery_conditions import pack_battery_conditions
from SUAVE.Methods.Power.Battery.append_initial_battery_conditions import append_initial_battery_conditions

from SUAVE.Core import Data , Units

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Solar(Network):
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
        self.motors                    = Container()
        self.propellers                = Container()
        self.esc                       = None
        self.avionics                  = None
        self.payload                   = None
        self.solar_logic               = None
        self.battery                   = None 
        self.engine_length             = None
        self.number_of_engines         = None
        self.tag                       = 'Solar'
        self.use_surrogate             = False
        self.generative_design_minimum = 0
        self.identical_propellers      = True
    
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
                battery_power_draw   [watts]
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
        motors      = self.motors
        propellers  = self.propellers
        esc         = self.esc
        avionics    = self.avionics
        payload     = self.payload
        solar_logic = self.solar_logic
        battery     = self.battery
        num_engines = self.number_of_engines
        
        # Unpack conditions
        a = conditions.freestream.speed_of_sound        
        
        # Set battery energy
        battery.current_energy           = conditions.propulsion.battery_energy
        battery.pack_temperature         = conditions.propulsion.battery_pack_temperature
        battery.cell_charge_throughput   = conditions.propulsion.battery_cell_charge_throughput     
        battery.age                      = conditions.propulsion.battery_cycle_day            
        battery.R_growth_factor          = conditions.propulsion.battery_resistance_growth_factor
        battery.E_growth_factor          = conditions.propulsion.battery_capacity_fade_factor 
        battery.max_energy               = conditions.propulsion.battery_max_aged_energy   
        
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
        esc.inputs.voltagein = solar_logic.outputs.system_voltage
        
        # Step 4
        esc.voltageout(conditions)
        
        # How many evaluations to do
        if self.identical_propellers:
            n_evals = 1
            factor  = num_engines*1
        else:
            n_evals = int(num_engines)
            factor  = 1.
            
        # Setup numbers for iteration
        total_motor_current = 0.
        total_thrust        = 0. * state.ones_row(3)
        total_power         = 0.
        
        # Iterate over motor/props
        for ii in range(n_evals):
            
            # Unpack the motor and props
            motor_key = list(motors.keys())[ii]
            prop_key  = list(propellers.keys())[ii]
            motor     = self.motors[motor_key]
            prop      = self.propellers[prop_key]            
        
            # link
            motor.inputs.voltage = esc.outputs.voltageout
            motor.inputs.propeller_CP = np.atleast_2d(conditions.propulsion.propeller_power_coefficient[:,ii]).T
            
            # step 5
            motor.omega(conditions)
            
            # link
            prop.inputs.omega =  motor.outputs.omega
            
            # step 6
            F, Q, P, Cplast ,  outputs  , etap   = prop.spin(conditions)
         
            # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
            eta = conditions.propulsion.throttle[:,0,None]
            P[eta>1.0] = P[eta>1.0]*eta[eta>1.0]
            F[eta[:,0]>1.0,:] = F[eta[:,0]>1.0,:]*eta[eta[:,0]>1.0,:]
            
            # Run the motor for current
            _ , etam =  motor.current(conditions)         
            
            # Conditions specific to this instantation of motor and propellers
            R                   = prop.tip_radius
            rpm                 = motor.outputs.omega / Units.rpm
            F_mag               = np.atleast_2d(np.linalg.norm(F, axis=1)).T
            total_thrust        = total_thrust + F * factor
            total_power         = total_power  + P * factor
            total_motor_current = total_motor_current + factor*motor.outputs.current

            # Pack specific outputs
            conditions.propulsion.propeller_motor_efficiency[:,ii] = etam[:,0]  
            conditions.propulsion.propeller_motor_torque[:,ii]     = motor.outputs.torque[:,0]
            conditions.propulsion.propeller_torque[:,ii]           = Q[:,0]
            conditions.propulsion.propeller_thrust[:,ii]           = np.linalg.norm(total_thrust ,axis = 1) 
            conditions.propulsion.propeller_rpm[:,ii]              = rpm[:,0]
            conditions.propulsion.propeller_tip_mach[:,ii]         = (R*rpm[:,0]*Units.rpm)/a[:,0]
            conditions.propulsion.disc_loading[:,ii]               = (F_mag[:,0])/(np.pi*(R**2)) # N/m^2                  
            conditions.propulsion.power_loading[:,ii]              = (F_mag[:,0])/(P[:,0])      # N/W      
            conditions.propulsion.propeller_efficiency[:,ii]       = etap[:,0]      
            conditions.noise.sources.propellers[prop.tag]          = outputs
            
        # Run the avionics
        avionics.power()
        
        # link
        solar_logic.inputs.pavionics =  avionics.outputs.power
        
        # Run the payload
        payload.power()
        
        # link
        solar_logic.inputs.ppayload = payload.outputs.power
        
        # link
        esc.inputs.currentout = total_motor_current
        
        # Run the esc
        esc.currentin(conditions)
        
        # link
        solar_logic.inputs.currentesc  = esc.outputs.currentin
        solar_logic.inputs.volts_motor = esc.outputs.voltageout 
        solar_logic.logic(conditions,numerics)
        
        # link
        battery.inputs = solar_logic.outputs
        battery.energy_calc(numerics)
        
        # Calculate avionics and payload power
        avionics_payload_power = avionics.outputs.power + payload.outputs.power
        
        # Pack the conditions for outputs 
        battery.inputs.current             = solar_logic.inputs.currentesc
        conditions.propulsion.solar_flux   = solar_flux.outputs.flux          
        pack_battery_conditions(conditions,battery,avionics_payload_power,P)  

        # Create the outputs
        results = Data()
        results.thrust_force_vector = total_thrust
        results.vehicle_mass_rate   = state.ones_row(1)*0.0

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
        q_motor   = segment.state.conditions.propulsion.propeller_motor_torque
        q_prop    = segment.state.conditions.propulsion.propeller_torque
        
        # Return the residuals
        segment.state.residuals.network[:,0:] = q_motor - q_prop
        
        return
    
    
    
    def add_unknowns_and_residuals_to_segment(self, segment, initial_power_coefficient = 0.005):
        """ This function sets up the information that the mission needs to run a mission segment using this network
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            segment
            initial_voltage                   [v]
            initial_power_coefficient         [float]s
            
            Outputs:
            segment.state.unknowns.propeller_power_coefficient
            segment.state.conditions.propulsion.propeller_motor_torque
            segment.state.conditions.propulsion.propeller_torque   
    
            Properties Used:
            N/A
        """           
        
        # unpack the ones function
        ones_row = segment.state.ones_row
        
        # Count how many unknowns and residuals based on p
        n_props  = len(self.propellers)
        n_motors = len(self.motors)
        n_eng    = self.number_of_engines
        
        if n_props!=n_motors!=n_eng:
            print('The number of propellers is not the same as the number of motors')
            
        # Now check if the propellers are all identical, in this case they have the same of residuals and unknowns
        if self.identical_propellers:
            n_props = 1
            
        # number of residuals, props plus the battery voltage
        n_res = n_props 

        # Assign initial segment conditions to segment if missing
        battery = self.battery
        append_initial_battery_conditions(segment,battery)          
        
        # Setup the residuals
        segment.state.residuals.network = 0. * ones_row(n_res)
        
        # Setup the unknowns
        segment.state.unknowns.propeller_power_coefficient = initial_power_coefficient * ones_row(n_props)
        
        # Setup the conditions
        segment.state.conditions.propulsion.propeller_motor_efficiency = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller_motor_torque     = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller_torque           = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller_thrust           = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller_rpm              = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.disc_loading               = 0. * ones_row(n_props)                 
        segment.state.conditions.propulsion.power_loading              = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller_tip_mach         = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller_efficiency       = 0. * ones_row(n_props)        
        
        # Ensure the mission knows how to pack and unpack the unknowns and residuals
        segment.process.iterate.unknowns.network  = self.unpack_unknowns
        segment.process.iterate.residuals.network = self.residuals        

        return segment    
            
    __call__ = evaluate_thrust
