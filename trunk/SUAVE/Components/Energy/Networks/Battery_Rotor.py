## @ingroup Components-Energy-Networks
# Battery_Rotor.py
# 
# Created:  Jul 2015, E. Botero
# Modified: Feb 2016, T. MacDonald
#           Mar 2020, M. Clarke 
#           Apr 2021, M. Clarke
#           Jul 2021, E. Botero
#           Jul 2021, R. Erhard
#           Aug 2021, M. Clarke
#           Feb 2022, R. Erhard
#           Mar 2022, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import SUAVE
import numpy as np
from .Network import Network
from SUAVE.Analyses.Mission.Segments.Conditions import Residuals
from SUAVE.Components.Energy.Converters   import  Rotor, Propeller, Lift_Rotor, Prop_Rotor 
from SUAVE.Components.Physical_Component import Container 
from SUAVE.Methods.Power.Battery.pack_battery_conditions import pack_battery_conditions
from SUAVE.Methods.Power.Battery.append_initial_battery_conditions import append_initial_battery_conditions
from SUAVE.Core import Data , Units 
import copy

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Battery_Rotor(Network):
    """ This is a simple network with a battery powering a rotor through
        an electric motor
        
        This network adds 2 extra unknowns to the mission. The first is
        a voltage, to calculate the thevenin voltage drop in the pack.
        The second is torque matching between motor and rotor.
    
        Assumptions:
        The y axis rotation is used for rotating the rotor about the Y-axis for tilt rotors and tiltwings
        
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
        
        self.rotor_motors                 = Container()
        self.rotors                       = Container()
        self.esc                          = None
        self.avionics                     = None
        self.payload                      = None
        self.battery                      = None
        self.nacelle_diameter             = None
        self.engine_length                = None
        self.number_of_rotor_engines      = None
        self.voltage                      = None
        self.tag                          = 'Battery_Rotor'
        self.use_surrogate                = False 
        self.generative_design_minimum    = 0 
        self.identical_rotors             = True
        self.y_axis_rotation              = 0.
    
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
                rpm                          [radians/sec]
                current                      [amps]
                battery_power_draw           [watts]
                battery_energy               [joules]
                battery_voltage_open_circuit [V]
                battery_voltage_under_load   [V]
                motor_torque                 [N-M]
                rotor.torque             [N-M]
    
            Properties Used:
            Defaulted values
        """          
    
        # unpack  
        conditions     = state.conditions
        numerics       = state.numerics
        esc            = self.esc
        avionics       = self.avionics
        payload        = self.payload
        battery        = self.battery 
        num_engines    = self.number_of_rotor_engines
        identical_flag = self.identical_rotors
        motors         = self.rotor_motors
        props          = self.rotors

        
        # Set battery energy
        battery.current_energy           = conditions.propulsion.battery.energy
        battery.pack.temperature         = conditions.propulsion.battery.pack.temperature
        battery.cell.charge_throughput   = conditions.propulsion.battery.cell.charge_throughput     
        battery.age                      = conditions.propulsion.battery.cycle_day        
        discharge_flag                   = conditions.propulsion.battery.discharge_flag    
        battery.R_growth_factor          = conditions.propulsion.battery.resistance_growth_factor
        battery.E_growth_factor          = conditions.propulsion.battery.capacity_fade_factor 
        battery.max_energy               = conditions.propulsion.battery.max_aged_energy 
        n_series                         = battery.pack.electrical_configuration.series  
        n_parallel                       = battery.pack.electrical_configuration.parallel
        
        # update ambient temperature based on altitude
        battery.ambient_temperature                   = conditions.freestream.temperature   
        battery.cooling_fluid.thermal_conductivity    = conditions.freestream.thermal_conductivity
        battery.cooling_fluid.kinematic_viscosity     = conditions.freestream.kinematic_viscosity
        battery.cooling_fluid.prandtl_number          = conditions.freestream.prandtl_number
        battery.cooling_fluid.density                 = conditions.freestream.density  
        battery.ambient_pressure                      = conditions.freestream.pressure  
        a                                             = conditions.freestream.speed_of_sound 
        
        # Predict voltage based on battery  
        volts = battery.compute_voltage(state)  
            
        # --------------------------------------------------------------------------------
        # Run Motor, Avionics and Systems (Discharge Model)
        # --------------------------------------------------------------------------------    
        if discharge_flag:    
            # Step 1 battery power
            esc.inputs.voltagein = volts
            
            # Step 2 throttle the voltage
            esc.voltageout(conditions)
            
            # How many evaluations to do
            if identical_flag:
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
                prop_key  = list(props.keys())[ii]
                motor     = self.rotor_motors[motor_key]
                prop      = self.rotors[prop_key]

                # Set rotor y-axis rotation                
                prop.inputs.y_axis_rotation = conditions.propulsion.rotor.y_axis_rotation    
            
                if identical_flag:
                    for idx in range(1,int(num_engines)) :
                        rotor_remainder      = self.rotors[list(props.keys())[idx]]
                        rotor_remainder.inputs.y_axis_rotation = conditions.propulsion.rotor.y_axis_rotation
                        
                # link 
                motor.inputs.voltage        = esc.outputs.voltageout
                motor.inputs.rotor_CP   = np.atleast_2d(conditions.propulsion.rotor.power_coefficient[:,ii]).T
                
                # step 3
                motor.omega(conditions)
                
                # link
                prop.inputs.omega           = motor.outputs.omega 
                
                # step 4
                F, Q, P, Cp, outputs, etap = prop.spin(conditions)
                    
                # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
                eta        = conditions.propulsion.throttle[:,0,None]
                P[eta>1.0] = P[eta>1.0]*eta[eta>1.0]
                F[eta[:,0]>1.0,:] = F[eta[:,0]>1.0,:]*eta[eta[:,0]>1.0,:]
    
                # Run the motor for current
                _ , etam =  motor.current(conditions)
                
                # Conditions specific to this instantation of motor and rotors
                R                   = prop.tip_radius
                rpm                 = motor.outputs.omega / Units.rpm
                F_mag               = np.atleast_2d(np.linalg.norm(F, axis=1)).T
                total_thrust        = total_thrust + F * factor
                total_power         = total_power  + P * factor
                total_motor_current = total_motor_current + factor*motor.outputs.current
    
                # Pack specific outputs
                conditions.propulsion.rotor_motor.efficiency[:,ii]  = etam[:,0]  
                conditions.propulsion.rotor_motor.torque[:,ii]      = motor.outputs.torque[:,0]
                conditions.propulsion.rotor.torque[:,ii]            = Q[:,0]
                conditions.propulsion.rotor.thrust[:,ii]            = np.linalg.norm(total_thrust ,axis = 1) 
                conditions.propulsion.rotor.rpm[:,ii]               = rpm[:,0]
                conditions.propulsion.rotor.tip_mach[:,ii]          = (R*rpm[:,0]*Units.rpm)/a[:,0]
                conditions.propulsion.rotor.disc_loading[:,ii]      = (F_mag[:,0])/(np.pi*(R**2)) # N/m^2                  
                conditions.propulsion.rotor.power_loading[:,ii]     = (F_mag[:,0])/(P[:,0])       # N/W      
                conditions.propulsion.rotor.efficiency[:,ii]        = etap[:,0]  
                conditions.propulsion.rotor.figure_of_merit[:,ii]   = outputs.figure_of_merit[:,0] 
                
                conditions.noise.sources.rotors[prop.tag]      = outputs
            
            if identical_flag and prop.Wake.wake_method=="Fidelity_One":
                # append wakes to all rotors, shifted by new origin
                for p in props:
                    # make copy of prop wake and vortex distribution
                    base_wake = copy.deepcopy(prop.Wake)
                    wake_vd   = base_wake.vortex_distribution
                    
                    # apply offset 
                    origin_offset = np.array(p.origin[0]) - np.array(prop.origin[0])
                    p.Wake = base_wake
                    p.Wake.shift_wake_VD(wake_vd, origin_offset)
            elif identical_flag and prop.Wake.wake_method=="Fidelity_Zero":
                for p in props:
                    p.outputs = outputs
                    
            # Run the avionics
            avionics.power()
    
            # Run the payload
            payload.power()
            
            # link
            esc.inputs.currentout = total_motor_current
    
            # Run the esc
            esc.currentin(conditions)  
            
            # Calculate avionics and payload power
            avionics_payload_power = avionics.outputs.power + payload.outputs.power
        
            # Calculate avionics and payload current
            avionics_payload_current = avionics_payload_power/self.voltage 
        
            # link
            battery.inputs.current  =   esc.outputs.currentin + avionics_payload_current
            battery.inputs.power_in = -(esc.outputs.power_in  + avionics_payload_power)
            battery.energy_calc(numerics,discharge_flag)         
             
        # --------------------------------------------------------------------------------
        # Run Charge Model 
        # --------------------------------------------------------------------------------               
        else:  
            # link 
            battery.inputs.current  = -battery.cell.charging_current*n_parallel * np.ones_like(volts)
            battery.inputs.voltage  =  battery.cell.charging_voltage*n_series * np.ones_like(volts)
            battery.inputs.power_in = -battery.inputs.current * battery.inputs.voltage             
            battery.energy_calc(numerics,discharge_flag)        
            
            avionics_payload_power   = np.zeros((len(volts),1)) 
            total_thrust             = np.zeros((len(volts),3)) 
            P                        = battery.inputs.power_in
            
        # Pack the conditions for outputs
        pack_battery_conditions(conditions,battery,avionics_payload_power,P)  
        
         # Create the outputs
        results = Data()
        results.thrust_force_vector       = total_thrust
        results.vehicle_mass_rate         = state.ones_row(1)*0.0     
        results.network_y_axis_rotation   = conditions.propulsion.rotor.y_axis_rotation
     
        return results
     
    def unpack_unknowns(self,segment):
        """ This is an extra set of unknowns which are unpacked from the mission solver and send to the network.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            state.unknowns.rotor_power_coefficient [None] 
            unknowns specific to the battery cell 
    
            Outputs:
            state.conditions.propulsion.rotor_power_coefficient [None] 
            conditions specific to the battery cell
    
            Properties Used:
            N/A
        """                          
        
        # unpack the ones function
        ones_row = segment.state.ones_row
        
        # Here we are going to unpack the unknowns (Cp) provided for this network
        ss = segment.state 
        if segment.battery_discharge:
            ss.conditions.propulsion.rotor.power_coefficient = ss.unknowns.rotor_power_coefficient   
        else: 
            ss.conditions.propulsion.rotor.power_coefficient = 0. * ones_row(1)
        
        # fixed y axis rotation
        net                                            = list(segment.analyses.energy.network.keys())[0]
        y_rot                                          = segment.analyses.energy.network[net].y_axis_rotation
        ss.conditions.propulsion.rotor.y_axis_rotation = y_rot * ones_row(1)
        self.y_axis_rotation                           = y_rot
        
        battery = self.battery 
        battery.append_battery_unknowns(segment)          
        
        return  

    def unpack_tiltrotor_transition_unknowns(self,segment):
        """ This is an extra set of unknowns which are unpacked from the mission solver and send to the network.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            state.unknowns.rotor_y_axis_rotation          [rad] 
            state.unknowns.rotor_power_coefficient    [None] 
            unknowns specific to the battery cell 
    
            Outputs:
            state.conditions.propulsion.rotor.power_coefficient [None] 
            conditions specific to the battery cell
    
            Properties Used:
            N/A
        """                          
        
        # unpack the ones function
        ones_row = segment.state.ones_row   
        
        # Here we are going to unpack the unknowns (Cp) provided for this network
        ss = segment.state 
        if segment.battery_discharge:
            ss.conditions.propulsion.rotor.power_coefficient     = ss.unknowns.rotor_power_coefficient       
            ss.conditions.propulsion.throttle                    = ss.unknowns.throttle
            ss.conditions.propulsion.rotor.y_axis_rotation       = ss.unknowns.rotor_y_axis_rotation 
        else: 
            ss.conditions.propulsion.rotor_power_coefficient = 0. * ones_row(1)
        
        # update y axis rotation
        self.y_axis_rotation = ss.conditions.propulsion.rotor.y_axis_rotation 
        
        battery = self.battery 
        battery.append_battery_unknowns(segment)          
        
        
        return          
    
    
    def residuals(self,segment):
        """ This packs the residuals to be sent to the mission solver.
   
           Assumptions:
           None
   
           Source:
           N/A
   
           Inputs:
           state.conditions.propulsion:
               motor_torque                      [N-m]
               rotor.torque                      [N-m] 
           unknowns specific to the battery cell 
           
           Outputs:
           residuals specific to battery cell and network
   
           Properties Used: 
           N/A
       """           
           
        if segment.battery_discharge:    
            q_motor   = segment.state.conditions.propulsion.rotor_motor.torque
            q_prop    = segment.state.conditions.propulsion.rotor.torque                
            segment.state.residuals.network.rotors = q_motor - q_prop
            
        network       = self
        battery       = self.battery 
        battery.append_battery_residuals(segment,network)           
         
        return     
    
    ## @ingroup Components-Energy-Networks
    def add_unknowns_and_residuals_to_segment(self, segment, initial_voltage = None, initial_power_coefficient = None,
                                              initial_battery_cell_temperature = 283. , initial_battery_state_of_charge = 0.5,
                                              initial_battery_cell_current = 5.):
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
            segment.state.unknowns.battery_voltage_under_load
            segment.state.unknowns.rotor_power_coefficient
            segment.state.conditions.propulsion.rotor_motor.torque
            segment.state.conditions.propulsion.rotor.torque   
    
            Properties Used:
            N/A
        """           

        n_eng          = int(self.number_of_rotor_engines)
        identical_flag = self.identical_rotors
        n_props        = len(self.rotors)
        n_motors       = len(self.rotor_motors)
        
        # unpack the ones function
        ones_row = segment.state.ones_row
        
        # unpack the initial values if the user doesn't specify
        if initial_voltage==None:
            initial_voltage = self.battery.max_voltage
            
        if initial_power_coefficient==None:
            prop_key = list(self.rotors.keys())[0] # Use the first rotor
            if type(self.rotors[prop_key]) == Propeller or type(self.rotors[prop_key]) == Rotor:
                initial_power_coefficient = float(self.rotors[prop_key].cruise.design_power_coefficient)
            if type(self.rotors[prop_key]) == Lift_Rotor or type(self.rotors[prop_key]) == Prop_Rotor:
                initial_power_coefficient = float(self.rotors[prop_key].hover.design_power_coefficient)

        # Count how many unknowns and residuals based on p
        if n_props!=n_motors!=n_eng:
            print('The number of rotors is not the same as the number of motors')
            
        # Now check if the rotors are all identical, in this case they have the same of residuals and unknowns
        if identical_flag:
            n_props = 1   

        # Assign initial segment conditions to segment if missing
        battery = self.battery
        append_initial_battery_conditions(segment,battery)          
        
        # add unknowns and residuals specific to battery cell 
        segment.state.residuals.network = Residuals()  
        battery.append_battery_unknowns_and_residuals_to_segment(segment,initial_voltage,
                                              initial_battery_cell_temperature , initial_battery_state_of_charge,
                                              initial_battery_cell_current)  

        if segment.battery_discharge:
            segment.state.unknowns.rotor_power_coefficient = initial_power_coefficient * ones_row(n_props)  
        
        # Setup the conditions
        segment.state.conditions.propulsion.rotor_motor.efficiency = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.rotor_motor.torque     = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.rotor.torque           = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.rotor.thrust           = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.rotor.rpm              = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.rotor.disc_loading     = 0. * ones_row(n_props)                 
        segment.state.conditions.propulsion.rotor.power_loading    = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.rotor.tip_mach         = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.rotor.efficiency       = 0. * ones_row(n_props)   
        segment.state.conditions.propulsion.rotor.figure_of_merit  = 0. * ones_row(n_props)      
        
        # Ensure the mission knows how to pack and unpack the unknowns and residuals
        segment.process.iterate.unknowns.network  = self.unpack_unknowns
        segment.process.iterate.residuals.network = self.residuals        

        return segment
    
    ## @ingroup Components-Energy-Networks
    def add_tiltrotor_transition_unknowns_and_residuals_to_segment(self, segment, initial_voltage = None, 
                                                        initial_y_axis_rotation = 0.0,
                                                        initial_power_coefficient = None,
                                                        initial_battery_cell_temperature = 283. , 
                                                        initial_battery_state_of_charge = 0.5,
                                                        initial_battery_cell_current = 5.):
        """ This function sets up the information that the mission needs to run a mission segment using this network
    
            Assumptions:
            Network of tiltrotors used to converge on transition residuals, all rotors having same tilt angle
    
            Source:
            N/A
    
            Inputs:
            segment
            initial_y_axis_rotation           [float]
            initial_power_coefficient         [float]
            initial_battery_cell_temperature  [float]
            initial_battery_state_of_charge   [float]
            initial_battery_cell_current      [float]
            
            Outputs:
            segment.state.unknowns.battery_voltage_under_load
            segment.state.unknowns.rotor_power_coefficient
            segment.state.unknowns.throttle
            segment.state.unknowns.rotor_y_axis_rotation
            segment.state.conditions.propulsion.rotor_motor.torque
            segment.state.conditions.propulsion.rotor.torque   
    
            Properties Used:
            N/A
        """           

        n_eng          = int(self.number_of_rotor_engines)
        identical_flag = self.identical_rotors
        n_props        = len(self.rotors)
        n_motors       = len(self.rotor_motors)
        
        # unpack the ones function
        ones_row = segment.state.ones_row
        
        # unpack the initial values if the user doesn't specify
        if initial_voltage==None:
            initial_voltage = self.battery.max_voltage
            
        if initial_power_coefficient==None:
            prop_key = list(self.rotors.keys())[0] # Use the first rotor
            initial_power_coefficient = float(self.rotors[prop_key].design_power_coefficient)
        
        # Count how many unknowns and residuals based on p)         
        if n_props!=n_motors!=n_eng:
            print('The number of rotors is not the same as the number of motors')
            
        # Now check if the rotors are all identical, in this case they have the same of residuals and unknowns
        if identical_flag:
            n_props = 1   

        # Assign initial segment conditions to segment if missing
        battery = self.battery
        append_initial_battery_conditions(segment,battery)          
        
        # add unknowns and residuals specific to battery cell 
        segment.state.residuals.network = Residuals()  
        battery.append_battery_unknowns_and_residuals_to_segment(segment,initial_voltage,
                                              initial_battery_cell_temperature , initial_battery_state_of_charge,
                                              initial_battery_cell_current)  

        if segment.battery_discharge: 
            segment.state.unknowns.rotor_power_coefficient = initial_power_coefficient * ones_row(n_props)  
            segment.state.unknowns.rotor_y_axis_rotation   = initial_y_axis_rotation * ones_row(1)  
            segment.state.unknowns.throttle                = 0.7 * ones_row(1)
        
        # Setup the conditions
        segment.state.conditions.propulsion.rotor_motor.efficiency = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.rotor_motor.torque     = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.rotor.torque           = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.rotor.thrust           = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.rotor.rpm              = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.rotor.disc_loading     = 0. * ones_row(n_props)                 
        segment.state.conditions.propulsion.rotor.power_loading    = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.rotor.tip_mach         = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.rotor.efficiency       = 0. * ones_row(n_props) 
        segment.state.conditions.propulsion.rotor.figure_of_merit  = 0. * ones_row(n_props)       
        
        # Ensure the mission knows how to pack and unpack the unknowns and residuals
        segment.process.iterate.unknowns.network  = self.unpack_tiltrotor_transition_unknowns
        segment.process.iterate.residuals.network = self.residuals   
        segment.process.iterate.unknowns.mission  = SUAVE.Methods.skip

        return segment    
       
    
    __call__ = evaluate_thrust


