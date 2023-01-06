## @ingroup Components-Energy-Networks
# Lift_Cruise.py
# 
# Created:  Jan 2016, E. Botero
# Modified: Mar 2020, M. Clarke
#           Apr 2021, M. Clarke
#           Jul 2021, E. Botero
#           Jul 2021, R. Erhard
#           Aug 2021, M. Clarke
#           Feb 2022, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import copy
from SUAVE.Core import Units, Data
from .Network import Network
from SUAVE.Analyses.Mission.Segments.Conditions import Residuals
from SUAVE.Components.Physical_Component import Container 
from SUAVE.Components.Energy.Converters   import  Rotor, Propeller, Lift_Rotor, Prop_Rotor
from SUAVE.Methods.Power.Battery.pack_battery_conditions import pack_battery_conditions
from SUAVE.Methods.Power.Battery.append_initial_battery_conditions import append_initial_battery_conditions

# ----------------------------------------------------------------------
#  Lift_Forward
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Lift_Cruise(Network):
    """ This is a complex version of Battery_Rotor with a battery powering propellers through
        electric motors. In this case we have 2 sets of motors at different motors that can be controlled seperately
        
        This network adds 2 extra unknowns to the mission. The first is
        a voltage, to calculate the thevenin voltage drop in the pack.
        The second is torque matching between motor and propeller.
        
        We have two inputs, the forward throttle and the lift throttle setting
        Since this is an airplane first and foremost, the "throttle" will be for forward thrust
        The new unknown will be for lift throttle, because of the assumption on throttle something needs to be done...
        Want only 1 residual on voltage
    
        Assumptions:
        For any segment using this, body angle can't be an unknown.
        
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
        self.lift_rotor_motors            = Container()
        self.propeller_motors             = Container()
        self.lift_rotors                  = Container()
        self.propellers                   = Container()
        self.avionics                     = None
        self.payload                      = None
        self.battery                      = None 
        self.lift_rotor_esc               = None
        self.propeller_esc                = None
        self.lift_rotor_engine_length     = None
        self.propeller_engine_length      = None
        self.number_of_lift_rotor_engines = 0
        self.number_of_propeller_engines  = 0
        self.identical_propellers         = True
        self.identical_lift_rotors        = True
        self.voltage                      = None   
        self.tag                          = 'Lift_Cruise'
        self.generative_design_minimum    = 0
        
        pass
        
    def evaluate_thrust(self,state):
        """ Calculate thrust given the current state of the vehicle
    
            Assumptions:
            Caps the throttle at 110% and linearly interpolates thrust off that
    
            Source:
            N/A
    
            Inputs:
            state [state()]
    
            Outputs:
            results.thrust_force_vector           [Newtons]
            results.vehicle_mass_rate             [kg/s]
            conditions.propulsion:    
                lift_rotor.rpm                    [radians/sec]
                lift_rotor.current_draw           [amps]
                lift_rotor.thrust                 [N]
                lift_rotor.torque                 [N-M]
                lift_rotor_motor.torque           [N-M]
                propeller_motor.torque            [N-M]
                propeller.thrust                  [N]
                propeller.torque                  [N-M]
                propeller.rpm                     [radians/sec]
                propeller.current_draw            [amps]
                battery.pack.power_draw           [watts]
                battery.pack.energy               [joules]
                battery.pack.voltage_open_circuit [volts]
                battery.pack.voltage_under_load   [volts]
    
            Properties Used:
            Defaulted values
        """          
        
        # unpack
        conditions        = state.conditions
        numerics          = state.numerics
        lift_rotor_motors = self.lift_rotor_motors
        propeller_motors  = self.propeller_motors
        lift_rotors       = self.lift_rotors
        propellers        = self.propellers
        lift_rotor_esc    = self.lift_rotor_esc
        propeller_esc     = self.propeller_esc        
        avionics          = self.avionics
        payload           = self.payload
        battery           = self.battery
        num_lift          = self.number_of_lift_rotor_engines
        num_forward       = self.number_of_propeller_engines  
        
        #-----------------------------------------------------------------
        # SETUP BATTERIES AND ESC's
        #-----------------------------------------------------------------
        # Set battery energy
        battery.pack.current_energy                   = conditions.propulsion.battery.pack.energy
        battery.pack.temperature                      = conditions.propulsion.battery.pack.temperature
        battery.cell.charge_throughput                = conditions.propulsion.battery.cell.charge_throughput     
        battery.cell.age                              = conditions.propulsion.battery.cell.cycle_in_day         
        battery_discharge_flag                        = conditions.propulsion.battery.discharge_flag    
        battery.cell.R_growth_factor                  = conditions.propulsion.battery.cell.resistance_growth_factor
        battery.cell.E_growth_factor                  = conditions.propulsion.battery.cell.capacity_fade_factor 
        battery.pack.max_energy                       = conditions.propulsion.battery.pack.max_aged_energy 
        n_series                                      = battery.pack.electrical_configuration.series  
        n_parallel                                    = battery.pack.electrical_configuration.parallel
        
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
        if battery_discharge_flag:    
                
            # ESC Voltage
            lift_rotor_esc.inputs.voltagein = volts      
            propeller_esc.inputs.voltagein  = volts 
            
            #---------------------------------------------------------------
            # EVALUATE THRUST FROM FORWARD PROPULSORS 
            #---------------------------------------------------------------
            # Throttle the voltage
            propeller_esc.voltageout(conditions) 
            
            # How many evaluations to do
            if self.identical_propellers:
                n_evals = 1
                factor  = num_forward*1
            else:
                n_evals = int(num_forward)
                factor  = 1.
            
            # Setup numbers for iteration
            total_prop_motor_current = 0.
            total_prop_thrust        = 0. * state.ones_row(3)
            total_prop_power         = 0.
            
            # Iterate over motor/props
            for ii in range(n_evals):    
                
                # Unpack the motor and props
                motor_key = list(propeller_motors.keys())[ii]
                prop_key  = list(propellers.keys())[ii]
                motor     = self.propeller_motors[motor_key]
                prop      = self.propellers[prop_key]            
            
                # link 
                motor.inputs.voltage  = propeller_esc.outputs.voltageout
                motor.inputs.rotor_CP = np.atleast_2d(conditions.propulsion.propeller.power_coefficient[:,ii]).T
                
                # Run the motor
                motor.omega(conditions)
                
                # link
                prop.inputs.omega           = motor.outputs.omega 
                
                # Run the propeller
                F_forward, Q_forward, P_forward, Cp_forward, outputs_forward, etap_forward = prop.spin(conditions)
                    
                # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
                eta                       = conditions.propulsion.throttle[:,0,None]
                P_forward[eta>1.0]        = P_forward[eta>1.0]*eta[eta>1.0]
                F_forward[eta[:,0]>1.0,:] = F_forward[eta[:,0]>1.0,:]*eta[eta[:,0]>1.0,:]  
                
                # Run the motor for current
                _, etam_prop = motor.current(conditions)
                
                # Conditions specific to this instantation of motor and propellers
                R                        = prop.tip_radius
                rpm                      = motor.outputs.omega / Units.rpm
                F_mag                    = np.atleast_2d(np.linalg.norm(F_forward, axis=1)).T
                total_prop_thrust        = total_prop_thrust + F_forward * factor
                total_prop_power         = total_prop_power  + P_forward * factor
                total_prop_motor_current = total_prop_motor_current + factor*motor.outputs.current
    
                # Pack specific outputs
                conditions.propulsion.propeller_motor.torque[:,ii]     = motor.outputs.torque[:,0]
                conditions.propulsion.propeller_motor.efficiency[:,ii] = etam_prop[:,0]
                conditions.propulsion.propeller.torque[:,ii]           = Q_forward[:,0]
                conditions.propulsion.propeller.rpm[:,ii]              = rpm[:,0]
                conditions.propulsion.propeller.thrust[:,ii]           = np.linalg.norm(total_prop_thrust ,axis = 1) 
                conditions.propulsion.propeller.tip_mach[:,ii]         = (R*rpm[:,0]*Units.rpm)/a[:,0]
                conditions.propulsion.propeller.disc_loading[:,ii]     = (F_mag[:,0])/(np.pi*(R**2))                 
                conditions.propulsion.propeller.power_loading[:,ii]    = (F_mag[:,0])/(P_forward[:,0])   
                conditions.propulsion.propeller.efficiency[:,ii]       = etap_forward[:,0]
                conditions.propulsion.propeller.figure_of_merit[:,ii]  = outputs_forward.figure_of_merit[:,0]  
                conditions.noise.sources.propellers[prop.tag]          = outputs_forward
                
                
            if self.identical_propellers :
                for p in self.propellers:
                    conditions.noise.sources.propellers[p.tag]      = outputs_forward
                    
                    # Append wake to each identical propeller
                    if p.Wake.wake_method=="Fidelity_One":
                    
                        # make copy of prop wake and vortex distribution
                        base_wake = copy.deepcopy(prop.Wake)
                        wake_vd   = base_wake.vortex_distribution
                        
                        # apply offset 
                        origin_offset = np.array(p.origin[0]) - np.array(prop.origin[0])
                        p.Wake = base_wake
                        p.Wake.shift_wake_VD(wake_vd, origin_offset)                    
                            
                                
                
            
            # link
            propeller_esc.inputs.currentout = total_prop_motor_current
            
            # Run the esc
            propeller_esc.currentin(conditions)        
           
            #-------------------------------------------------------------------
            # EVALUATE THRUST FROM LIFT PROPULSORS 
            #-------------------------------------------------------------------
            
            # Make a new set of konditions, since there are differences for the esc and motor
            konditions                                         = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
            konditions._size                                   = conditions._size
            konditions.propulsion                              = Data()
            konditions.freestream                              = Data()
            konditions.frames                                  = Data()
            konditions.frames.inertial                         = Data()
            konditions.frames.body                             = Data()
            konditions.propulsion.throttle                     = conditions.propulsion.throttle_lift* 1.
            konditions.propulsion.lift_rotor                   = Data()
            konditions.propulsion.lift_rotor.power_coefficient = conditions.propulsion.lift_rotor.power_coefficient * 1.
            konditions.freestream.density                      = conditions.freestream.density * 1.
            konditions.freestream.velocity                     = conditions.freestream.velocity * 1.
            konditions.freestream.dynamic_viscosity            = conditions.freestream.dynamic_viscosity * 1.
            konditions.freestream.speed_of_sound               = conditions.freestream.speed_of_sound *1.
            konditions.freestream.temperature                  = conditions.freestream.temperature * 1.
            konditions.freestream.altitude                     = conditions.freestream.altitude * 1.
            konditions.frames.inertial.velocity_vector         = conditions.frames.inertial.velocity_vector *1.
            konditions.frames.body.transform_to_inertial       = conditions.frames.body.transform_to_inertial
    
            # Throttle the voltage
            lift_rotor_esc.voltageout(konditions)      
            
            # How many evaluations to do
            if self.identical_lift_rotors:
                n_evals = 1
                factor  = num_lift*1
            else:
                n_evals = int(num_lift)
                factor  = 1.
            
            # Setup numbers for iteration
            total_lift_rotor_motor_current = 0.
            total_lift_rotor_thrust        = 0. * state.ones_row(3)
            total_lift_rotor_power         = 0.        
            
            # Iterate over motor/lift_rotors
            for ii in range(n_evals):          
                
                # Unpack the motor and props
                motor_key   = list(lift_rotor_motors.keys())[ii]
                lift_rotor_key   = list(lift_rotors.keys())[ii]
                lift_rotor_motor = self.lift_rotor_motors[motor_key]
                lift_rotor       = self.lift_rotors[lift_rotor_key]            
                        
                # link 
                lift_rotor_motor.inputs.voltage = lift_rotor_esc.outputs.voltageout
                lift_rotor_motor.inputs.rotor_CP = np.atleast_2d(conditions.propulsion.lift_rotor.power_coefficient[:,ii]).T
                
                # Run the motor
                lift_rotor_motor.omega(konditions)
                
                # link
                lift_rotor.inputs.omega           = lift_rotor_motor.outputs.omega   
                
                # Run the propeller
                F_lift, Q_lift, P_lift, Cp_lift, outputs_lift, etap_lift = lift_rotor.spin(konditions)
                
                # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
                eta                       = conditions.propulsion.throttle_lift[:,0,None]
                P_lift[eta>1.0]           = P_lift[eta>1.0]*eta[eta>1.0]
                F_lift[eta[:,0]>1.0,:]    = F_lift[eta[:,0]>1.0,:]*eta[eta[:,0]>1.0,:]  
                    
                
                # Run the motor for current
                _, etam_lift_rotor =lift_rotor_motor.current(konditions)  
                
                # Conditions specific to this instantation of motor and propellers
                R                              = lift_rotor.tip_radius
                rpm                            = lift_rotor_motor.outputs.omega / Units.rpm
                F_mag                          = np.atleast_2d(np.linalg.norm(F_lift, axis=1)).T
                total_lift_rotor_thrust        = total_lift_rotor_thrust + F_lift * factor
                total_lift_rotor_power         = total_lift_rotor_power  + P_lift * factor
                total_lift_rotor_motor_current = total_lift_rotor_motor_current + factor*lift_rotor_motor.outputs.current
                
                
                # Pack specific outputs
                conditions.propulsion.lift_rotor_motor.torque[:,ii]     = lift_rotor_motor.outputs.torque[:,0]
                conditions.propulsion.lift_rotor_motor.efficiency[:,ii] = etam_lift_rotor[:,0]
                conditions.propulsion.lift_rotor.torque[:,ii]           = Q_lift[:,0]
                conditions.propulsion.lift_rotor.rpm[:,ii]              = rpm[:,0]
                conditions.propulsion.lift_rotor.thrust[:,ii]           = np.linalg.norm(total_lift_rotor_thrust ,axis = 1) 
                conditions.propulsion.lift_rotor.tip_mach[:,ii]         = (R*rpm[:,0]*Units.rpm)/a[:,0]
                conditions.propulsion.lift_rotor.disc_loading[:,ii]     = (F_mag[:,0])/(np.pi*(R**2))                  
                conditions.propulsion.lift_rotor.power_loading[:,ii]    = (F_mag[:,0])/(P_lift[:,0])                 
                conditions.propulsion.lift_rotor.figure_of_merit[:,ii]  = outputs_lift.figure_of_merit[:,0]     
                conditions.propulsion.lift_rotor.efficiency[:,ii]       = etap_lift[:,0] 
                conditions.noise.sources.lift_rotors[lift_rotor.tag]    = outputs_lift
                
                
            if self.identical_lift_rotors:
                for r in self.lift_rotors:
                    conditions.noise.sources.propellers[r.tag]      = outputs_lift
                    
                    # Append wake to each identical propeller
                    if r.Wake.wake_method=="Fidelity_One":
                    
                        # make copy of prop wake and vortex distribution
                        base_wake = copy.deepcopy(lift_rotor.Wake)
                        wake_vd   = base_wake.vortex_distribution
                        
                        # apply offset 
                        origin_offset = np.array(p.origin[0]) - np.array(prop.origin[0])
                        r.Wake = base_wake
                        r.Wake.shift_wake_VD(wake_vd, origin_offset)  
                        
                
            # link
            lift_rotor_esc.inputs.currentout = total_lift_rotor_motor_current
            
            # Run the lift_rotor esc
            lift_rotor_esc.currentin(konditions)          
            
            # Run the avionics
            avionics.power()
        
            # Run the payload
            payload.power()
            
            # Calculate avionics and payload power
            avionics_payload_power = avionics.outputs.power + payload.outputs.power
        
            # Calculate avionics and payload current
            i_avionics_payload = avionics_payload_power/volts   
            
            # Add up the power usages
            i_lift    = lift_rotor_esc.outputs.currentin
            i_forward = propeller_esc.outputs.currentin
            
            current_total = i_lift + i_forward + i_avionics_payload
            power_total   = current_total * volts   
            
            battery.inputs.current  = current_total
            battery.inputs.power_in = - power_total
            
            # Run the battery
            battery.energy_calc(numerics,battery_discharge_flag)   
    
            
        # --------------------------------------------------------------------------------
        # Run Charge Model 
        # --------------------------------------------------------------------------------               
        else:  
            # link 
            battery.inputs.current  = -battery.cell.charging_current*n_parallel * np.ones_like(volts)
            battery.inputs.voltage  =  battery.cell.charging_voltage*n_series * np.ones_like(volts)
            battery.inputs.power_in =  -battery.inputs.current * battery.inputs.voltage             
            battery.energy_calc(numerics,battery_discharge_flag)        
             
            total_prop_thrust       = np.zeros((len(volts),3))  
            total_lift_rotor_thrust = np.zeros((len(volts),3))  
            P_forward               = np.zeros((len(volts),1))  
            P_lift                  = np.zeros((len(volts),1))  
            current_total           = battery.cell.charging_current*n_parallel * np.ones_like(volts) 
        
        # Pack the conditions for outputs     
        P = P_forward + P_lift
        avionics_payload_power = avionics.outputs.power + payload.outputs.power
        pack_battery_conditions(conditions,battery,avionics_payload_power,P)  

        F_total = total_prop_thrust + total_lift_rotor_thrust

        results = Data()
        results.thrust_force_vector       = F_total
        results.vehicle_mass_rate         = state.ones_row(1)*0.0  
        
        return results
    
    def unpack_unknowns_transition(self,segment):
        """ This is an extra set of unknowns which are unpacked from the mission solver and sent to the network.
            This uses all the motors.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            state.unknowns.lift_rotor_power_coefficient [None]
            state.unknowns.propeller_power_coefficient  [None]
            state.unknowns.battery_voltage_under_load   [volts]
            state.unknowns.throttle_lift                [0-1]
            state.unknowns.throttle                     [0-1]
    
            Outputs:
            state.conditions.propulsion.lift_rotor.power_coefficient      [None]
            state.conditions.propulsion.propeller.power_coefficient       [None]
            state.conditions.propulsion.battery.pack.voltage_under_load   [volts]
            state.conditions.propulsion.throttle_lift                     [0-1]
            state.conditions.propulsion.throttle                          [0-1]
    
            Properties Used:
            N/A
        """          
        
        # unpack the ones function
        ones_row = segment.state.ones_row
        
        # Unpack the unknowns provided for this network  
         
        ss = segment.state 
        if segment.battery_discharge: 
            ss.conditions.propulsion.lift_rotor.power_coefficient = segment.state.unknowns.lift_rotor_power_coefficient
            ss.conditions.propulsion.propeller.power_coefficient  = segment.state.unknowns.propeller_power_coefficient   
            ss.conditions.propulsion.throttle_lift                = segment.state.unknowns.throttle_lift        
            ss.conditions.propulsion.throttle                     = segment.state.unknowns.throttle  
        else: 
            ss.conditions.propulsion.propeller.power_coefficient = 0. * ones_row(1)
            
            
        battery = self.battery 
        battery.append_battery_unknowns(segment)  

        
        return
    
    
    def unpack_unknowns_cruise(self,segment):
        """ This is an extra set of unknowns which are unpacked from the mission solver and send to the network.
            This uses only the forward motors and turns the rest off.
    
            Assumptions:
            Only the forward motors and turns the rest off.
    
            Source:
            N/A
    
            Inputs:
            state.unknowns.propeller_power_coefficient [None]
            state.unknowns.battery_voltage_under_load  [volts]
            state.unknowns.throttle_lift               [0-1]
            state.unknowns.throttle                    [0-1]
    
            Outputs:
            state.conditions.propulsion.propeller.power_coefficient      [None]
            state.conditions.propulsion.battery.pack.voltage_under_load  [volts]
            state.conditions.propulsion.throttle_lift                    [0-1]
            state.conditions.propulsion.throttle                         [0-1]
    
            Properties Used:
            N/A
        """             
        
        ones_row = segment.state.ones_row  

        # Unpack the unknowns provided for this network  
        ss = segment.state 
        if segment.battery_discharge: 
            ss.conditions.propulsion.throttle_lift                       = 0.0 * ones_row(1)
            ss.conditions.propulsion.lift_rotor.power_coefficient        = 0.0 * ones_row(1) 
            ss.conditions.propulsion.propeller.power_coefficient         = segment.state.unknowns.propeller_power_coefficient
            ss.conditions.propulsion.throttle                            = segment.state.unknowns.throttle   
        else: 
            ss.conditions.propulsion.propeller.power_coefficient = 0. * ones_row(1)   
        
        battery = self.battery 
        battery.append_battery_unknowns(segment)  
        
        return    
    
    def unpack_unknowns_lift(self,segment):
        """ This is an extra set of unknowns which are unpacked from the mission solver and send to the network.
            This uses only the lift motors.
    
            Assumptions:
            Only the lift motors.
    
            Source:
            N/A
    
            Inputs:
            state.unknowns.propeller_power_coefficient [None]
            state.unknowns.battery_voltage_under_load  [volts]
            state.unknowns.throttle_lift               [0-1]
            state.unknowns.throttle                    [0-1]
    
            Outputs:
            state.conditions.propulsion.propeller.power_coefficient      [None]
            state.conditions.propulsion.battery.pack.voltage_under_load  [volts]
            state.conditions.propulsion.throttle_lift                    [0-1]
            state.conditions.propulsion.throttle                         [0-1]
    
            Properties Used:
            N/A
        """         
        
        # unpack the ones function
        ones_row = segment.state.ones_row 
        
        # Unpack the unknowns provided for this network  
        ss = segment.state 
        if segment.battery_discharge:
            ss.conditions.propulsion.throttle_lift                = segment.state.unknowns.throttle_lift 
            ss.conditions.propulsion.lift_rotor.power_coefficient = segment.state.unknowns.lift_rotor_power_coefficient
            ss.conditions.propulsion.propeller.power_coefficient  = 0.0 * ones_row(1)
            ss.conditions.propulsion.throttle                     = 0.0 * ones_row(1)
        else: 
            ss.conditions.propulsion.propeller.power_coefficient  = 0.0 * ones_row(1) 
            ss.conditions.propulsion.throttle_lift                = 0.0 * ones_row(1) 
            ss.conditions.propulsion.lift_rotor.power_coefficient = 0.0 * ones_row(1)  
            ss.conditions.propulsion.throttle                     = 0.0 * ones_row(1)
            
        battery = self.battery 
        battery.append_battery_unknowns(segment)  
            
        return    
    
    def residuals_transition(self,segment):
        """ This packs the residuals to be sent to the mission solver.
            Use this if all motors are operational
    
            Assumptions:
            All motors are operational
    
            Source:
            N/A
    
            Inputs:
            state.conditions.propulsion:
                propeller_motor_torque                [N-m]
                lift_rotor_motor.torque               [N-m]
                propeller.torque                      [N-m]
                lift_rotor.torque                     [N-m]
                voltage_under_load                    [volts]
            state.unknowns.battery_voltage_under_load [volts]
    
            Outputs:
            None
    
            Properties Used:
            self.voltage                              [volts]
        """              


        if segment.battery_discharge:  
            q_propeller_motor  = segment.state.conditions.propulsion.propeller_motor.torque
            q_prop_forward     = segment.state.conditions.propulsion.propeller.torque
            q_lift_rotor_motor = segment.state.conditions.propulsion.lift_rotor_motor.torque
            q_prop_lift        = segment.state.conditions.propulsion.lift_rotor.torque  
            segment.state.residuals.network.propellers  = (q_propeller_motor - q_prop_forward)
            segment.state.residuals.network.lift_rotors = (q_lift_rotor_motor - q_prop_lift)
               
        network       = self
        battery       = self.battery 
        battery.append_battery_residuals(segment,network)    
        return
    
    
    def residuals_cruise(self,segment):
        """ This packs the residuals to be sent to the mission solver.
            Use this if only the forward motors are operational
    
            Assumptions:
            Only the forward motors are operational
    
            Source:
            N/A
    
            Inputs:
            state.conditions.propulsion:
                propeller_motor_torque                [N-m]
                lift_rotor_motor.torque               [N-m]
                propeller.torque                      [N-m]
                lift_rotor.torque                     [N-m]
                voltage_under_load                    [volts]
            state.unknowns.battery_voltage_under_load [volts]
            
            Outputs:
            None
    
            Properties Used:
            self.voltage                              [volts]
        """          
        
        if segment.battery_discharge:   
            q_propeller_motor = segment.state.conditions.propulsion.propeller_motor.torque
            q_prop_forward    = segment.state.conditions.propulsion.propeller.torque    
            segment.state.residuals.network.propellers = (q_propeller_motor - q_prop_forward)
            
        network       = self
        battery       = self.battery 
        battery.append_battery_residuals(segment,network)    

        return    
    
    def residuals_lift(self,segment):
        """ This packs the residuals to be sent to the mission solver.
            Only the lift motors are operational
    
            Assumptions:
            The lift motors are operational
    
            Source:
            N/A
    
            Inputs:
            state.conditions.propulsion:
                propeller_motor.torque                [N-m]
                lift_rotor_motor.torque               [N-m]
                propeller.torque                      [N-m]
                lift_rotor.torque                     [N-m]
                voltage_under_load                    [volts]
            state.unknowns.battery_voltage_under_load [volts]
    
            Outputs:
            None
    
            Properties Used:
            self.voltage                              [volts]
        """            
          
        if segment.battery_discharge:   
            q_lift_rotor_motor   = segment.state.conditions.propulsion.lift_rotor_motor.torque
            q_lift_rotor_lift    = segment.state.conditions.propulsion.lift_rotor.torque        
            segment.state.residuals.network.lift_rotors  = (q_lift_rotor_motor - q_lift_rotor_lift)
            
        network       = self
        battery       = self.battery 
        battery.append_battery_residuals(segment,network)    
        return
    
    
    def add_transition_unknowns_and_residuals_to_segment(self, segment, initial_voltage = None, 
                                                         initial_prop_power_coefficient = None,
                                                         initial_lift_rotor_power_coefficient = None,
                                                         initial_throttle_lift = 0.9,
                                                         initial_battery_cell_temperature = 283. ,
                                                         initial_battery_state_of_charge = 0.5,
                                                         initial_battery_cell_current = 5.):
        """ This function sets up the information that the mission needs to run a mission segment using this network
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            segment
            initial_voltage                                              [v]
            initial_rotor_power_coefficients                                    [float]s
            
            Outputs:
            segment.state.unknowns.battery_voltage_under_load            [v]
            segment.state.unknowns.propeller_power_coefficient           [float]s
            segment.state.conditions.propulsion.propeller_motor.torque   [N-m]
            segment.state.conditions.propulsion.propeller.torque         [N-m]
    
            Properties Used:
            N/A
        """           
        
        # unpack the ones function
        ones_row = segment.state.ones_row 
        
        # Count how many unknowns and residuals based on p
        n_props       = len(self.propellers)
        n_lift_rotors = len(self.lift_rotors)
        n_motors_p    = len(self.propeller_motors)
        n_motors_r    = len(self.lift_rotor_motors)
        n_eng_p       = self.number_of_propeller_engines
        n_eng_r       = self.number_of_lift_rotor_engines
        
        if n_props!=n_motors_p!=n_eng_p:
            assert('The number of propellers is not the same as the number of motors')
            
        if n_lift_rotors!=n_motors_r!=n_eng_r:
            assert('The number of lift_rotors is not the same as the number of motors')
            
        # unpack the initial values if the user doesn't specify
        if initial_voltage==None:
            initial_voltage = self.battery.pack.max_voltage
            
        if initial_prop_power_coefficient==None:
            prop_key = list(self.propellers.keys())[0] # Use the first propeller
            initial_prop_power_coefficient = float(self.propellers[prop_key].cruise.design_power_coefficient)          
            
        if initial_lift_rotor_power_coefficient==None:
            prop_key = list(self.lift_rotors.keys())[0] # Use the first propeller
            initial_lift_rotor_power_coefficient = float(self.lift_rotors[prop_key].hover.design_power_coefficient)                
            
        # Now check if the props/lift_rotors are all identical, in this case they have the same of residuals and unknowns
        if self.identical_propellers:
            n_props = 1
        else:
            self.number_of_propeller_engines = int(self.number_of_propeller_engines)
            
        if self.identical_lift_rotors:
            n_lift_rotors = 1
        else:
            self.number_of_lift_rotor_engines = int(self.number_of_lift_rotor_engines)

        # Assign initial segment conditions to segment if missing
        battery = self.battery
        append_initial_battery_conditions(segment,battery)               

        # add unknowns and residuals specific to battery cell
        segment.state.residuals.network = Residuals()
        battery.append_battery_unknowns_and_residuals_to_segment(segment,initial_voltage, initial_battery_cell_temperature ,
                                                                           initial_battery_state_of_charge, initial_battery_cell_current)   
        if segment.battery_discharge: 
            segment.state.residuals.network.propellers          = 0. * ones_row(n_props)
            segment.state.residuals.network.lift_rotors         = 0. * ones_row(n_lift_rotors)
            segment.state.unknowns.throttle_lift                = initial_throttle_lift           * ones_row(1)   
            segment.state.unknowns.propeller_power_coefficient  = initial_prop_power_coefficient  * ones_row(n_props)
            segment.state.unknowns.lift_rotor_power_coefficient = initial_lift_rotor_power_coefficient * ones_row(n_lift_rotors)   
        
        # Setup the conditions for the propellers
        segment.state.conditions.propulsion.propeller_motor.torque     = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller_motor.efficiency = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller.torque           = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller.rpm              = 0. * ones_row(n_props)      
        segment.state.conditions.propulsion.propeller.disc_loading     = 0. * ones_row(n_props)                 
        segment.state.conditions.propulsion.propeller.power_loading    = 0. * ones_row(n_props)      
        segment.state.conditions.propulsion.propeller.thrust           = 0. * ones_row(n_props)         
        segment.state.conditions.propulsion.propeller.tip_mach         = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller.efficiency       = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller.figure_of_merit  = 0. * ones_row(n_props)

        # Setup the conditions for the lift_rotors
        segment.state.conditions.propulsion.lift_rotor_motor.torque      = 0. * ones_row(n_lift_rotors)
        segment.state.conditions.propulsion.lift_rotor_motor.efficiency  = 0. * ones_row(n_lift_rotors)
        segment.state.conditions.propulsion.lift_rotor.torque            = 0. * ones_row(n_lift_rotors)
        segment.state.conditions.propulsion.lift_rotor.rpm               = 0. * ones_row(n_lift_rotors)
        segment.state.conditions.propulsion.lift_rotor.disc_loading      = 0. * ones_row(n_lift_rotors)                 
        segment.state.conditions.propulsion.lift_rotor.power_loading     = 0. * ones_row(n_lift_rotors)                
        segment.state.conditions.propulsion.lift_rotor.figure_of_merit   = 0. * ones_row(n_lift_rotors)              
        segment.state.conditions.propulsion.lift_rotor.thrust            = 0. * ones_row(n_lift_rotors)         
        segment.state.conditions.propulsion.lift_rotor.tip_mach          = 0. * ones_row(n_lift_rotors)
        segment.state.conditions.propulsion.lift_rotor.efficiency        = 0. * ones_row(n_lift_rotors)
        
        # Ensure the mission knows how to pack and unpack the unknowns and residuals
        segment.process.iterate.unknowns.network  = self.unpack_unknowns_transition
        segment.process.iterate.residuals.network = self.residuals_transition      

        return segment
    
    
    def add_cruise_unknowns_and_residuals_to_segment(self, segment, initial_voltage = None, 
                                                         initial_prop_power_coefficient = None,
                                                         initial_battery_cell_temperature = 283.,
                                                         initial_battery_state_of_charge = 0.5,
                                                         initial_battery_cell_current = 5.):
        """ This function sets up the information that the mission needs to run a mission segment using this network
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            segment
            initial_voltage                                              [v]
            initial_rotor_power_coefficients                                    [float]s
            
            Outputs:
            segment.state.unknowns.battery_voltage_under_load            [v]
            segment.state.unknowns.propeller_power_coefficient           [float]s
            segment.state.conditions.propulsion.propeller_motor.torque   [N-m]
            segment.state.conditions.propulsion.propeller.torque         [N-m]
    
            Properties Used:
            N/A
        """           
        
        # unpack the ones function
        ones_row = segment.state.ones_row 
        
        # Count how many unknowns and residuals based on p
        n_props       = len(self.propellers)
        n_lift_rotors = len(self.lift_rotors)
        n_motors_p    = len(self.propeller_motors)
        n_eng_p       = self.number_of_propeller_engines
        
        if n_props!=n_motors_p!=n_eng_p:
            assert('The number of propellers is not the same as the number of motors')
            
        # Now check if the props/lift_rotors are all identical, in this case they have the same of residuals and unknowns
        if self.identical_propellers:
            n_props = 1
        else:
            self.number_of_propeller_engines = int(self.number_of_propeller_engines)
            
        if self.identical_lift_rotors:
            n_lift_rotors = 1
        else:
            self.number_of_lift_rotor_engines = int(self.number_of_lift_rotor_engines)  
            
        # unpack the initial values if the user doesn't specify
        if initial_voltage==None:
            initial_voltage = self.battery.pack.max_voltage
            
        if initial_prop_power_coefficient==None:
            prop_key = list(self.propellers.keys())[0] # Use the first propeller
            if type(self.propellers[prop_key]) == Propeller or type(self.propellers[prop_key]) == Rotor: 
                initial_prop_power_coefficient = float(self.propellers[prop_key].cruise.design_power_coefficient)         
            elif type(self.propellers[prop_key]) == Lift_Rotor or type(self.propellers[prop_key]) == Prop_Rotor: 
                initial_prop_power_coefficient = float(self.propellers[prop_key].hover.design_power_coefficient)  
            else:
                initial_prop_power_coefficient = float(self.propellers[prop_key].design_power_coefficient)
                                                       
        # Assign initial segment conditions to segment if missing  
        battery = self.battery
        append_initial_battery_conditions(segment,battery)          
      
        # add unknowns and residuals specific to to battery cell
        segment.state.residuals.network = Residuals()
        battery.append_battery_unknowns_and_residuals_to_segment(segment,initial_voltage, initial_battery_cell_temperature ,
                                                                           initial_battery_state_of_charge, initial_battery_cell_current)   
        if segment.battery_discharge: 
            segment.state.residuals.network.propellers         = 0. * ones_row(n_props)
            segment.state.unknowns.propeller_power_coefficient = initial_prop_power_coefficient * ones_row(n_props)    
        
        # Setup the conditions for the propellers
        segment.state.conditions.propulsion.propeller_motor.torque     = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller_motor.efficiency = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller.torque           = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller.rpm              = 0. * ones_row(n_props)      
        segment.state.conditions.propulsion.propeller.disc_loading     = 0. * ones_row(n_props)      
        segment.state.conditions.propulsion.propeller.thrust           = 0. * ones_row(n_props)                
        segment.state.conditions.propulsion.propeller.power_loading    = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller.tip_mach         = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller.efficiency       = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller.figure_of_merit  = 0. * ones_row(n_props)
        
        # Setup the conditions for the lift_rotors
        segment.state.conditions.propulsion.lift_rotor_motor.torque      = 0. * ones_row(n_lift_rotors)
        segment.state.conditions.propulsion.lift_rotor_motor.efficiency  = 0. * ones_row(n_lift_rotors)
        segment.state.conditions.propulsion.lift_rotor.torque            = 0. * ones_row(n_lift_rotors)
        segment.state.conditions.propulsion.lift_rotor.rpm               = 0. * ones_row(n_lift_rotors)
        segment.state.conditions.propulsion.lift_rotor.disc_loading      = 0. * ones_row(n_lift_rotors)               
        segment.state.conditions.propulsion.lift_rotor.thrust            = 0. * ones_row(n_lift_rotors)                
        segment.state.conditions.propulsion.lift_rotor.power_loading     = 0. * ones_row(n_lift_rotors)                
        segment.state.conditions.propulsion.lift_rotor.figure_of_merit   = 0. * ones_row(n_lift_rotors)        
        segment.state.conditions.propulsion.lift_rotor.tip_mach          = 0. * ones_row(n_lift_rotors)       
        segment.state.conditions.propulsion.lift_rotor.efficiency        = 0. * ones_row(n_lift_rotors)

        # Ensure the mission knows how to pack and unpack the unknowns and residuals
        segment.process.iterate.unknowns.network  = self.unpack_unknowns_cruise
        segment.process.iterate.residuals.network = self.residuals_cruise     

        return segment
    
    
    def add_lift_unknowns_and_residuals_to_segment(self, segment, initial_voltage = None,
                                                   initial_lift_rotor_power_coefficient = None,
                                                         initial_throttle_lift = 0.9,
                                                         initial_battery_cell_temperature = 283.,
                                                         initial_battery_state_of_charge = 0.5,
                                                         initial_battery_cell_current = 5.):
        """ This function sets up the information that the mission needs to run a mission segment using this network

            Assumptions:
            None

            Source:
            N/A

            Inputs:
            segment
            initial_voltage                                            [v]
            initial_rotor_power_coefficients                                  [float]s

            Outputs:
            segment.state.unknowns.battery_voltage_under_load          [v]
            segment.state.unknowns.propeller.power_coefficient         [float]s
            segment.state.conditions.propulsion.propeller_motor.torque [N-m]
            segment.state.conditions.propulsion.propeller.torque       [N-m]

            Properties Used:
            N/A
        """           

        # unpack the ones function
        ones_row = segment.state.ones_row

        # unpack the initial values if the user doesn't specify
        if initial_voltage==None:
            initial_voltage = self.battery.pack.max_voltage    
            
        if initial_lift_rotor_power_coefficient==None:
            prop_key = list(self.lift_rotors.keys())[0] # Use the first propeller 
            if type(self.lift_rotors[prop_key]) == Propeller or type(self.lift_rotors[prop_key]) == Rotor: 
                initial_lift_rotor_power_coefficient = float(self.lift_rotors[prop_key].cruise.design_power_coefficient)         
            elif type(self.lift_rotors[prop_key]) == Lift_Rotor or type(self.lift_rotors[prop_key]) == Prop_Rotor: 
                initial_lift_rotor_power_coefficient = float(self.lift_rotors[prop_key].hover.design_power_coefficient)  
            else:
                initial_lift_rotor_power_coefficient = float(self.lift_rotors[prop_key].design_power_coefficient)
                
                        
        # Count how many unknowns and residuals based on p
        n_lift_rotors   = len(self.lift_rotors)
        n_motors_r = len(self.lift_rotor_motors)
        n_eng_r    = self.number_of_lift_rotor_engines

        if n_lift_rotors!=n_motors_r!=n_eng_r:
            assert('The number of lift_rotors is not the same as the number of motors')

        # Now check if the props/lift_rotors are all identical, in this case they have the same of residuals and unknowns
        if self.identical_propellers:
            n_props = 1
        else:
            self.number_of_propeller_engines = int(self.number_of_propeller_engines)

        if self.identical_lift_rotors:
            n_lift_rotors = 1
        else:
            self.number_of_lift_rotor_engines = int(self.number_of_lift_rotor_engines)
 
        # Assign initial segment conditions to segment if missing  
        battery = self.battery
        append_initial_battery_conditions(segment,battery)          

        # add unknowns and residuals specific to battery cell
        segment.state.residuals.network = Residuals()
        battery.append_battery_unknowns_and_residuals_to_segment(segment,initial_voltage, initial_battery_cell_temperature ,
                                                                           initial_battery_state_of_charge, initial_battery_cell_current)   
        if segment.battery_discharge: 
            segment.state.residuals.network.lift_rotors = 0. * ones_row(n_lift_rotors) 
            segment.state.unknowns.__delitem__('throttle')
            segment.state.unknowns.throttle_lift                = initial_throttle_lift  * ones_row(1) 
            segment.state.unknowns.lift_rotor_power_coefficient = initial_lift_rotor_power_coefficient * ones_row(n_lift_rotors)  

        # Setup the conditions for the propellers
        segment.state.conditions.propulsion.propeller_motor.torque     = 0. * ones_row(n_props)    
        segment.state.conditions.propulsion.propeller_motor.efficiency = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller.torque           = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller.rpm              = 0. * ones_row(n_props)      
        segment.state.conditions.propulsion.propeller.disc_loading     = 0. * ones_row(n_props)     
        segment.state.conditions.propulsion.propeller.thrust           = 0. * ones_row(n_props)                 
        segment.state.conditions.propulsion.propeller.power_loading    = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller.tip_mach         = 0. * ones_row(n_props)   
        segment.state.conditions.propulsion.propeller.efficiency       = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller.figure_of_merit  = 0. * ones_row(n_props)     
        segment.state.conditions.propulsion.propeller.y_axis_rotation  = 0. * ones_row(n_props)

        # Setup the conditions for the lift_rotors
        segment.state.conditions.propulsion.lift_rotor_motor.torque      = 0. * ones_row(n_lift_rotors)
        segment.state.conditions.propulsion.lift_rotor_motor.efficiency  = 0. * ones_row(n_lift_rotors)
        segment.state.conditions.propulsion.lift_rotor.torque            = 0. * ones_row(n_lift_rotors)
        segment.state.conditions.propulsion.lift_rotor.rpm               = 0. * ones_row(n_lift_rotors)             
        segment.state.conditions.propulsion.lift_rotor.thrust            = 0. * ones_row(n_lift_rotors) 
        segment.state.conditions.propulsion.lift_rotor.disc_loading      = 0. * ones_row(n_lift_rotors)                 
        segment.state.conditions.propulsion.lift_rotor.power_loading     = 0. * ones_row(n_lift_rotors)             
        segment.state.conditions.propulsion.lift_rotor.figure_of_merit   = 0. * ones_row(n_lift_rotors)           
        segment.state.conditions.propulsion.lift_rotor.tip_mach          = 0. * ones_row(n_lift_rotors)
        segment.state.conditions.propulsion.lift_rotor.efficiency        = 0. * ones_row(n_lift_rotors)
        segment.state.conditions.propulsion.lift_rotor.y_axis_rotation   = 0. * ones_row(n_lift_rotors)

        # Ensure the mission knows how to pack and unpack the unknowns and residuals
        segment.process.iterate.unknowns.network  = self.unpack_unknowns_lift
        segment.process.iterate.residuals.network = self.residuals_lift 

        return segment      