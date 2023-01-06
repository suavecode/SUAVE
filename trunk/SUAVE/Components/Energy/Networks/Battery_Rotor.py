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
        
        self.motors                       = Container()
        self.rotors                       = Container()
        self.esc                          = None
        self.avionics                     = None
        self.payload                      = None
        self.battery                      = None  
        self.voltage                      = None
        self.tag                          = 'Battery_Rotor'
        self.use_surrogate                = False 
        self.generative_design_minimum    = 0 
        self.rotor_group_indexes          = [0]
        self.motor_group_indexes          = [0]
    
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
        conditions             = state.conditions
        numerics               = state.numerics
        esc                    = self.esc
        avionics               = self.avionics
        payload                = self.payload
        battery                = self.battery  
        rotor_group_indexes    = self.rotor_group_indexes
        motor_group_indexes    = self.motor_group_indexes
        motors                 = self.motors
        props                  = self.rotors

        
        # Set battery energy
        battery.pack.current_energy      = conditions.propulsion.battery.pack.energy
        battery.pack.temperature         = conditions.propulsion.battery.pack.temperature
        battery.pack.max_energy          = conditions.propulsion.battery.pack.max_aged_energy        
        battery.cell.charge_throughput   = conditions.propulsion.battery.cell.charge_throughput     
        battery.cell.age                 = conditions.propulsion.battery.cell.cycle_in_day  
        battery.cell.R_growth_factor     = conditions.propulsion.battery.cell.resistance_growth_factor
        battery.cell.E_growth_factor     = conditions.propulsion.battery.cell.capacity_fade_factor 
        discharge_flag                   = conditions.propulsion.battery.discharge_flag   
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
            unique_rotor_groups,factors = np.unique(rotor_group_indexes, return_counts=True)
            unique_motor_groups,factors = np.unique(motor_group_indexes, return_counts=True)
            if (unique_rotor_groups == unique_motor_groups).all(): # rotors and motors are paired 
                n_evals = len(unique_rotor_groups)
                rotor_indexes = unique_rotor_groups
                motor_indexes = unique_motor_groups
                factor        = factors
            else:
                n_evals = len(rotor_group_indexes)
                rotor_indexes = rotor_group_indexes
                motor_indexes = motor_group_indexes
                factor        = np.ones_like(motor_group_indexes)
            
            # Setup numbers for iteration
            total_motor_current = 0.
            total_thrust        = 0. * state.ones_row(3)
            total_power         = 0.
            
            # Iterate over motor/props
            for ii in range(n_evals):
                
                # Unpack the motor and props
                motor_key = list(motors.keys())[motor_indexes[ii]]
                prop_key  = list(props.keys())[rotor_indexes[ii]]
                motor     = self.motors[motor_key]
                rotor     = self.rotors[prop_key]

                # Set rotor y-axis rotation                
                rotor.inputs.y_axis_rotation = conditions.propulsion['propulsor_group_' + str(ii)].y_axis_rotation    
                        
                # link 
                motor.inputs.voltage    = esc.outputs.voltageout
                motor.inputs.rotor_CP   = np.atleast_2d(conditions.propulsion['propulsor_group_' + str(ii)].rotor.power_coefficient[:,ii]).T # TO CHANGE
                
                # step 3
                motor.omega(conditions)
                
                # link
                rotor.inputs.omega         = motor.outputs.omega 
                
                # step 4
                F, Q, P, Cp, outputs, etap = rotor.spin(conditions)
                    
                # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
                eta        = conditions.propulsion['propulsor_group_' + str(ii)].throttle[:,0,None]
                P[eta>1.0] = P[eta>1.0]*eta[eta>1.0]
                F[eta[:,0]>1.0,:] = F[eta[:,0]>1.0,:]*eta[eta[:,0]>1.0,:]
    
                # Run the motor for current
                _ , etam =  motor.current(conditions)
                
                # Conditions specific to this instantation of motor and rotors
                R                   = rotor.tip_radius
                rpm                 = motor.outputs.omega / Units.rpm
                F_mag               = np.atleast_2d(np.linalg.norm(F, axis=1)).T
                total_thrust        = total_thrust + F * factor[ii]
                total_power         = total_power  + P * factor[ii]
                total_motor_current = total_motor_current + factor[ii]*motor.outputs.current
    
                # Pack specific outputs
                conditions.propulsion['propulsor_group_' + str(ii)].motor.efficiency[:,ii]        = etam[:,0]  
                conditions.propulsion['propulsor_group_' + str(ii)].motor.torque[:,ii]            = motor.outputs.torque[:,0]
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.torque[:,ii]            = Q[:,0]
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.thrust[:,ii]            = np.linalg.norm(total_thrust ,axis = 1) 
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.rpm[:,ii]               = rpm[:,0]
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.tip_mach[:,ii]          = (R*rpm[:,0]*Units.rpm)/a[:,0]
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.disc_loading[:,ii]      = (F_mag[:,0])/(np.pi*(R**2)) # N/m^2                  
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.power_loading[:,ii]     = (F_mag[:,0])/(P[:,0])       # N/W      
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.efficiency[:,ii]        = etap[:,0]  
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.figure_of_merit[:,ii]   = outputs.figure_of_merit[:,0]  
                conditions.propulsion['propulsor_group_' + str(ii)].throttle[:,ii]                = eta[:,0] 
                conditions.noise.sources.rotors[rotor.tag]                                        = outputs 

                identical_rotors = np.where(rotor_indexes[ii] == rotor_group_indexes)[0] 
                for idx in range(len(identical_rotors)) :
                    identical_rotor = self.rotors[list(props.keys())[rotor_group_indexes[idx]]]
                    identical_rotor.inputs.y_axis_rotation = conditions.propulsion['propulsor_group_' + str(ii)].y_axis_rotation 
                    if rotor.Wake.wake_method=="Fidelity_One":

                        # make copy of rotor wake and vortex distribution
                        base_wake = copy.deepcopy(rotor.Wake)
                        wake_vd   = base_wake.vortex_distribution
                        
                        # apply offset 
                        origin_offset = np.array(identical_rotor.origin[0]) - np.array(rotor.origin[0])
                        identical_rotor.Wake = base_wake
                        identical_rotor.Wake.shift_wake_VD(wake_vd, origin_offset)   
                        
                    elif rotor.Wake.wake_method=="Fidelity_Zero":
                        identical_rotor.outputs = outputs 
                 
                    
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
     
        return results
     
    def unpack_unknowns(self,segment):
        """ This is an extra set of unknowns which are unpacked from the mission solver and send to the network.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            state.unknowns.rotor_power_coefficient              [None] 
            unknowns specific to the battery cell 
    
            Outputs:
            state.conditions.propulsion.rotor_power_coefficient [None] 
            conditions specific to the battery cell
    
            Properties Used:
            N/A
        """                          
        
        # unpack the ones function
        ones_row               = segment.state.ones_row  
        rotor_group_indexes    = self.rotor_group_indexes
        motor_group_indexes    = self.motor_group_indexes        
        
        # How many evaluations to do 
        unique_rotor_groups,factors = np.unique(rotor_group_indexes, return_counts=True)
        unique_motor_groups,factors = np.unique(motor_group_indexes, return_counts=True) 

        y_rotations   = []
        net           = list(segment.analyses.energy.network.keys())[0]
        for rotor in segment.analyses.energy.network[net].rotors:
            y_rotations.append(rotor.inputs.y_axis_rotation) 
        
        # Here we are going to unpack the unknowns (Cp) provided for this network
        ss       = segment.state 
        n_groups = ss.conditions.propulsion.number_of_propulsor_groups  
        for i in range(n_groups):  
            if segment.battery_discharge:            
                ss.conditions.propulsion['propulsor_group_' + str(i)].rotor.power_coefficient = ss.unknowns['rotor_power_coefficient_' + str(i)]  
            else: 
                ss.conditions.propulsion['propulsor_group_' + str(i)].rotor.power_coefficient = 0. * ones_row(1)
            loc = np.where(unique_rotor_groups[i] == rotor_group_indexes)[0][0]
            ss.conditions.propulsion['propulsor_group_' + str(i)].y_axis_rotation = y_rotations[loc] * ones_row(1)
                                
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
            state.unknowns.y_axis_rotations                [rad] 
            state.unknowns.rotor_power_coefficient              [None] 
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
        ss       = segment.state 
        n_groups = ss.conditions.propulsion.number_of_propulsor_groups 
        
        for i in range(n_groups):  
            if segment.battery_discharge:    
                ss.conditions.propulsion['propulsor_group_' + str(i)].rotor.power_coefficient = ss.unknowns['rotor_power_coefficient_' + str(i)]  
                ss.conditions.propulsion['propulsor_group_' + str(i)].y_axis_rotation         = ss.unknowns['y_axis_rotation_' + str(i)]        
                ss.conditions.propulsion.throttle                                             = ss.unknowns.throttle           
            else: 
                ss.conditions.propulsion['propulsor_group_' + str(i)].rotor.power_coefficient = 0. * ones_row(1)
                ss.conditions.propulsion['propulsor_group_' + str(i)].y_axis_rotation         = 0. * ones_row(1)
               
        
        # update y axis rotation
        self.y_axis_rotation = ss.conditions.propulsion['propulsor_group_' + str(i)].y_axis_rotation
                
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
            segment.state.residuals.network.rotors = Residuals()  
            for i in range(segment.state.conditions.propulsion.number_of_propulsor_groups):
                q_motor   = segment.state.conditions.propulsion['propulsor_group_' + str(i)].motor.torque
                q_prop    = segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.torque 
                segment.state.residuals.network.rotors['propulsor_group_' + str(i)] = q_motor - q_prop
                
        network       = self
        battery       = self.battery 
        battery.append_battery_residuals(segment,network)           
         
        return     
    
    ## @ingroup Components-Energy-Networks
    def add_unknowns_and_residuals_to_segment(self, segment, initial_voltage = None, initial_rotor_power_coefficients = None,
                                              initial_battery_cell_temperature = 283. , initial_battery_state_of_charge = 0.5,
                                              initial_battery_cell_current = 5.):
        """ This function sets up the information that the mission needs to run a mission segment using this network
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            segment
            initial_voltage                                        [v]
            initial_rotor_power_coefficients                              [float]s
            
            Outputs:
            segment.state.unknowns.battery_voltage_under_load
            segment.state.unknowns.rotor_power_coefficient
            segment.state.conditions.propulsion.motor.torque
            segment.state.conditions.propulsion.rotor.torque   
    
            Properties Used:
            N/A
        """            
        rotor_group_indexes    = self.rotor_group_indexes
        motor_group_indexes    = self.motor_group_indexes
        n_rotors               = len(self.rotors)
        n_motors               = len(self.motors)
        
        # unpack the ones function
        ones_row = segment.state.ones_row
                
        # Count how many unknowns and residuals based on p)  
        if n_rotors!=len(rotor_group_indexes):
            assert('The number of rotor group indexes must be equal to the number of rotors')
        if n_motors!=len(motor_group_indexes):
            assert('The number of motor group indexes must be equal to the number of motors') 
        if len(rotor_group_indexes)!=len(motor_group_indexes):
            assert('The number of rotors is not the same as the number of motors')
            
        # Count the number of unique pairs of rotors and motors to determine number of unique pairs of residuals and unknowns 
        unique_rotor_groups = np.unique(rotor_group_indexes)
        unique_motor_groups = np.unique(motor_group_indexes)
        if (unique_rotor_groups == unique_motor_groups).all(): # rotors and motors are paired 
            rotor_indexes = unique_rotor_groups
            n_groups      = len(unique_rotor_groups) 
        else:
            rotor_indexes = rotor_group_indexes
            n_groups      = len(rotor_group_indexes)  
        segment.state.conditions.propulsion.number_of_propulsor_groups = n_groups
            
        # unpack the initial values if the user doesn't specify
        if initial_voltage==None:
            initial_voltage = self.battery.pack.max_voltage    
                
        if initial_rotor_power_coefficients==None:  
            initial_rotor_power_coefficients = []
            for i in range(n_groups):             
                identical_rotor = self.rotors[list(self.rotors.keys())[rotor_indexes[i]]] 
                if type(identical_rotor) == Propeller:
                    initial_rotor_power_coefficients.append(float(identical_rotor.cruise.design_power_coefficient))
                if type(identical_rotor) == Lift_Rotor or type(identical_rotor) == Prop_Rotor:
                    initial_rotor_power_coefficients.append(float(identical_rotor.hover.design_power_coefficient))  

        # Assign initial segment conditions to segment if missing
        battery = self.battery
        append_initial_battery_conditions(segment,battery)          
        
        # add unknowns and residuals specific to battery cell 
        segment.state.residuals.network = Residuals()  
        battery.append_battery_unknowns_and_residuals_to_segment(segment,initial_voltage,
                                              initial_battery_cell_temperature , initial_battery_state_of_charge,
                                              initial_battery_cell_current)  

        if segment.battery_discharge:
            for i in range(n_groups): 
                segment.state.unknowns['rotor_power_coefficient_' + str(i)] = initial_rotor_power_coefficients[i] * ones_row(n_groups)  
        
    
        for i in range(n_groups):         
            # Setup the conditions
            segment.state.conditions.propulsion['propulsor_group_' + str(i)]                        = SUAVE.Analyses.Mission.Segments.Conditions.Conditions()
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].motor                  = SUAVE.Analyses.Mission.Segments.Conditions.Conditions()
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor                  = SUAVE.Analyses.Mission.Segments.Conditions.Conditions()
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].throttle               = 0. * ones_row(n_groups)  
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].y_axis_rotation        = 0. * ones_row(n_groups) 
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].motor.efficiency       = 0. * ones_row(n_groups)
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].motor.torque           = 0. * ones_row(n_groups) 
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.torque           = 0. * ones_row(n_groups)
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.thrust           = 0. * ones_row(n_groups)
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.rpm              = 0. * ones_row(n_groups)
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.disc_loading     = 0. * ones_row(n_groups)                 
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.power_loading    = 0. * ones_row(n_groups)
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.tip_mach         = 0. * ones_row(n_groups)
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.efficiency       = 0. * ones_row(n_groups)   
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.figure_of_merit  = 0. * ones_row(n_groups) 
            
        # Ensure the mission knows how to pack and unpack the unknowns and residuals
        segment.process.iterate.unknowns.network  = self.unpack_unknowns
        segment.process.iterate.residuals.network = self.residuals        

        return segment
    
    ## @ingroup Components-Energy-Networks
    def add_tiltrotor_transition_unknowns_and_residuals_to_segment(self, segment, initial_voltage = None, 
                                                        initial_y_axis_rotations = None,
                                                        initial_rotor_power_coefficients = None,
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
            initial_rotor_power_coefficients         [float]
            initial_battery_cell_temperature  [float]
            initial_battery_state_of_charge   [float]
            initial_battery_cell_current      [float]
            
            Outputs:
            segment.state.unknowns.battery_voltage_under_load
            segment.state.unknowns.rotor_power_coefficient
            segment.state.unknowns.throttle
            segment.state.unknowns.y_axis_rotations
            segment.state.conditions.propulsion.motor.torque
            segment.state.conditions.propulsion.rotor.torque   
    
            Properties Used:
            N/A
        """            
        rotor_group_indexes    = self.rotor_group_indexes
        motor_group_indexes    = self.motor_group_indexes
        n_rotors               = len(self.rotors)
        n_motors               = len(self.motors)
        
        # unpack the ones function
        ones_row = segment.state.ones_row 
        
        # Count how many unknowns and residuals based on p)  
        if n_rotors!=len(rotor_group_indexes):
            assert('The number of rotor group indexes must be equal to the number of rotors')
        if n_motors!=len(motor_group_indexes):
            assert('The number of motor group indexes must be equal to the number of motors') 
        if len(rotor_group_indexes)!=len(motor_group_indexes):
            assert('The number of rotors is not the same as the number of motors')
            
        # Count the number of unique pairs of rotors and motors to determine number of unique pairs of residuals and unknowns 
        unique_rotor_groups = np.unique(rotor_group_indexes)
        unique_motor_groups = np.unique(motor_group_indexes)
        if (unique_rotor_groups == unique_motor_groups).all(): # rotors and motors are paired 
            rotor_indexes = unique_rotor_groups
            n_groups      = len(unique_rotor_groups) 
        else:
            rotor_indexes = rotor_group_indexes
            n_groups      = len(rotor_group_indexes)  
        segment.state.conditions.propulsion.number_of_propulsor_groups = n_groups  
    
        # unpack the initial values if the user doesn't specify
        if initial_voltage==None:
            initial_voltage = self.battery.pack.max_voltage
            
        if initial_rotor_power_coefficients==None:  
            initial_rotor_power_coefficients = []
            for i in range(n_groups):             
                identical_rotor = self.rotors[list(self.rotors.keys())[rotor_indexes[i]]] 
                if type(identical_rotor) == Propeller:
                    initial_rotor_power_coefficients.append(float(identical_rotor.cruise.design_power_coefficient))
                if type(identical_rotor) == Lift_Rotor or type(identical_rotor) == Prop_Rotor:
                    initial_rotor_power_coefficients.append(float(identical_rotor.hover.design_power_coefficient))  

            
        if initial_y_axis_rotations==None:  
            initial_y_axis_rotations = list(np.zeros(n_groups))
                        
        # Assign initial segment conditions to segment if missing
        battery = self.battery
        append_initial_battery_conditions(segment,battery)          
        
        # add unknowns and residuals specific to battery cell 
        segment.state.residuals.network = Residuals()  
        battery.append_battery_unknowns_and_residuals_to_segment(segment,initial_voltage,
                                              initial_battery_cell_temperature , initial_battery_state_of_charge,
                                              initial_battery_cell_current)  
  
        if segment.battery_discharge:
            for i in range(n_groups): 
                segment.state.unknowns['rotor_power_coefficient_' + str(i)] = initial_rotor_power_coefficients[i] * ones_row(n_groups)  
                segment.state.unknowns['y_axis_rotation_' + str(i)]         = initial_y_axis_rotations[i] * ones_row(1) 
                segment.state.unknowns.throttle                             = 0.7 * ones_row(1)  
    
        for i in range(n_groups):         
            # Setup the conditions
            segment.state.conditions.propulsion['propulsor_group_' + str(i)]       = SUAVE.Analyses.Mission.Segments.Conditions.Conditions()
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].motor = SUAVE.Analyses.Mission.Segments.Conditions.Conditions()
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor = SUAVE.Analyses.Mission.Segments.Conditions.Conditions()
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].throttle               = 0. * ones_row(n_groups)  
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].y_axis_rotation        = 0. * ones_row(n_groups) 
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].motor.efficiency       = 0. * ones_row(n_groups)
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].motor.torque           = 0. * ones_row(n_groups) 
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.torque           = 0. * ones_row(n_groups)
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.thrust           = 0. * ones_row(n_groups)
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.rpm              = 0. * ones_row(n_groups)
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.disc_loading     = 0. * ones_row(n_groups)                 
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.power_loading    = 0. * ones_row(n_groups)
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.tip_mach         = 0. * ones_row(n_groups)
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.efficiency       = 0. * ones_row(n_groups)   
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.figure_of_merit  = 0. * ones_row(n_groups) 
            
        
        # Ensure the mission knows how to pack and unpack the unknowns and residuals
        segment.process.iterate.unknowns.network  = self.unpack_tiltrotor_transition_unknowns
        segment.process.iterate.residuals.network = self.residuals   
        segment.process.iterate.unknowns.mission  = SUAVE.Methods.skip

        return segment    
       
    
    __call__ = evaluate_thrust


