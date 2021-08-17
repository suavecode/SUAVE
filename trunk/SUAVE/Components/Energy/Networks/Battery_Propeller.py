## @ingroup Components-Energy-Networks
# Battery_Propeller.py
# 
# Created:  Jul 2015, E. Botero
# Modified: Feb 2016, T. MacDonald
#           Mar 2020, M. Clarke 
#           Apr 2021, M. Clarke
#           Jul 2021, E. Botero
#           Jul 2021, R. Erhard
#           Aug 2021, M. Clarke 
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np
from .Network import Network
from SUAVE.Components.Physical_Component import Container
from SUAVE.Components.Energy.Storages.Batteries.Constant_Mass import Lithium_Ion_LiFePO4_38120, Lithium_Ion_LiNCA_18650, Lithium_Ion_LiNiMnCoO2_18650
from SUAVE.Core import Data , Units

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Battery_Propeller(Network):
    """ This is a simple network with a battery powering a propeller through
        an electric motor
        
        This network adds 2 extra unknowns to the mission. The first is
        a voltage, to calculate the thevenin voltage drop in the pack.
        The second is torque matching between motor and propeller.
    
        Assumptions:
        The y axis rotation is used for rotating the propeller about the Y-axis for tilt rotors and tiltwings
        
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
        self.propeller_motors             = Container()
        self.propellers                   = Container()
        self.esc                          = None
        self.avionics                     = None
        self.payload                      = None
        self.battery                      = None
        self.nacelle_diameter             = None
        self.engine_length                = None
        self.number_of_propeller_engines  = None
        self.voltage                      = None
        self.tag                          = 'Battery_Propeller'
        self.use_surrogate                = False
        self.generative_design_minimum    = 0
        self.identical_propellers         = True
        self.thrust_angle                 = 0. 
    
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
                propeller_torque             [N-M]
    
            Properties Used:
            Defaulted values
        """          
    
        # unpack  
        conditions   = state.conditions
        numerics     = state.numerics
        motors       = self.propeller_motors
        props        = self.propellers
        esc          = self.esc
        avionics     = self.avionics
        payload      = self.payload
        battery      = self.battery
        num_engines  = self.number_of_propeller_engines 
        D            = numerics.time.differentiate        
        battery_data = battery.discharge_performance_map  
        
        # Set battery energy
        battery.current_energy      = conditions.propulsion.battery_energy
        battery.pack_temperature    = conditions.propulsion.battery_pack_temperature
        battery.charge_throughput   = conditions.propulsion.battery_cumulative_charge_throughput     
        battery.age_in_days         = conditions.propulsion.battery_age_in_days 
        discharge_flag              = conditions.propulsion.battery_discharge    
        battery.R_growth_factor     = conditions.propulsion.battery_resistance_growth_factor
        battery.E_growth_factor     = conditions.propulsion.battery_capacity_fade_factor 
        battery.max_energy          = conditions.propulsion.battery_max_aged_energy
        V_th0                       = conditions.propulsion.battery_initial_thevenin_voltage
        n_series                    = battery.pack_config.series  
        n_parallel                  = battery.pack_config.parallel
        n_total                     = n_series*n_parallel
        
        # update ambient temperature based on altitude
        battery.ambient_temperature                   = conditions.freestream.temperature   
        battery.cooling_fluid.thermal_conductivity    = conditions.freestream.thermal_conductivity
        battery.cooling_fluid.kinematic_viscosity     = conditions.freestream.kinematic_viscosity
        battery.cooling_fluid.density                 = conditions.freestream.density  
        
        # Unpack conditions
        a = conditions.freestream.speed_of_sound
        
        # Set battery energy
        battery.current_energy = conditions.propulsion.battery_energy  


        # --------------------------------------------------------------------------------
        # Predict Voltage and Battery Properties Depending on Battery Chemistry
        # --------------------------------------------------------------------------------  
        if isinstance(battery,Lithium_Ion_LiFePO4_38120):
            volts                            = state.unknowns.battery_voltage_under_load
            battery.battery_thevenin_voltage = 0             
            battery.temperature              = conditions.propulsion.battery_pack_temperature
            
        elif isinstance(battery,Lithium_Ion_LiNCA_18650):  
            
            SOC       = state.unknowns.battery_state_of_charge
            T_cell    = state.unknowns.battery_cell_temperature
            V_Th_cell = state.unknowns.battery_thevenin_voltage/n_series
            
            # link temperature 
            battery.cell_temperature = T_cell     
            
            # look up tables  
            V_oc_cell = np.zeros_like(SOC)
            R_Th_cell = np.zeros_like(SOC)
            C_Th_cell = np.zeros_like(SOC)
            R_0_cell  = np.zeros_like(SOC)
            SOC[SOC<0.] = 0.
            SOC[SOC>1.] = 1.
            for i in range(len(SOC)): 
                T_cell_Celcius = T_cell[i] - 272.65
                V_oc_cell[i] = battery_data.V_oc_interp(T_cell_Celcius, SOC[i])[0]
                C_Th_cell[i] = battery_data.C_Th_interp(T_cell_Celcius, SOC[i])[0]
                R_Th_cell[i] = battery_data.R_Th_interp(T_cell_Celcius, SOC[i])[0]
                R_0_cell[i]  = battery_data.R_0_interp( T_cell_Celcius, SOC[i])[0]  
                
            dV_TH_dt =  np.dot(D,V_Th_cell)
            I_cell   = V_Th_cell/(R_Th_cell * battery.R_growth_factor)  + C_Th_cell*dV_TH_dt
            R_0_cell = R_0_cell * battery.R_growth_factor
             
            # Voltage under load:
            volts =  n_series*(V_oc_cell - V_Th_cell - (I_cell  * R_0_cell)) 

        elif isinstance(battery,Lithium_Ion_LiNiMnCoO2_18650): 

            SOC        = state.unknowns.battery_state_of_charge 
            T_cell     = state.unknowns.battery_cell_temperature
            I_cell     = state.unknowns.battery_current/n_parallel 
            
            # Link Temperature 
            battery.cell_temperature         = T_cell  
            battery.initial_thevenin_voltage = V_th0  
            
            # Make sure things do not break by limiting current, temperature and current 
            SOC[SOC < 0.]            = 0.  
            SOC[SOC > 1.]            = 1.    
            DOD                      = 1 - SOC 
            
            T_cell[np.isnan(T_cell)] = 302.65
            T_cell[T_cell<272.65]    = 272.65 # model does not fit for below 0  degrees
            T_cell[T_cell>322.65]    = 322.65 # model does not fit for above 50 degrees
             
            I_cell[I_cell<0.0]       = 0.0
            I_cell[I_cell>8.0]       = 8.0   
            
            # create vector of conditions for battery data sheet response surface for OCV
            T_cell_Celcius           = T_cell  - 272.65
            pts                      = np.hstack((np.hstack((I_cell, T_cell_Celcius)),DOD  )) # amps, temp, SOC   
            V_ul_cell                = np.atleast_2d(battery_data.Voltage(pts)[:,1]).T   
            volts                    = n_series*V_ul_cell   
 
            
        # --------------------------------------------------------------------------------
        # Run Motor, Avionics and Systems (Discharge Model)
        # --------------------------------------------------------------------------------    
        if discharge_flag:    
            # Step 1 battery power
            esc.inputs.voltagein = volts
            
            # Step 2
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
                prop_key  = list(props.keys())[ii]
                motor     = self.propeller_motors[motor_key]
                prop      = self.propellers[prop_key]
                
                # link
                motor.inputs.voltage      = esc.outputs.voltageout 
                motor.inputs.propeller_CP = np.atleast_2d(conditions.propulsion.propeller_power_coefficient[:,ii]).T
                
                # step 3
                motor.omega(conditions)
                
                # link
                prop.inputs.omega           = motor.outputs.omega
                prop.inputs.pitch_command   = self.pitch_command
                prop.inputs.y_axis_rotation = self.thrust_angle
                
                # step 4
                F, Q, P, Cp, outputs, etap = prop.spin(conditions)
                    
                # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
                eta        = conditions.propulsion.throttle[:,0,None]
                P[eta>1.0] = P[eta>1.0]*eta[eta>1.0]
                F[eta[:,0]>1.0,:] = F[eta[:,0]>1.0,:]*eta[eta[:,0]>1.0,:]
    
                # Run the motor for current
                motor.current(conditions)
                
                # Conditions specific to this instantation of motor and propellers
                R                   = prop.tip_radius
                rpm                 = motor.outputs.omega / Units.rpm
                F_mag               = np.atleast_2d(np.linalg.norm(F, axis=1)).T
                total_thrust        = total_thrust + F * factor
                total_power         = total_power  + P * factor
                total_motor_current = total_motor_current + factor*motor.outputs.current
    
                # Pack specific outputs
                conditions.propulsion.propeller_motor_torque[:,ii] = motor.outputs.torque[:,0]
                conditions.propulsion.propeller_torque[:,ii]       = Q[:,0]
                conditions.propulsion.propeller_rpm[:,ii]          = rpm[:,0]
                conditions.propulsion.propeller_tip_mach[:,ii]     = (R*rpm[:,0]*Units.rpm)/a[:,0]
                conditions.propulsion.disc_loading[:,ii]           = (F_mag[:,0])/(np.pi*(R**2)) # N/m^2                  
                conditions.propulsion.power_loading[:,ii]          = (F_mag[:,0])/(P[:,0])      # N/W      
                conditions.propulsion.propeller_efficiency         = etap[:,0]      
                
                conditions.noise.sources.propellers[prop.tag]      = outputs
    
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
            battery.inputs.current  = esc.outputs.currentin + avionics_payload_current
            battery.inputs.power_in = -(esc.outputs.voltageout*esc.outputs.currentin + avionics_payload_power)
            battery.energy_calc(numerics)         

        # --------------------------------------------------------------------------------
        # Run Charge Model 
        # --------------------------------------------------------------------------------               
        else:  
            # link 
            battery.inputs.current  = -battery.cell.charging_current * n_parallel * np.ones_like(volts)
            battery.inputs.power_in =  battery.cell.charging_current * n_parallel * volts * np.ones_like(volts)
            battery.inputs.voltage  =  battery.charging_voltage 
            
            # Run the avionics
            avionics.power()
        
            # Run the payload
            payload.power()
        
            # link
            esc.inputs.currentout = total_motor_current
        
            # Run the esc
            esc.currentin(conditions)  
            
            battery.inputs.current  = esc.outputs.currentin + avionics_payload_current
            battery.inputs.power_in = -(esc.outputs.voltageout*esc.outputs.currentin + avionics_payload_power)           
            battery.energy_charge(numerics)        
            
            conditions.propulsion.power_loading   = np.zeros_like(volts)     
            
        # Pack the conditions for outputs
        conditions.propulsion.battery_current                      = esc.outputs.currentin 
        conditions.propulsion.battery_energy                       = battery.current_energy
        conditions.propulsion.battery_voltage_open_circuit         = battery.voltage_open_circuit
        conditions.propulsion.battery_voltage_under_load           = battery.voltage_under_load
        conditions.propulsion.state_of_charge                      = battery.state_of_charge
        conditions.propulsion.battery_specfic_power                = -battery.inputs.power_in/battery.mass_properties.mass # Wh/kg 
        conditions.propulsion.battery_power_draw                   = battery.inputs.power_in 
        conditions.propulsion.battery_max_aged_energy              = battery.max_energy 
        conditions.propulsion.battery_cumulative_charge_throughput = battery.cumulative_cell_charge_throughput 
        conditions.propulsion.battery_charge_throughput            = battery.cell_charge_throughput 
        conditions.propulsion.battery_state_of_charge              = battery.state_of_charge 
        conditions.propulsion.battery_pack_temperature             = battery.pack_temperature 
        conditions.propulsion.battery_thevenin_voltage             = battery.thevenin_voltage           
        conditions.propulsion.battery_age_in_days                  = battery.age_in_days  

        conditions.propulsion.battery_cell_power_draw              = battery.inputs.power_in /n_series
        conditions.propulsion.battery_cell_energy                  = battery.current_energy/n_total   
        conditions.propulsion.battery_cell_voltage_under_load      = battery.cell_voltage_under_load    
        conditions.propulsion.battery_cell_voltage_open_circuit    = battery.cell_voltage_open_circuit  
        conditions.propulsion.battery_cell_current                 = abs(battery.cell_current)        
        conditions.propulsion.battery_cell_temperature             = battery.cell_temperature
        conditions.propulsion.battery_cell_heat_energy_generated   = battery.heat_energy_generated
        conditions.propulsion.battery_cell_joule_heat_fraction     = battery.cell_joule_heat_fraction   
        conditions.propulsion.battery_cell_entropy_heat_fraction   = battery.cell_entropy_heat_fraction 
        
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
            state.unknowns.battery_voltage_under_load  [volts]
    
            Outputs:
            state.conditions.propulsion.propeller_power_coefficient [None]
            state.conditions.propulsion.battery_voltage_under_load  [volts]
    
            Properties Used:
            N/A
        """                  
        
        # Here we are going to unpack the unknowns (Cp) provided for this network
        ss = segment.state
        if segment.battery_discharge: 
            ss.conditions.propulsion.propeller_power_coefficient = ss.unknowns.propeller_power_coefficient  
       
        if isinstance(self.network.battery,Lithium_Ion_LiFePO4_38120):
            ss.conditions.propulsion.battery_voltage_under_load  = ss.unknowns.battery_voltage_under_load
            
        elif isinstance(self.network.battery,Lithium_Ion_LiNiMnCoO2_18650): 
            ss.conditions.propulsion.battery_cell_temperature    = ss.unknowns.battery_cell_temperature 
            ss.conditions.propulsion.battery_state_of_charge     = ss.unknowns.battery_state_of_charge
            ss.conditions.propulsion.battery_current             = ss.unknowns.battery_current   
            
        elif isinstance(self.network.battery,Lithium_Ion_LiNCA_18650): 
            ss.conditions.propulsion.battery_cell_temperature    = ss.unknowns.battery_cell_temperature 
            ss.conditions.propulsion.battery_state_of_charge     = ss.unknowns.battery_state_of_charge
            ss.conditions.propulsion.battery_thevenin_voltage    = ss.unknowns.battery_thevenin_voltage   
            
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
                voltage_under_load                    [volts]
            state.unknowns.battery_voltage_under_load [volts]
            
            Outputs:
            None
    
            Properties Used:
            self.voltage                              [volts]
        """          
            
        if isinstance(self.network.battery,Lithium_Ion_LiFePO4_38120):  
            v_actual  = segment.state.conditions.propulsion.battery_voltage_under_load
            v_predict = segment.state.unknowns.battery_voltage_under_load
            v_max     = self.voltage
            
            # Return the residuals
            segment.state.residuals.network[:,0]  = (v_predict[:,0] - v_actual[:,0])/v_max
            if segment.discharge:    
                q_motor   = segment.state.conditions.propulsion.propeller_motor_torque
                q_prop    = segment.state.conditions.propulsion.propeller_torque                
                segment.state.residuals.network[:,1:] = q_motor - q_prop
            
        elif isinstance(self.network.battery,Lithium_Ion_LiNiMnCoO2_18650):   
            q_motor      = segment.state.conditions.propulsion.motor_torque
            q_prop       = segment.state.conditions.propulsion.propeller_torque  
        
            SOC_actual   = segment.state.conditions.propulsion.battery_state_of_charge
            SOC_predict  = segment.state.unknowns.battery_state_of_charge 
        
            Temp_actual  = segment.state.conditions.propulsion.battery_cell_temperature 
            Temp_predict = segment.state.unknowns.battery_cell_temperature   
        
            i_actual     = segment.state.conditions.propulsion.battery_current
            i_predict    = segment.state.unknowns.battery_current      
        
            # Return the residuals  
            segment.state.residuals.network[:,0]  =  SOC_predict[:,0]  - SOC_actual[:,0]  
            segment.state.residuals.network[:,1]  =  Temp_predict[:,0] - Temp_actual[:,0]
            segment.state.residuals.network[:,2]  =  i_predict[:,0] - i_actual[:,0]  
            if segment.discharge: 
                q_motor   = segment.state.conditions.propulsion.propeller_motor_torque
                q_prop    = segment.state.conditions.propulsion.propeller_torque                
                segment.state.residuals.network[:,3:] =  q_motor - q_prop    
            
            
        elif isinstance(self.network.battery,Lithium_Ion_LiNCA_18650):   
            SOC_actual   = segment.state.conditions.propulsion.battery_state_of_charge
            SOC_predict  = segment.state.unknowns.battery_state_of_charge 
        
            Temp_actual  = segment.state.conditions.propulsion.battery_cell_temperature 
            Temp_predict = segment.state.unknowns.battery_cell_temperature   
        
            v_th_actual  = segment.state.conditions.propulsion.battery_thevenin_voltage
            v_th_predict = segment.state.unknowns.battery_thevenin_voltage        
        
            # Return the residuals   
            segment.state.residuals.network[:,0]  = v_th_predict[:,0] - v_th_actual[:,0]     
            segment.state.residuals.network[:,1]  = SOC_predict[:,0] - SOC_actual[:,0]  
            segment.state.residuals.network[:,2]  = Temp_predict[:,0] - Temp_actual[:,0]
            if segment.discharge: 
                q_motor   = segment.state.conditions.propulsion.propeller_motor_torque
                q_prop    = segment.state.conditions.propulsion.propeller_torque                
                segment.state.residuals.network[:,3:] = q_motor - q_prop  
                
        return     

    def add_unknowns_and_residuals_to_segment(self, segment, initial_voltage = None, initial_power_coefficient = 0.005,
                                              initial_battery_cell_temperature = None, initial_battery_state_of_charge = None ,
                                              initial_battery_current = None,initial_battery_thevenin_voltage= None ):
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
            segment.state.unknowns.propeller_power_coefficient
            segment.state.conditions.propulsion.propeller_motor_torque
            segment.state.conditions.propulsion.propeller_torque   
    
            Properties Used:
            N/A
        """           
        
        # unpack the ones function
        ones_row = segment.state.ones_row
        
        # unpack the initial values if the user doesn't specify
        if initial_voltage==None:
            initial_voltage = self.battery.max_voltage
        
        # Count how many unknowns and residuals based on p
        n_props  = len(self.propellers)
        n_motors = len(self.propeller_motors)
        n_eng    = self.number_of_propeller_engines
        
        if n_props!=n_motors!=n_eng:
            print('The number of propellers is not the same as the number of motors')
            
        # Now check if the propellers are all identical, in this case they have the same of residuals and unknowns
        if self.identical_propellers:
            n_props = 1 

        if isinstance(self.battery,Lithium_Ion_LiFePO4_38120):   
            segment.state.unknowns.battery_voltage_under_load  = initial_voltage * ones_row(1)
            if segment.discharge: 
                n_res = n_props + 1
                segment.state.unknowns.propeller_power_coefficient = initial_power_coefficient * ones_row(n_props)
            else:
                n_res = 1
                
            segment.state.residuals.network = 0. * ones_row(n_res)
            
            
        elif isinstance(self.battery,Lithium_Ion_LiNiMnCoO2_18650):    
            segment.state.unknowns.battery_cell_temperature    = initial_battery_cell_temperature  * ones_row(n_props) 
            segment.state.unknowns.battery_state_of_charge     = initial_battery_state_of_charge   * ones_row(n_props)  
            segment.state.unknowns.battery_current             = initial_battery_current           * ones_row(n_props) 
            if segment.discharge: 
                n_res = n_props + 3  
                segment.state.unknowns.propeller_power_coefficient = initial_power_coefficient * ones_row(n_props) 
            else:
                n_res = 3        
            segment.state.residuals.network = 0. * ones_row(n_res) 
            
            
        elif isinstance(self.battery,Lithium_Ion_LiNCA_18650):   
            segment.state.unknowns.battery_state_of_charge      = initial_battery_state_of_charge   * ones_row(n_props)  
            segment.state.unknowns.battery_cell_temperature     = initial_battery_cell_temperature  * ones_row(n_props)       
            segment.state.unknowns.battery_thevenin_voltage     = initial_battery_thevenin_voltage  * ones_row(n_props)   
            if segment.discharge: 
                n_res = n_props + 3 
                segment.state.unknowns.propeller_power_coefficient = initial_power_coefficient * ones_row(n_props)     
            else:
                n_res = 3         
            segment.state.residuals.network = 0. * ones_row(n_res) 
       
        
        # Setup the conditions
        segment.state.conditions.propulsion.propeller_motor_torque = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller_torque       = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller_rpm          = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.disc_loading           = 0. * ones_row(n_props)                 
        segment.state.conditions.propulsion.power_loading          = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller_tip_mach     = 0. * ones_row(n_props)
        segment.state.conditions.propulsion.propeller_efficiency   = 0. * ones_row(n_props)        
        
        # Ensure the mission knows how to pack and unpack the unknowns and residuals
        segment.process.iterate.unknowns.network  = self.unpack_unknowns
        segment.process.iterate.residuals.network = self.residuals        

        return segment
    
    __call__ = evaluate_thrust


