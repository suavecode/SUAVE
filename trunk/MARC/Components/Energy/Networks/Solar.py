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
import MARC
import numpy as np
from .Network import Network
from MARC.Components.Physical_Component import Container
from MARC.Analyses.Mission.Segments.Conditions import Residuals
from MARC.Methods.Power.Battery.pack_battery_conditions import pack_battery_conditions
from MARC.Components.Energy.Converters   import Propeller, Lift_Rotor, Prop_Rotor 
from MARC.Methods.Power.Battery.append_initial_battery_conditions import append_initial_battery_conditions 
from MARC.Core import Data , Units

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Solar(Network):
    """ A solar powered system with batteries and maximum power point tracking.
        
        This network adds an extra unknowns to the mission, the torque matching between motor and rotor.
    
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
        self.solar_flux                      = None
        self.solar_panel                     = None
        self.motors                          = Container()
        self.rotors                          = Container()
        self.electronic_speed_controllers    = Container()
        self.avionics                        = None
        self.payload                         = None
        self.solar_logic                     = None
        self.battery                         = None 
        self.engine_length                   = None
        self.number_of_engines               = None
        self.tag                             = 'Solar'
        self.use_surrogate                   = False
        self.generative_design_minimum       = 0
        self.rotor_group_indexes             = [0]
        self.motor_group_indexes             = [0] 
        self.active_propulsor_groups         = [True]
    
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
            results.thrust_force_vector        [newtons]
            results.vehicle_mass_rate          [kg/s]
            conditions.propulsion:
                       solar_flux              [watts/m^2] 
                       rpm                     [radians/sec]
                       battery.
                         pack.
                            current            [amps]
                            power_draw         [watts]
                         energy.               [joules]
                       motor.torque            [N-M]
                       rotor.torque            [N-M]
    
            Properties Used:
            Defaulted values
        """          
    
        # unpack
        conditions              = state.conditions
        numerics                = state.numerics
        solar_flux              = self.solar_flux
        solar_panel             = self.solar_panel
        motors                  = self.motors
        rotors                  = self.rotors
        escs                    = self.electronic_speed_controllers
        avionics                = self.avionics
        payload                 = self.payload
        solar_logic             = self.solar_logic
        battery                 = self.battery
        rotor_group_indexes     = self.rotor_group_indexes
        motor_group_indexes     = self.motor_group_indexes
        active_propulsor_groups = self.active_propulsor_groups
        
        # Unpack conditions
        a = conditions.freestream.speed_of_sound        
        
        # Set battery energy
        battery.pack.current_energy           = conditions.propulsion.battery.pack.energy
        battery.pack.temperature              = conditions.propulsion.battery.pack.temperature
        battery.pack.max_energy               = conditions.propulsion.battery.pack.max_aged_energy   
        battery.cell.charge_throughput        = conditions.propulsion.battery.cell.charge_throughput     
        battery.cell.age                      = conditions.propulsion.battery.cell.cycle_in_day            
        battery.cell.R_growth_factor          = conditions.propulsion.battery.cell.resistance_growth_factor
        battery.cell.E_growth_factor          = conditions.propulsion.battery.cell.capacity_fade_factor 
        
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
        

        # Iterate over motor/rotors
        for ii in range(n_evals):
            if active_propulsor_groups[ii]:
                motor_key = list(motors.keys())[motor_indexes[ii]]
                rotor_key = list(rotors.keys())[rotor_indexes[ii]]
                esc_key   = list(escs.keys())[motor_indexes[ii]]
                motor     = motors[motor_key]
                rotor     = rotors[rotor_key]
                esc       = escs[esc_key]
                
                # Step 1 battery power
                esc.inputs.voltagein = solar_logic.outputs.system_voltage
                
                # Step 2 throttle the voltage
                esc.voltageout(conditions.propulsion['propulsor_group_' + str(ii)].throttle)    
        
                # link
                motor.inputs.voltage  = esc.outputs.voltageout
                motor.inputs.rotor_CP = conditions.propulsion['propulsor_group_' + str(ii)].rotor.power_coefficient
                
                # step 5
                motor.omega(conditions)
                
                # link
                rotor.inputs.omega =  motor.outputs.omega
                
                # step 6
                F, Q, P, Cplast ,  outputs  , etap   = rotor.spin(conditions)
             
                # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
                eta = conditions.propulsion.throttle[:,0,None]
                P[eta>1.0] = P[eta>1.0]*eta[eta>1.0]
                F[eta[:,0]>1.0,:] = F[eta[:,0]>1.0,:]*eta[eta[:,0]>1.0,:]
                
                # Run the motor for current
                _ , etam =  motor.current(conditions)         
                
                # Conditions specific to this instantation of motor and rotors
                R                   = rotor.tip_radius
                rpm                 = motor.outputs.omega / Units.rpm
                F_mag               = np.atleast_2d(np.linalg.norm(F, axis=1)).T
                total_thrust        = total_thrust + F * factor
                total_power         = total_power  + P * factor
                total_motor_current = total_motor_current + factor*motor.outputs.current
    
                # Pack specific outputs
                conditions.propulsion['propulsor_group_' + str(ii)].motor.efficiency       = etam  
                conditions.propulsion['propulsor_group_' + str(ii)].motor.torque           = motor.outputs.torque
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.torque           = Q
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.thrust           = np.linalg.norm(total_thrust ,axis = 1) 
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.rpm              = rpm
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.tip_mach         = (R*rpm*Units.rpm)/a
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.disc_loading     = (F_mag)/(np.pi*(R**2)) # N/m^2                  
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.power_loading    = (F_mag)/(P)      # N/W      
                conditions.propulsion['propulsor_group_' + str(ii)].rotor.efficiency       = etap  
                conditions.propulsion['propulsor_group_' + str(ii)].throttle               = eta 
                conditions.noise.sources.rotors[rotor.tag]                                 = outputs
            
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
        esc.currentin(conditions.propulsion.throttle)
        
        # link
        solar_logic.inputs.currentesc  = esc.outputs.currentin
        solar_logic.logic(conditions,numerics)
        
        # link
        battery.inputs = solar_logic.outputs
        battery.energy_calc(numerics)
        
        # Calculate avionics and payload power
        avionics_payload_power = avionics.outputs.power + payload.outputs.power
        
        # Pack the conditions for outputs 
        conditions.propulsion.solar_flux   = solar_flux.outputs.flux          
        pack_battery_conditions(conditions,battery,avionics_payload_power)  

        # Create the outputs
        results = Data()
        results.thrust_force_vector     = total_thrust
        results.vehicle_mass_rate       = state.ones_row(1)*0.0 

        return results
    
    
    def unpack_unknowns(self,segment):
        """ This is an extra set of unknowns which are unpacked from the mission solver and send to the network.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            state.unknowns.rotor_power_coefficient [None]
    
            Outputs:
            state.conditions.propulsion.rotor_power_coefficient [None]
    
            Properties Used:
            N/A
        """       
         
        # Here we are going to unpack the unknowns (Cp) provided for this network
        ss       = segment.state 
        n_groups = ss.conditions.propulsion.number_of_propulsor_groups   
        for i in range(n_groups):   
            if segment.battery_discharge:           
                ss.conditions.propulsion['propulsor_group_' + str(i)].rotor.power_coefficient = ss.unknowns['rotor_power_coefficient_' + str(i)] 
                    
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
                rotor_torque                      [N-m]
            
            Outputs:
            None
    
            Properties Used:
            None
        """  
         
        for i in range(segment.state.conditions.propulsion.number_of_propulsor_groups): 
            q_motor   = segment.state.conditions.propulsion['propulsor_group_' + str(i)].motor.torque
            q_prop    = segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.torque 
            segment.state.residuals.network['propulsor_group_' + str(i)] = q_motor - q_prop 
        
        return
    
    
    
    def add_unknowns_and_residuals_to_segment(self, segment, initial_rotor_power_coefficients = None):
        """ This function sets up the information that the mission needs to run a mission segment using this network
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            segment
            initial_voltage                          [v]
            initial_rotor_power_coefficients         [float]s
            
            Outputs:
            segment.state.unknowns.rotor_power_coefficient
            segment.state.conditions.propulsion.motor.torque
            segment.state.conditions.propulsion.rotor_torque   
    
            Properties Used:
            N/A
        """           
        
        # unpack the ones function
        ones_row = segment.state.ones_row
        
        # Count how many unknowns and residuals based on p

        rotor_group_indexes     = self.rotor_group_indexes
        motor_group_indexes     = self.motor_group_indexes
        active_propulsor_groups = segment.analyses.energy.network.solar.active_propulsor_groups
        n_rotors                = len(self.rotors)
        n_motors                = len(self.motors)
         


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
                
        if len(active_propulsor_groups)!= n_groups:
            assert('The dimension of propulsor groups rotors must be equal to the number of distinct groups')            
        segment.state.conditions.propulsion.number_of_propulsor_groups = n_groups 
              
            
        # unpack the initial values if the user doesn't specify
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
        segment.state.residuals.network = Residuals() 

        for i in range(n_groups):  
            if active_propulsor_groups[i]:                
                segment.state.residuals.network['propulsor_group_' + str(i)] =  0. * ones_row(1)              
                segment.state.unknowns['rotor_power_coefficient_' + str(i)]  = initial_rotor_power_coefficients[i] * ones_row(1)  
 

        for i in range(n_groups):         
            # Setup the conditions
            segment.state.conditions.propulsion['propulsor_group_' + str(i)]       = MARC.Analyses.Mission.Segments.Conditions.Conditions()
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].motor = MARC.Analyses.Mission.Segments.Conditions.Conditions()
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor = MARC.Analyses.Mission.Segments.Conditions.Conditions()
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].throttle               = 0. * ones_row(n_groups)   
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
            
    __call__ = evaluate_thrust
