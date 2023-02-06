## @ingroup Components-Energy-Networks
# Internal_Combustion_Propeller.py
# 
# Created:  Sep 2016, E. Botero
# Modified: Apr 2018, M. Clarke 
#           Mar 2020, M. Clarke 
#           Apr 2021, M. Clarke
#           Jul 2021, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import MARC
import numpy as np
from .Network import Network
from MARC.Components.Physical_Component import Container
from MARC.Core import Data, Units

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Networks
class Internal_Combustion_Propeller(Network):
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
        self.engines                      = Container()
        self.propellers                   = Container()
        self.engine_length                = None
        self.number_of_engines            = None 
        self.rotor_group_indexes          = [0]
        self.motor_group_indexes          = [0] 
    
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
                propeller.torque     [N-M]
                power                [W]
    
            Properties Used:
            Defaulted values
        """           
        # unpack
        conditions              = state.conditions
        engines                 = self.engines
        propellers              = self.propellers 
        rotor_group_indexes     = self.rotor_group_indexes
        motor_group_indexes     = self.motor_group_indexes 
        
        # Unpack conditions
        a = conditions.freestream.speed_of_sound        

        # How many evaluations to do 
        unique_rotor_groups,factors = np.unique(rotor_group_indexes, return_counts=True)
        unique_motor_groups,factors = np.unique(motor_group_indexes, return_counts=True)
        if (unique_rotor_groups == unique_motor_groups).all(): # rotors and motors are paired 
            n_evals = len(unique_rotor_groups)
            rotor_indexes = unique_rotor_groups 
            factor        = factors
        else:
            n_evals = len(rotor_group_indexes)
            rotor_indexes = rotor_group_indexes 
            factor        = np.ones_like(motor_group_indexes) 
            
        # Setup numbers for iteration
        total_thrust        = 0. * state.ones_row(3)
        total_power         = 0.
        mdot                = 0.
        

        # Iterate over motor/rotors
        for ii in range(n_evals): 
            engine_key = list(engines.keys())[ii]
            engine     = self.engines[engine_key]                
            rotor_key  = list(propellers.keys())[rotor_indexes[ii]] 
            rot        = propellers[rotor_key]  
    
            # Throttle the engine
            engine.inputs.speed                              = state.conditions.propulsion.propulsor_group_0.rotor.rpm * Units.rpm
            conditions.propulsion.combustion_engine_throttle = conditions.propulsion.throttle
            
            # Run the engine
            engine.power(conditions)
            mdot         = mdot + engine.outputs.fuel_flow_rate * factor
            torque       = engine.outputs.torque     
            
            # link
            rot.inputs.omega = state.conditions.propulsion.propulsor_group_0.rotor.rpm * Units.rpm
            
            # step 4
            F, Q, P, Cp, outputs, etap = rot.spin(conditions)
            
            # Check to see if magic thrust is needed
            eta               = conditions.propulsion.throttle[:,0,None]
            P[eta>1.0]        = P[eta>1.0]*eta[eta>1.0]
            F[eta[:,0]>1.0,:] = F[eta[:,0]>1.0,:]*eta[eta[:,0]>1.0,:]
                
            # Pack the conditions
            R                   = rot.tip_radius
            rpm                 = engine.inputs.speed / Units.rpm
            F_mag               = np.atleast_2d(np.linalg.norm(F, axis=1))  
            total_thrust        = total_thrust + F * factor
            total_power         = total_power  + P * factor
              
            # Pack specific outputs
            conditions.propulsion['propulsor_group_' + str(ii)].engine_torque        = torque
            conditions.propulsion['propulsor_group_' + str(ii)].rotor.torque         = Q
            conditions.propulsion['propulsor_group_' + str(ii)].rotor.rpm            = rpm
            conditions.propulsion['propulsor_group_' + str(ii)].rotor.tip_mach       = (R*rpm*Units.rpm)/a
            conditions.propulsion['propulsor_group_' + str(ii)].rotor.disc_loading   = (F_mag)/(np.pi*(R**2)) # N/m^2                  
            conditions.propulsion['propulsor_group_' + str(ii)].rotor.power_loading  = (F_mag)/(P)      # N/W   
            conditions.propulsion['propulsor_group_' + str(ii)].rotor.efficiency     = etap
            conditions.propulsion['propulsor_group_' + str(ii)].rotor.figure_of_merit= outputs.figure_of_merit
            conditions.propulsion['propulsor_group_' + str(ii)].throttle             = conditions.propulsion.throttle
            conditions.noise.sources.rotors[rot.tag]  = outputs 

        # Create the outputs
        conditions.propulsion.power = total_power
        
        results = Data()
        results.thrust_force_vector       = total_thrust
        results.vehicle_mass_rate         = mdot 
        
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

        for i in range(segment.state.conditions.propulsion.number_of_propulsor_groups):        
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.rpm = segment.state.unknowns['rpm_' + str(i)]
        
        return
    
    def residuals(self,segment):
        """ Calculates a residual based on torques 
        
        Assumptions:
        
        Inputs:
            segment.state.conditions.propulsion.
                motor.torque                       [newtom-meters]                 
                propeller.torque                   [newtom-meters] 
        
        Outputs:
            segment.state:
                residuals.network                  [newtom-meters] 
                
        Properties Used:
            N/A
                                
        """          
            
        for i in range(segment.state.conditions.propulsion.number_of_propulsor_groups): 
                q_motor   = segment.state.conditions.propulsion['propulsor_group_' + str(i)].engine_torque
                q_prop    = segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.torque 
                segment.state.residuals['propulsor_group_' + str(i)] = q_motor - q_prop  
        
        return
    
    def add_unknowns_and_residuals_to_segment(self, segment,rpms=[2500]):
        """ This function sets up the information that the mission needs to run a mission segment using this network
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            segment
            rpm                      [rpm]
            
            Outputs:
            segment.state.unknowns.battery_voltage_under_load
            segment.state.unknowns.propeller_power_coefficient
            segment.state.conditions.propulsion.propulsor_group_0.motor.torque
            segment.state.conditions.propulsion.propulsor_group_0.rotor.torque   
    
            Properties Used:
            N/A
        """                

        rotor_group_indexes     = self.rotor_group_indexes
        motor_group_indexes     = self.motor_group_indexes 
        
        # unpack the ones function
        ones_row = segment.state.ones_row
        
        # Count how many unknowns and residuals based on p
        n_props   = len(self.propellers)
        n_engines = len(self.engines)
        n_eng     = self.number_of_engines
        
        if n_props!=n_engines!=n_eng:
            print('The number of propellers is not the same as the number of engines')
        
    
        # Count the number of unique pairs of rotors and motors to determine number of unique pairs of residuals and unknowns 
        unique_rotor_groups = np.unique(rotor_group_indexes)
        unique_motor_groups = np.unique(motor_group_indexes)
        if (unique_rotor_groups == unique_motor_groups).all(): # rotors and motors are paired  
            n_groups      = len(unique_rotor_groups) 
        else: 
            n_groups      = len(rotor_group_indexes)   
        
        # Setup the residuals  
        for i in range(n_groups):                  
            segment.state.residuals['propulsor_group_' + str(i)] =  0. * ones_row(1)  
            segment.state.unknowns['rpm_' + str(i)]              = rpms[i] * ones_row(1) 
        
        # Setup the conditions  
        for i in range(n_groups):         
            # Setup the conditions
            segment.state.conditions.propulsion['propulsor_group_' + str(i)]                         = MARC.Analyses.Mission.Segments.Conditions.Conditions()
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].motor                   = MARC.Analyses.Mission.Segments.Conditions.Conditions()
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor                   = MARC.Analyses.Mission.Segments.Conditions.Conditions()
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].throttle                = 0. * ones_row(1)    
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.torque            = 0. * ones_row(1)
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.thrust            = 0. * ones_row(1)
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.rpm               = 0. * ones_row(1)
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.disc_loading      = 0. * ones_row(1)                 
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.power_loading     = 0. * ones_row(1)
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.tip_mach          = 0. * ones_row(1)
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.efficiency        = 0. * ones_row(1)   
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.figure_of_merit   = 0. * ones_row(1)  
            segment.state.conditions.propulsion['propulsor_group_' + str(i)].rotor.throttle          = 0. * ones_row(1)          

        # Ensure the mission knows how to pack and unpack the unknowns and residuals
        segment.process.iterate.unknowns.network  = self.unpack_unknowns
        segment.process.iterate.residuals.network = self.residuals        

        return segment
            
    __call__ = evaluate_thrust
