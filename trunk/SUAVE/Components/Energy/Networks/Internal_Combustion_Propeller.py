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
import numpy as np
from .Network import Network
from SUAVE.Components.Physical_Component import Container
from SUAVE.Core import Data, Units

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
        self.engines              = Container()
        self.propellers           = Container()
        self.engine_length        = None
        self.number_of_engines    = None
        self.identical_propellers = True
    
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
        engines     = self.engines
        props       = self.propellers
        num_engines = self.number_of_engines
        
        # Unpack conditions
        a = conditions.freestream.speed_of_sound        
        
        # How many evaluations to do
        if self.identical_propellers:
            n_evals = 1
            factor  = num_engines*1
        else:
            n_evals = int(num_engines)
            factor  = 1.
            
        # Setup numbers for iteration
        total_thrust        = 0. * state.ones_row(3)
        total_power         = 0.
        mdot                = 0.
        for ii in range(n_evals):     
            
            # Unpack the engine and props
            engine_key = list(engines.keys())[ii]
            prop_key   = list(props.keys())[ii]
            engine     = self.engines[engine_key]
            prop       = self.propellers[prop_key]            
        
            # Throttle the engine
            engine.inputs.speed                              = state.conditions.propulsion.rpm * Units.rpm
            conditions.propulsion.combustion_engine_throttle = conditions.propulsion.throttle
            
            # Run the engine
            engine.power(conditions)
            mdot         = mdot + engine.outputs.fuel_flow_rate * factor
            torque       = engine.outputs.torque     
            
            # link
            prop.inputs.omega = state.conditions.propulsion.rpm * Units.rpm
            
            # step 4
            F, Q, P, Cp, outputs, etap = prop.spin(conditions)
            
            # Check to see if magic thrust is needed
            eta               = conditions.propulsion.throttle[:,0,None]
            P[eta>1.0]        = P[eta>1.0]*eta[eta>1.0]
            F[eta[:,0]>1.0,:] = F[eta[:,0]>1.0,:]*eta[eta[:,0]>1.0,:]
                
            # Pack the conditions
            R                   = prop.tip_radius
            rpm                 = engine.inputs.speed / Units.rpm
            F_mag               = np.atleast_2d(np.linalg.norm(F, axis=1))  
            total_thrust        = total_thrust + F * factor
            total_power         = total_power  + P * factor
              
            # Pack specific outputs
            conditions.propulsion.engine_torque[:,ii]      = torque[:,0]
            conditions.propulsion.propeller_torque[:,ii]   = Q[:,0]
            conditions.propulsion.propeller_rpm[:,ii]      = rpm[:,0]
            conditions.propulsion.propeller_tip_mach[:,ii] = (R*rpm[:,0]*Units.rpm)/a[:,0]
            conditions.propulsion.disc_loading[:,ii]       = (F_mag[:,0])/(np.pi*(R**2)) # N/m^2                  
            conditions.propulsion.power_loading[:,ii]      = (F_mag[:,0])/(P[:,0])      # N/W   
            conditions.propulsion.propeller_efficiency     = etap[:,0]
            
            conditions.noise.sources.propellers[prop.tag]  = outputs


        # Create the outputs
        conditions.propulsion.power = total_power
        
        results = Data()
        results.thrust_force_vector = total_thrust
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
                motor_torque                       [newtom-meters]                 
                propeller_torque                   [newtom-meters] 
        
        Outputs:
            segment.state:
                residuals.network                  [newtom-meters] 
                
        Properties Used:
            N/A
                                
        """         
            
        # Here we are going to pack the residuals (torque,voltage) from the network
        
        # Unpack
        q_motor   = segment.state.conditions.propulsion.engine_torque
        q_prop    = segment.state.conditions.propulsion.propeller_torque
        
        # Return the residuals
        segment.state.residuals.network = q_motor - q_prop
        
        return
    
    def add_unknowns_and_residuals_to_segment(self, segment,rpm=2500):
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
            segment.state.conditions.propulsion.propeller_motor_torque
            segment.state.conditions.propulsion.propeller_torque   
    
            Properties Used:
            N/A
        """                
        
        # unpack the ones function
        ones_row = segment.state.ones_row
        
        # Count how many unknowns and residuals based on p
        n_props   = len(self.propellers)
        n_engines = len(self.engines)
        n_eng    = self.number_of_engines
        
        if n_props!=n_engines!=n_eng:
            print('The number of propellers is not the same as the number of engines')
            
        # Now check if the propellers are all identical, in this case they have the same of residuals and unknowns
        if self.identical_propellers:
            n_props = 1
            
        # number of residuals, number of props
        n_res = n_props
        
        # Setup the residuals
        segment.state.residuals.network = 0. * ones_row(n_res)
        
        # Setup the unknowns
        segment.state.unknowns.rpm = rpm * ones_row(1) 
        
        # Setup the conditions
        segment.state.conditions.propulsion.engine_torque          = 0. * ones_row(n_props)
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
