## @ingroup Components-Energy-Networks
#Ducted_Fan.py
# 
# Created:  Feb 2016, M. Vegh
# Modified: Aug 2017, E. Botero
#           Apr 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np

from SUAVE.Core import Data
from SUAVE.Analyses.Mission.Segments.Conditions import Conditions
from .Network import Network

# ----------------------------------------------------------------------
#  Ducted_Fan Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Ducted_Fan(Network):
    """ A ducted fan 
    
        Assumptions:
        None
        
        Source:
        Most of the components come from this book:
        https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/
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
        
        #setting the default values
        self.tag = 'Ducted_Fan'
        self.number_of_engines = 1.0
        self.nacelle_diameter  = 1.0
        self.engine_length     = 1.0
        self.bypass_ratio      = 0.0
        self.areas             = Data()
        
    _component_root_map = None

    def evaluate_thrust(self,state):
        """ Calculate thrust given the current state of the vehicle
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            state [state()]
    
            Outputs:
            results.thrust_force_vector [newtons]
            results.vehicle_mass_rate   [kg/s]
            results.power               [Watts]
            conditions.noise.sources.ducted_fan:
                fan:
                    exit_static_temperature      
                    exit_static_pressure       
                    exit_stagnation_temperature 
                    exit_stagnation_pressure
                    exit_velocity 
    
            Properties Used:
            Defaulted values
        """          

        #Unpack
        conditions                = state.conditions
        ram                       = self.ram
        inlet_nozzle              = self.inlet_nozzle
        fan                       = self.fan
        fan_nozzle                = self.fan_nozzle
        thrust                    = self.thrust
        
        #Creating the network by manually linking the different components
        #set the working fluid to determine the fluid properties
        ram.inputs.working_fluid                               = self.working_fluid
        
        #Flow through the ram , this computes the necessary flow quantities and stores it into conditions
        ram(conditions)
        
        #link inlet nozzle to ram 
        inlet_nozzle.inputs.stagnation_temperature             = ram.outputs.stagnation_temperature 
        inlet_nozzle.inputs.stagnation_pressure                = ram.outputs.stagnation_pressure 
        
        #Flow through the inlet nozzle
        inlet_nozzle(conditions)
        
        #Link the fan to the inlet nozzle
        fan.inputs.stagnation_temperature                      = inlet_nozzle.outputs.stagnation_temperature
        fan.inputs.stagnation_pressure                         = inlet_nozzle.outputs.stagnation_pressure
        
        #flow through the fan
        fan(conditions)

        #link the fan nozzle to the fan
        fan_nozzle.inputs.stagnation_temperature               = fan.outputs.stagnation_temperature
        fan_nozzle.inputs.stagnation_pressure                  = fan.outputs.stagnation_pressure
        thrust.inputs.fuel_to_air_ratio                        = 0.
        
         # flow through the fan nozzle
        fan_nozzle(conditions)
        
        # compute the thrust using the thrust component
        #link the thrust component to the fan nozzle
        thrust.inputs.fan_exit_velocity                        = fan_nozzle.outputs.velocity
        thrust.inputs.fan_area_ratio                           = fan_nozzle.outputs.area_ratio
        thrust.inputs.fan_nozzle                               = fan_nozzle.outputs
        thrust.inputs.bypass_ratio                             = self.bypass_ratio  #0.0
        
        #compute the thrust
        thrust(conditions)

        #obtain the network outputs from the thrust outputs
        
        u0                    = conditions.freestream.velocity
        u8                    = fan_nozzle.outputs.velocity
       
        propulsive_efficiency =2./(1+u8/u0)
        F               = thrust.outputs.thrust*[1,0,0]
        mdot            = thrust.outputs.fuel_flow_rate
        Isp             = thrust.outputs.specific_impulse
        output_power    = thrust.outputs.power
        F_vec           = conditions.ones_row(3) * 0.0
        F_vec[:,0]      = F[:,0]
        F               = F_vec
      
        #pack outputs
        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot
        results.power               = np.divide(output_power[:,0],propulsive_efficiency[:,0])
        
        # store data
        results_conditions = Data
        
        fan_outputs = results_conditions(
            exit_static_temperature             = fan_nozzle.outputs.static_temperature,
            exit_static_pressure                = fan_nozzle.outputs.static_pressure,
            exit_stagnation_temperature         = fan_nozzle.outputs.stagnation_temperature,
            exit_stagnation_pressure            = fan_nozzle.outputs.static_pressure,
            exit_velocity                       = fan_nozzle.outputs.velocity
            )
        
        conditions.noise.sources.ducted_fan     = Conditions()
        conditions.noise.sources.ducted_fan.fan = fan_outputs

        return results
    
    
    
    def size(self,state):
        """ Size the turbofan
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            State [state()]
    
            Outputs:
            None
    
            Properties Used:
            N/A
        """          
        
        #Unpack components
        conditions = state.conditions       
        ram                       = self.ram
        inlet_nozzle              = self.inlet_nozzle
        fan                       = self.fan
        fan_nozzle                = self.fan_nozzle
        thrust                    = self.thrust
        
        #Creating the network by manually linking the different components
        #set the working fluid to determine the fluid properties
        ram.inputs.working_fluid                             = self.working_fluid
        
        #Flow through the ram , this computes the necessary flow quantities and stores it into conditions
        ram(conditions)

        #link inlet nozzle to ram 
        inlet_nozzle.inputs.stagnation_temperature             = ram.outputs.stagnation_temperature 
        inlet_nozzle.inputs.stagnation_pressure                = ram.outputs.stagnation_pressure 
        
        #Flow through the inlet nozzle
        inlet_nozzle(conditions)
        
        #Link the fan to the inlet nozzle
        fan.inputs.stagnation_temperature                      = inlet_nozzle.outputs.stagnation_temperature
        fan.inputs.stagnation_pressure                         = inlet_nozzle.outputs.stagnation_pressure
        
        #flow through the fan
        fan(conditions)

        #link the dan nozzle to the fan
        fan_nozzle.inputs.stagnation_temperature               = fan.outputs.stagnation_temperature
        fan_nozzle.inputs.stagnation_pressure                  = fan.outputs.stagnation_pressure
        
         # flow through the fan nozzle
        fan_nozzle(conditions)
        
        # compute the thrust using the thrust component
        #link the thrust component to the fan nozzle
        thrust.inputs.fan_exit_velocity                        = fan_nozzle.outputs.velocity
        thrust.inputs.fan_area_ratio                           = fan_nozzle.outputs.area_ratio
        thrust.inputs.fan_nozzle                               = fan_nozzle.outputs
        
        #compute the thrust
        thrust.size(conditions)

    __call__ = evaluate_thrust