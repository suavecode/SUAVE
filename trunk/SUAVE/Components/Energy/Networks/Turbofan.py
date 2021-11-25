## @ingroup Components-Energy-Networks
#Turbofan.py
# 
# Created:  Oct 2014, A. Variyar, 
# Modified: Feb 2016, M. Vegh
#           Jul 2017, M. Clarke
#           Aug 2017, E. Botero
#           Oct 2017, E. Botero
#           Nov 2018, T. MacDonald
#           Apr 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np

from SUAVE.Core import Data
from .Network import Network
from SUAVE.Analyses.Mission.Segments.Conditions import Conditions
# ----------------------------------------------------------------------
#  Turbofan Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Turbofan(Network):
    """ This is a turbofan. 
    
        Assumptions:
        None
        
        Source:
        Most of the componentes come from this book:
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
        self.tag = 'Turbofan'
        self.number_of_engines    = 0.0 
        self.engine_length        = 0.0
        self.bypass_ratio         = 0.0
        self.SFC_adjustment       = 0.0 # Less than 1 is a reduction
        self.OpenVSP_flow_through = False
        
        #areas needed for drag; not in there yet
        self.areas             = Data()
        self.areas.wetted      = 0.0
        self.areas.maximum     = 0.0
        self.areas.exit        = 0.0
        self.areas.inflow      = 0.0
        
        self.generative_design_minimum         = 0
        self.generative_design_max_per_vehicle = 1
        self.generative_design_characteristics = ['sealevel_static_thrust','number_of_engines','bypass_ratio','non_dimensional_origin[0][0]','non_dimensional_origin[0][1]','non_dimensional_origin[0][2]']
        self.generative_design_char_min_bounds = [1000.,2.,0.1,0.,-0.7,-0.7]   
        self.generative_design_char_max_bounds = [np.inf,2,np.inf,0.7,0.7,0.7]    
        
    # linking the different network components
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
            conditions.noise.sources.turbofan:
                core:
                    exit_static_temperature      
                    exit_static_pressure       
                    exit_stagnation_temperature 
                    exit_stagnation_pressure
                    exit_velocity 
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
        conditions = state.conditions
        
        ram                       = self.ram
        inlet_nozzle              = self.inlet_nozzle
        low_pressure_compressor   = self.low_pressure_compressor
        high_pressure_compressor  = self.high_pressure_compressor
        fan                       = self.fan
        combustor                 = self.combustor
        high_pressure_turbine     = self.high_pressure_turbine
        low_pressure_turbine      = self.low_pressure_turbine
        core_nozzle               = self.core_nozzle
        fan_nozzle                = self.fan_nozzle
        thrust                    = self.thrust
        bypass_ratio              = self.bypass_ratio
        number_of_engines         = self.number_of_engines
        
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

        #--link low pressure compressor to the inlet nozzle
        low_pressure_compressor.inputs.stagnation_temperature  = inlet_nozzle.outputs.stagnation_temperature
        low_pressure_compressor.inputs.stagnation_pressure     = inlet_nozzle.outputs.stagnation_pressure
        
        #Flow through the low pressure compressor
        low_pressure_compressor(conditions)

        #link the high pressure compressor to the low pressure compressor
        high_pressure_compressor.inputs.stagnation_temperature = low_pressure_compressor.outputs.stagnation_temperature
        high_pressure_compressor.inputs.stagnation_pressure    = low_pressure_compressor.outputs.stagnation_pressure
        
        #Flow through the high pressure compressor
        high_pressure_compressor(conditions)
        
        #Link the fan to the inlet nozzle
        fan.inputs.stagnation_temperature                      = inlet_nozzle.outputs.stagnation_temperature
        fan.inputs.stagnation_pressure                         = inlet_nozzle.outputs.stagnation_pressure
        
        #flow through the fan
        fan(conditions)
        
        #link the combustor to the high pressure compressor
        combustor.inputs.stagnation_temperature                = high_pressure_compressor.outputs.stagnation_temperature
        combustor.inputs.stagnation_pressure                   = high_pressure_compressor.outputs.stagnation_pressure
        
        #flow through the high pressor comprresor
        combustor(conditions)

        # link the shaft power output to the low pressure compressor
        try:
            shaft_power = self.Shaft_Power_Off_Take       
            shaft_power.inputs.mdhc                            = thrust.compressor_nondimensional_massflow
            shaft_power.inputs.Tref                            = thrust.reference_temperature
            shaft_power.inputs.Pref                            = thrust.reference_pressure
            shaft_power.inputs.total_temperature_reference     = low_pressure_compressor.outputs.stagnation_temperature
            shaft_power.inputs.total_pressure_reference        = low_pressure_compressor.outputs.stagnation_pressure
    
            shaft_power(conditions)
        except:
            pass

        #link the high pressure turbine to the combustor
        high_pressure_turbine.inputs.stagnation_temperature    = combustor.outputs.stagnation_temperature
        high_pressure_turbine.inputs.stagnation_pressure       = combustor.outputs.stagnation_pressure
        high_pressure_turbine.inputs.fuel_to_air_ratio         = combustor.outputs.fuel_to_air_ratio
        
        #link the high pressuer turbine to the high pressure compressor
        high_pressure_turbine.inputs.compressor                = high_pressure_compressor.outputs
        
        #link the high pressure turbine to the fan
        high_pressure_turbine.inputs.fan                       = fan.outputs
        high_pressure_turbine.inputs.bypass_ratio              = 0.0 #set to zero to ensure that fan not linked here
        
        #flow through the high pressure turbine
        high_pressure_turbine(conditions)
                
        #link the low pressure turbine to the high pressure turbine
        low_pressure_turbine.inputs.stagnation_temperature     = high_pressure_turbine.outputs.stagnation_temperature
        low_pressure_turbine.inputs.stagnation_pressure        = high_pressure_turbine.outputs.stagnation_pressure
        
        #link the low pressure turbine to the low_pressure_compresor
        low_pressure_turbine.inputs.compressor                 = low_pressure_compressor.outputs
        
        #link the low pressure turbine to the combustor
        low_pressure_turbine.inputs.fuel_to_air_ratio          = combustor.outputs.fuel_to_air_ratio
        
        #link the low pressure turbine to the fan
        low_pressure_turbine.inputs.fan                        = fan.outputs
        
        # link the low pressure turbine to the shaft power, if needed
        try:
            low_pressure_turbine.inputs.shaft_power_off_take   = shaft_power.outputs
        except:
            pass
        
        #get the bypass ratio from the thrust component
        low_pressure_turbine.inputs.bypass_ratio               = bypass_ratio
        
        #flow through the low pressure turbine
        low_pressure_turbine(conditions)
        
        #link the core nozzle to the low pressure turbine
        core_nozzle.inputs.stagnation_temperature              = low_pressure_turbine.outputs.stagnation_temperature
        core_nozzle.inputs.stagnation_pressure                 = low_pressure_turbine.outputs.stagnation_pressure
        
        #flow through the core nozzle
        core_nozzle(conditions)

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
        
        #link the thrust component to the core nozzle
        thrust.inputs.core_exit_velocity                       = core_nozzle.outputs.velocity
        thrust.inputs.core_area_ratio                          = core_nozzle.outputs.area_ratio
        thrust.inputs.core_nozzle                              = core_nozzle.outputs
        
        #link the thrust component to the combustor
        thrust.inputs.fuel_to_air_ratio                        = combustor.outputs.fuel_to_air_ratio
        
        #link the thrust component to the low pressure compressor 
        thrust.inputs.total_temperature_reference              = low_pressure_compressor.outputs.stagnation_temperature
        thrust.inputs.total_pressure_reference                 = low_pressure_compressor.outputs.stagnation_pressure
        thrust.inputs.number_of_engines                        = number_of_engines
        thrust.inputs.bypass_ratio                             = bypass_ratio
        thrust.inputs.flow_through_core                        = 1./(1.+bypass_ratio) #scaled constant to turn on core thrust computation
        thrust.inputs.flow_through_fan                         = bypass_ratio/(1.+bypass_ratio) #scaled constant to turn on fan thrust computation        

        #compute the thrust
        thrust(conditions)

        #getting the network outputs from the thrust outputs
        F            = thrust.outputs.thrust*[1,0,0]
        mdot         = thrust.outputs.fuel_flow_rate
        output_power = thrust.outputs.power
        F_vec        = conditions.ones_row(3) * 0.0
        F_vec[:,0]   = F[:,0]
        F            = F_vec

        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot
        
        # store data
        core_outputs = Data(
            exit_static_temperature             = core_nozzle.outputs.static_temperature,
            exit_static_pressure                = core_nozzle.outputs.static_pressure,
            exit_stagnation_temperature         = core_nozzle.outputs.stagnation_temperature,
            exit_stagnation_pressure            = core_nozzle.outputs.static_pressure,
            exit_velocity                       = core_nozzle.outputs.velocity
            )
        
        fan_outputs = Data(
            exit_static_temperature             = fan_nozzle.outputs.static_temperature,
            exit_static_pressure                = fan_nozzle.outputs.static_pressure,
            exit_stagnation_temperature         = fan_nozzle.outputs.stagnation_temperature,
            exit_stagnation_pressure            = fan_nozzle.outputs.static_pressure,
            exit_velocity                       = fan_nozzle.outputs.velocity
            )
        
        conditions.noise.sources.turbofan       = Conditions()        
        conditions.noise.sources.turbofan.fan   = fan_outputs
        conditions.noise.sources.turbofan.core  = core_outputs

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
        thrust     = self.thrust
        thrust.size(conditions)
        
    def engine_out(self,state):
        """ Lose an engine
    
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
        
        temp_throttle = np.zeros(len(state.conditions.propulsion.throttle))
        
        for i in range(0,len(state.conditions.propulsion.throttle)):
            temp_throttle[i] = state.conditions.propulsion.throttle[i]
            state.conditions.propulsion.throttle[i] = 1.0
        
        results = self.evaluate_thrust(state)
        
        for i in range(0,len(state.conditions.propulsion.throttle)):
            state.conditions.propulsion.throttle[i] = temp_throttle[i]
        
        results.thrust_force_vector = results.thrust_force_vector/self.number_of_engines*(self.number_of_engines-1)
        results.vehicle_mass_rate   = results.vehicle_mass_rate/self.number_of_engines*(self.number_of_engines-1)

        return results
        
    __call__ = evaluate_thrust