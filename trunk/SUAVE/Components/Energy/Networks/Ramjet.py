# Ramjet.py
# 
# Created:  May 2017, P. Goncalves
# Modified:  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import scipy as sp
import datetime
import time
from SUAVE.Core import Units

from SUAVE.Core import Data
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components.Propulsors.Propulsor import Propulsor


# ----------------------------------------------------------------------
#  Turbojet Network
# ----------------------------------------------------------------------

class Ramjet(Propulsor):
    
    def __defaults__(self):
        
        #setting the default values
        self.tag = 'Ramjet'
        self.number_of_engines = 1.0
        self.nacelle_diameter  = 1.0
        self.engine_length     = 1.0
    
    _component_root_map = None
        
    
    # linking the different network components
    def evaluate_thrust(self,state):

    
        #Unpack
        
        conditions = state.conditions
        numerics   = state.numerics
        
        ram                       = self.ram
        inlet_nozzle              = self.inlet_nozzle
        combustor                 = self.combustor
        core_nozzle               = self.core_nozzle
        thrust                    = self.thrust
        
        number_of_engines         = self.number_of_engines        
        
        #Creating the network by manually linking the different components
        
        
        #set the working fluid to determine the fluid properties
        ram.inputs.working_fluid                             = self.working_fluid
        
        #Flow through the ram , this computes the necessary flow quantities and stores it into conditions
        ram(conditions)
        
        
        #################################
        
        #link inlet nozzle to ram 
        inlet_nozzle.inputs.stagnation_temperature             = ram.outputs.stagnation_temperature #conditions.freestream.stagnation_temperature
        inlet_nozzle.inputs.stagnation_pressure                = ram.outputs.stagnation_pressure #conditions.freestream.stagnation_pressure
        
        #Flow through the inlet nozzle
        inlet_nozzle(conditions)

        
        #################################
        
        #link the combustor to the high pressure compressor
        combustor.inputs.stagnation_temperature                = inlet_nozzle.outputs.stagnation_temperature
        combustor.inputs.stagnation_pressure                   = inlet_nozzle.outputs.stagnation_pressure
        combustor.inputs.mach_number                           = inlet_nozzle.outputs.mach_number

        #combustor.inputs.nozzle_exit_stagnation_temperature = inlet_nozzle.outputs.stagnation_temperature
        
        #flow through the high pressor comprresor
        combustor(conditions)
        
        
        #################################
            
        #link the core nozzle to the low pressure turbine
        core_nozzle.inputs.stagnation_temperature              = combustor.outputs.stagnation_temperature
        core_nozzle.inputs.stagnation_pressure                 = combustor.outputs.stagnation_pressure
        
        #flow through the core nozzle
        core_nozzle(conditions)
        
        
        #link the thrust component to the core nozzle
        thrust.inputs.core_exit_velocity                       = core_nozzle.outputs.velocity
        thrust.inputs.core_area_ratio                          = core_nozzle.outputs.area_ratio
        thrust.inputs.core_nozzle                              = core_nozzle.outputs
        #link the thrust component to the combustor
        thrust.inputs.fuel_to_air_ratio                        = combustor.outputs.fuel_to_air_ratio
        #link the thrust component to the low pressure compressor 
        thrust.inputs.stag_temp_lpt_exit                       = inlet_nozzle.outputs.stagnation_temperature
        thrust.inputs.stag_press_lpt_exit                      = inlet_nozzle.outputs.stagnation_pressure
        thrust.inputs.flow_through_core                        =  1.0 #scaled constant to turn on core thrust computation
        thrust.inputs.flow_through_fan                         =  0.0 #scaled constant to turn on core thrust computation

        

        #compute the trust
        thrust(conditions)
 
        
        
        
        #getting the network outputs from the thrust outputs
        
        F            = thrust.outputs.thrust*[1,0,0]
        mdot         = thrust.outputs.fuel_flow_rate
        Isp          = thrust.outputs.specific_impulse
        output_power = thrust.outputs.power
        F_vec        = conditions.ones_row(3) * 0.0
        F_vec[:,0]   = F[:,0]
        F            = F_vec
        velocity     = thrust.outputs.exit_velocity
        pressure     = thrust.outputs.exit_pressure
        area_ratio   = thrust.outputs.area_ratio

        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot
        results.power               = output_power
        results.exit_velocity       = velocity
        results.exit_pressure       = pressure
        results.area_ratio          = area_ratio
        
        return results
    
    
    
    def size(self,state):  
        
        #Unpack components
        
        conditions = state.conditions
        numerics   = state.numerics        
        
        ram                       = self.ram
        inlet_nozzle              = self.inlet_nozzle
        combustor                 = self.combustor
        core_nozzle               = self.core_nozzle
        thrust                    = self.hyperthrust
        
        
        
        #Creating the network by manually linking the different components
        
        
        #set the working fluid to determine the fluid properties
        ram.inputs.working_fluid                             = self.working_fluid
        
        #Flow through the ram , this computes the necessary flow quantities and stores it into conditions
        ram(conditions)

        
        
        #link inlet nozzle to ram 
        inlet_nozzle.inputs.stagnation_temperature             = ram.outputs.stagnation_temperature #conditions.freestream.stagnation_temperature
        inlet_nozzle.inputs.stagnation_pressure                = ram.outputs.stagnation_pressure #conditions.freestream.stagnation_pressure
        
        #Flow through the inlet nozzle
        inlet_nozzle.size(conditions)
          
        
        
        #link the combustor to the high pressure compressor
        combustor.inputs.stagnation_temperature                = inlet_nozzle.outputs.stagnation_temperature
        combustor.inputs.stagnation_pressure                   = inlet_nozzle.outputs.stagnation_pressure
        #combustor.inputs.nozzle_exit_stagnation_temperature = inlet_nozzle.outputs.stagnation_temperature
        
        #flow through the high pressor comprresor
        combustor(conditions)
        
    
        
        
        
        #link the core nozzle to the low pressure turbine
        core_nozzle.inputs.stagnation_temperature              = combustor.outputs.stagnation_temperature
        core_nozzle.inputs.stagnation_pressure                 = combustor.outputs.stagnation_pressure
        
        #flow through the core nozzle
        core_nozzle(conditions)
        

        #link the thrust component to the core nozzle
        thrust.inputs.core_exit_velocity                       = core_nozzle.outputs.velocity
        thrust.inputs.core_area_ratio                          = core_nozzle.outputs.area_ratio
        thrust.inputs.core_nozzle                              = core_nozzle.outputs
        #link the thrust component to the combustor
        thrust.inputs.fuel_to_air_ratio                        = combustor.outputs.fuel_to_air_ratio
        #link the thrust component to the low pressure compressor 
        thrust.inputs.stag_temp_lpt_exit                       = inlet_nozzle.outputs.stagnation_temperature
        thrust.inputs.stag_press_lpt_exit                      = inlet_nozzle.outputs.stagnation_pressure
        #compute the trust
        thrust.size(conditions)
        
        
        
        
        
        #return
    
    
    


    __call__ = evaluate_thrust

