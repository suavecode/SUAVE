#Turbofan_Network.py
# 
# Created:  Anil Variyar, Oct 2014
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
from SUAVE.Attributes import Units
from SUAVE.Structure import Data

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

from SUAVE.Structure import (
Data, Container, Data_Exception, Data_Warning,
)

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components import Component_Exception
from SUAVE.Components.Propulsors.Propulsor import Propulsor


#--------------------------------------------------------------------------------------



# the network
class Turbofan_Network(Propulsor):
    def __defaults__(self):
        
        self.tag = 'Turbo_Fan'
        #self.Nozzle       = SUAVE.Components.Energy.Gas_Turbine.Nozzle()
        #self.Compressor   = SUAVE.Components.Energy.Gas_Turbine.Compressor()
        #self.Combustor    = SUAVE.Components.Energy.Gas_Turbine.Combustor()
        #self.Turbine      = SUAVE.Components.Energy.Gas_Turbine.Turbine()
        
        
        self.nacelle_dia = 0.0
        self.number_of_engines = 1.0
    #self.thrust = Data()
    #self.tag         = 'Network'
    
    _component_root_map = None
    
    
    
    
    
    # manage process with a driver function
    def evaluate(self,conditions,numerics):
        
        # unpack to shorter component names
        # table the equal signs
        
        
        #Unpack components
        
        ram = self.ram
        inlet_nozzle = self.inlet_nozzle
        low_pressure_compressor = self.low_pressure_compressor
        high_pressure_compressor = self.high_pressure_compressor
        fan = self.fan
        combustor=self.combustor
        high_pressure_turbine=self.high_pressure_turbine
        low_pressure_turbine=self.low_pressure_turbine
        core_nozzle = self.core_nozzle
        fan_nozzle = self.fan_nozzle
        thrust = self.thrust
        
        
        
        #Network
        
        
        ram.inputs.working_fluid = self.working_fluid
        ram(conditions)
        
        
        
        
        inlet_nozzle.inputs.stagnation_temperature = ram.outputs.stagnation_temperature #conditions.freestream.stagnation_temperature
        inlet_nozzle.inputs.stagnation_pressure = ram.outputs.stagnation_pressure #conditions.freestream.stagnation_pressure
        
        
        #print 'ram out temp ', ram.outputs.stagnation_temperature
        #print 'ram out press', ram.outputs.stagnation_pressure
        
        
        inlet_nozzle(conditions)
        
        
        #print 'inlet nozzle out temp ', inlet_nozzle.outputs.stagnation_temperature
        #print 'inlet nozzle out press', inlet_nozzle.outputs.stagnation_pressure
        #print 'inlet nozzle out h', inlet_nozzle.outputs.stagnation_enthalpy
        
        #---Flow through core------------------------------------------------------
        
        #--low pressure compressor
        low_pressure_compressor.inputs.stagnation_temperature = inlet_nozzle.outputs.stagnation_temperature
        low_pressure_compressor.inputs.stagnation_pressure = inlet_nozzle.outputs.stagnation_pressure
        
        low_pressure_compressor(conditions)
        
        #print 'low_pressure_compressor out temp ', low_pressure_compressor.outputs.stagnation_temperature
        #print 'low_pressure_compressor out press', low_pressure_compressor.outputs.stagnation_pressure
        #print 'low_pressure_compressor out h', low_pressure_compressor.outputs.stagnation_enthalpy
        #--high pressure compressor
        
        high_pressure_compressor.inputs.stagnation_temperature = low_pressure_compressor.outputs.stagnation_temperature
        high_pressure_compressor.inputs.stagnation_pressure = low_pressure_compressor.outputs.stagnation_pressure
        
        high_pressure_compressor(conditions)
        
        #print 'high_pressure_compressor out temp ', high_pressure_compressor.outputs.stagnation_temperature
        #print 'high_pressure_compressor out press', high_pressure_compressor.outputs.stagnation_pressure
        #print 'high_pressure_compressor out h', high_pressure_compressor.outputs.stagnation_enthalpy
        
        
        #Fan
        
        
        fan.inputs.stagnation_temperature = inlet_nozzle.outputs.stagnation_temperature
        fan.inputs.stagnation_pressure = inlet_nozzle.outputs.stagnation_pressure
        
        fan(conditions)
        
        #print 'fan out temp ', fan.outputs.stagnation_temperature
        #print 'fan out press', fan.outputs.stagnation_pressure
        #print 'fan out h', fan.outputs.stagnation_enthalpy
        
        
        
        #--Combustor
        combustor.inputs.stagnation_temperature = high_pressure_compressor.outputs.stagnation_temperature
        combustor.inputs.stagnation_pressure = high_pressure_compressor.outputs.stagnation_pressure
        combustor.inputs.nozzle_exit_stagnation_temperature = inlet_nozzle.outputs.stagnation_temperature
        
        combustor(conditions)
        
        #print 'combustor out temp ', combustor.outputs.stagnation_temperature
        #print 'combustor out press', combustor.outputs.stagnation_pressure
        #print 'combustor out f', combustor.outputs.fuel_to_air_ratio
        #print 'combustor out h', combustor.outputs.stagnation_enthalpy
        
        #high pressure turbine
        
        high_pressure_turbine.inputs.stagnation_temperature = combustor.outputs.stagnation_temperature
        high_pressure_turbine.inputs.stagnation_pressure = combustor.outputs.stagnation_pressure
        high_pressure_turbine.inputs.compressor = high_pressure_compressor.outputs
        high_pressure_turbine.inputs.fuel_to_air_ratio = combustor.outputs.fuel_to_air_ratio
        high_pressure_turbine.inputs.fan =  fan.outputs
        high_pressure_turbine.inputs.bypass_ratio =0.0
        
        high_pressure_turbine(conditions)
        
        #print 'high_pressure_turbine out temp ', high_pressure_turbine.outputs.stagnation_temperature
        #print 'high_pressure_turbine out press', high_pressure_turbine.outputs.stagnation_pressure
        #print 'high_pressure_turbine out h', high_pressure_turbine.outputs.stagnation_enthalpy
        
        #low pressure turbine
        
        low_pressure_turbine.inputs.stagnation_temperature = high_pressure_turbine.outputs.stagnation_temperature
        low_pressure_turbine.inputs.stagnation_pressure = high_pressure_turbine.outputs.stagnation_pressure
        low_pressure_turbine.inputs.compressor = low_pressure_compressor.outputs
        low_pressure_turbine.inputs.fuel_to_air_ratio = combustor.outputs.fuel_to_air_ratio
        low_pressure_turbine.inputs.fan =  fan.outputs
        low_pressure_turbine.inputs.bypass_ratio =  thrust.bypass_ratio
        
        low_pressure_turbine(conditions)
        
        #print 'low_pressure_turbine out temp ', low_pressure_turbine.outputs.stagnation_temperature
        #print 'low_pressure_turbine out press', low_pressure_turbine.outputs.stagnation_pressure
        #print 'low_pressure_turbine out h', low_pressure_turbine.outputs.stagnation_enthalpy
        
        #core nozzle
        
        core_nozzle.inputs.stagnation_temperature = low_pressure_turbine.outputs.stagnation_temperature
        core_nozzle.inputs.stagnation_pressure = low_pressure_turbine.outputs.stagnation_pressure
        
        core_nozzle(conditions)
        
        #print 'core_nozzle out temp ', core_nozzle.outputs.stagnation_temperature
        #print 'core_nozzle out press', core_nozzle.outputs.stagnation_pressure
        #print 'core_nozzle out h', core_nozzle.outputs.stagnation_enthalpy
        
        
        
        
        
        #fan nozzle
        
        fan_nozzle.inputs.stagnation_temperature = fan.outputs.stagnation_temperature
        fan_nozzle.inputs.stagnation_pressure = fan.outputs.stagnation_pressure
        
        fan_nozzle(conditions)
        
        #print 'fan_nozzle out temp ', fan_nozzle.outputs.stagnation_temperature
        #print 'fan_nozzle out press', fan_nozzle.outputs.stagnation_pressure
        #print 'fan_nozzle out h', fan_nozzle.outputs.stagnation_enthalpy
        
        #compute thrust
        
        thrust.inputs.fan_exit_velocity = fan_nozzle.outputs.velocity
        thrust.inputs.core_exit_velocity = core_nozzle.outputs.velocity
        thrust.inputs.fuel_to_air_ratio  = combustor.outputs.fuel_to_air_ratio
        thrust.inputs.stag_temp_lpt_exit  = low_pressure_compressor.outputs.stagnation_temperature
        thrust.inputs.stag_press_lpt_exit = low_pressure_compressor.outputs.stagnation_pressure
        thrust.inputs.fan_area_ratio = fan_nozzle.outputs.area_ratio
        thrust.inputs.core_area_ratio = core_nozzle.outputs.area_ratio
        thrust.inputs.fan_nozzle = fan_nozzle.outputs
        thrust.inputs.core_nozzle = core_nozzle.outputs
        thrust(conditions)
        
        
        #getting the output data from the thrust outputs
        
        F = thrust.outputs.thrust
        mdot = thrust.outputs.fuel_mass
        Isp = thrust.outputs.specific_impulse
        P = thrust.outputs.power
        
        
        return F,mdot,P
    #return F[:,0],mdot[:,0],P[:,0]  #return the 2d array instead of the 1D array
    
    
    __call__ = evaluate

