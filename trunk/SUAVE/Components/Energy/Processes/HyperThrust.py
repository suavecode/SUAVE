# HyperThrust.py
#
# Created:  May 2017, P. Goncalves

# Major modifications on thrust calculations


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

import SUAVE

from SUAVE.Core import Units

# package imports
import numpy as np
import scipy as sp


from SUAVE.Core import Data
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Components.Propulsors.Propulsor import Propulsor


# ----------------------------------------------------------------------
#  Thrust Process
# ----------------------------------------------------------------------

class HyperThrust(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Thrust
        a component that computes the thrust and other output properties
        
        this class is callable, see self.__call__
        
        """
    
    def __defaults__(self):
        
        #setting the default values
        self.tag ='Thrust'
        self.bypass_ratio                             = 0.0
        self.compressor_nondimensional_massflow       = 0.0
        self.reference_temperature                    = 288.15
        self.reference_pressure                       = 1.01325*10**5
        self.number_of_engines                        = 0.0
        self.inputs.fuel_to_air_ratio                 = 0.0  #changed
        self.outputs.thrust                           = 0.0 
        self.outputs.thrust_specific_fuel_consumption = 0.0
        self.outputs.specific_impulse                 = 0.0
        self.outputs.non_dimensional_thrust           = 0.0
        self.outputs.core_mass_flow_rate              = 0.0
        self.outputs.fuel_flow_rate                   = 0.0
        self.outputs.fuel_mass                        = 0.0 #changed
        self.outputs.power                            = 0.0
        self.design_thrust                            = 0.0
        self.mass_flow_rate_design                    = 0.0

	
 
    def compute(self,conditions):
        
        #unpack the values
        
        #unpacking from conditions
        gamma                = conditions.freestream.isentropic_expansion_factor
        Cp                   = conditions.freestream.specific_heat_at_constant_pressure
        u0                   = conditions.freestream.velocity
        a0                   = conditions.freestream.speed_of_sound
        M0                   = conditions.freestream.mach_number
        p0                   = conditions.freestream.pressure  
        g                    = conditions.freestream.gravity
        throttle             = conditions.propulsion.throttle   
        
        
        #unpacking from inputs
        f                           = self.inputs.fuel_to_air_ratio
        total_temperature_reference = self.inputs.total_temperature_reference
        total_pressure_reference    = self.inputs.total_pressure_reference
        core_nozzle                 = self.inputs.core_nozzle
        core_exit_velocity          = self.inputs.core_nozzle.velocity
        core_area_ratio             = self.inputs.core_nozzle.area_ratio
        no_eng                      = self.inputs.number_of_engines                      
        
        #unpacking from self
        Tref                 = self.reference_temperature
        Pref                 = self.reference_pressure
        mdhc                 = self.compressor_nondimensional_massflow

        

        #computing the non dimensional thrust
        
        Fsp = (u0*((1+f)*(core_nozzle.velocity/u0)-1))+((1+f)/(core_nozzle.rho*core_nozzle.velocity))*p0*(core_nozzle.static_pressure/p0-1) # N / kg/s

        #Computing the specific impulse
        #Isp              = Fsp*a0*(1+bypass_ratio)/(f*g)
        
        #Computing the TSFC
        TSFC             = 1.*f/Fsp

        #computing the core mass flow
        mdot_core        = mdhc*np.sqrt(Tref/total_temperature_reference)*(total_pressure_reference/Pref)

        #computing the dimensional thrust
        FD2              = Fsp*mdot_core*no_eng*throttle

     
        
        #fuel flow rate
        a = np.array([0.])        
        fuel_flow_rate   = np.fmax(FD2*TSFC,a) #use units package for the constants
        
        #computing the power 
        power            = FD2*u0
        


        #pack outputs
        
        self.outputs.thrust                            = FD2 
        self.outputs.thrust_specific_fuel_consumption  = TSFC
        self.outputs.non_dimensional_thrust            = Fsp 
        self.outputs.core_mass_flow_rate               = mdot_core
        self.outputs.fuel_flow_rate                    = fuel_flow_rate    
        self.outputs.exit_velocity                     = core_nozzle.velocity
        self.outputs.exit_pressure                     = core_nozzle.static_pressure
        self.outputs.area_ratio                        = core_nozzle.area_ratio
        
    
    def size(self,conditions):
        
        #unpack inputs
        a0                   = conditions.freestream.speed_of_sound
        throttle             = 1.0
        
        #unpack from self
        Tref                        = self.reference_temperature
        Pref                        = self.reference_pressure
        design_thrust               = self.total_design
        
        total_temperature_reference = self.inputs.total_temperature_reference  
        total_pressure_reference    = self.inputs.total_pressure_reference
        no_eng                      = self.inputs.number_of_engines
        
        #compute nondimensional thrust
        self.compute(conditions)
        
        #unpack results 
        Fsp                         = self.outputs.non_dimensional_thrust

                
        #compute dimensional mass flow rates
        mdot_core                   = design_thrust/(Fsp*no_eng*throttle)  
        print 'mdot_core', mdot_core
        
        mdhc                        = mdot_core/ (np.sqrt(Tref/total_temperature_reference)*(total_pressure_reference/Pref))
        print 'mdhc', mdhc
        #pack outputs
        self.mass_flow_rate_design               = mdot_core
        self.compressor_nondimensional_massflow  = mdhc
 
         
        
        return
    
    
    
    
    __call__ = compute         

