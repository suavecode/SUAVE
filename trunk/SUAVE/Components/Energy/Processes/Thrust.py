# Thrust.py
#
# Created:  Anil, July 2014

#--put in a folder

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

import SUAVE

from SUAVE.Attributes import Units

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp


from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Components import Component_Exception
from SUAVE.Components.Propulsors.Propulsor import Propulsor


# ----------------------------------------------------------------------
#  Thrust Process
# ----------------------------------------------------------------------

class Thrust(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Thrust
        a component that computes the thrust and other output properties
        
        this class is callable, see self.__call__
        
        """
    
    def __defaults__(self):
        
        #setting the default values
        self.tag ='Thrust'
        self.bypass_ratio                             = 1.0
        self.compressor_nondimensional_massflow       = 1.0
        self.reference_temperature                    = 1.0
        self.reference_pressure                       = 1.0
        self.number_of_engines                        = 1.0
        self.inputs.fuel_to_air_ratio                 = 1.0
        self.outputs.thrust                           = 1.0
        self.outputs.thrust_specific_fuel_consumption = 1.0
        self.outputs.specific_impulse                 = 1.0
        self.outputs.non_dimensional_thrust           = 1.0
        self.outputs.core_mass_flow_rate              = 1.0
        self.outputs.fuel_flow_rate                   = 1.0
        self.outputs.fuel_mass                        = 1.0
        self.outputs.power                            = 1.0
    
 
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
        f                    = self.inputs.fuel_to_air_ratio
        stag_temp_lpt_exit   = self.inputs.stag_temp_lpt_exit
        stag_press_lpt_exit  = self.inputs.stag_press_lpt_exit
        core_nozzle          = self.inputs.core_nozzle
        fan_nozzle           = self.inputs.fan_nozzle
        fan_exit_velocity    = self.inputs.fan_nozzle.velocity
        core_exit_velocity   = self.inputs.core_nozzle.velocity
        fan_area_ratio       = self.inputs.fan_nozzle.area_ratio
        core_area_ratio      = self.inputs.core_nozzle.area_ratio
        
        #unpacking from self
        bypass_ratio         =  self.bypass_ratio
        Tref                 = self.reference_temperature
        Pref                 = self.reference_pressure
        no_eng               = self.number_of_engines
        mdhc                 =  self.compressor_nondimensional_massflow
        
        
        
        #Computing the engine output properties, the thrust, SFC, fuel flow rate--------------
        
        ##----drela method--------------
        

        ##--specific thrust
        #specific_thrust=((1+f)*core_exit_velocity-u0+self.alpha*(fan_exit_velocity-u0))/((1+self.alpha)*a0)
        
        ##Specific impulse
        #Isp=specific_thrust*a0*(1+self.alpha)/(f*g)
        
        ##thrust specific fuel consumption
        #TSFC=3600/Isp
        
        ##mass flow sizing
        #mdot_core=self.mdhc*np.sqrt(self.Tref/stag_temp_lpt_exit)*(stag_press_lpt_exit/self.Pref)
        
        ##fuel flow rate computation
        #fuel_rate=mdot_core*f*self.no_eng
        
        ##dimensional thrust
        #thrust=specific_thrust*a0*(1+self.alpha)*mdot_core*self.no_eng*throttle
        
        ##--fuel mass flow rate
        #mfuel=0.1019715*thrust*TSFC/3600
        
        ##--Output power based on freestream velocity
        #power = thrust*u0
        
        
        
        ##--------Cantwell method---------------------------------
        
        #computing the area ratios for the core and fan
        Ae_b_Ao          = 1/(1+bypass_ratio)*core_area_ratio
        A1e_b_A1o        = bypass_ratio/(1+bypass_ratio)*fan_area_ratio

        #computing the non dimensional thrust
        Thrust_nd        = gamma*M0**2*(1/(1+bypass_ratio)*(core_nozzle.velocity/u0-1)+(bypass_ratio/(1+bypass_ratio))*(fan_nozzle.velocity/u0-1))+Ae_b_Ao*(core_nozzle.static_pressure/p0-1)+A1e_b_A1o*(fan_nozzle.static_pressure/p0-1)
        Fsp              = 1/(gamma*M0)*Thrust_nd
        
        #Computing the sepcific impulse
        Isp              = Fsp*a0*(1+bypass_ratio)/(f*g)
        
        #Computing the TSFC
        TSFC             = 3600/Isp  
        
        #computing the core mass flow
        mdot_core        = mdhc*np.sqrt(Tref/stag_temp_lpt_exit)*(stag_press_lpt_exit/Pref)
        
        #computing the air mass flow rate
        mass_flow_rate   = mdot_core*f*no_eng
        
        #computing the dimensional thrust
        FD2              = Fsp*a0*(1+bypass_ratio)*mdot_core*no_eng*throttle
        
        #fuel flow rate
        fuel_flow_rate   = 0.1019715*FD2*TSFC/3600
        
        #computing the power 
        power            = FD2*u0
        
        #pack outputs
        
        self.outputs.thrust                            = FD2 
        self.outputs.thrust_specific_fuel_consumption  = TSFC
        self.outputs.specific_impulse                  = Isp
        self.outputs.non_dimensional_thrust            = Fsp 
        self.outputs.core_mass_flow_rate               = mdot_core
        self.outputs.fuel_flow_rate                    = fuel_flow_rate    
        self.outputs.power                             = power  
    
    
    __call__ = compute         

