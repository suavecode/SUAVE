# Gas_Turbine.py
# 
# Created:  Anil, July 2014
      


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

import SUAVE

from SUAVE.Structure import Data
from SUAVE.Attributes import Units

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp

from SUAVE.Structure import (
Data, Container, Data_Exception, Data_Warning,
)

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components import Component_Exception


# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------
"""
Build a network with manual linking of components
Gas Turbine, with Nozzle Compressor, Fan, Combustor, Turbine, Nozzle, (Afterburner)
    in a test script, with manually set conditions
Use conditions data structure, as would be expected by mission segments
Sketch out the array of different propulsors (ie turboprop, hyrbric electric, 
    ducted fan, turbofan, turbojet, scramjet)
Maybe start thinking about general network
"""


# ----------------------------------------------------------------------
#  Energy Component Class
# ----------------------------------------------------------------------
from SUAVE.Components import Physical_Component



#-----------Energy component----------------------------------------------------------

class Energy_Component(Physical_Component):
    def __defaults__(self):
        
        # function handles for input
        self.inputs  = Data()
        
        # function handles for output
        self.outputs = Data()
        
        return

#--------------------------------------------------------------------------------------

class Ram(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Ram
        a Ram class that is used to convert static properties into
        stagnation properties
        
        this class is callable, see self.__call__
        
    """    
    
    def __defaults__(self):
        

        self.tag = 'Ram'

        self.etapold = 1.0
        self.pid = 1.0
        
        
        self.outputs.Tt =1.0
        self.outputs.Pt =1.0

        

        
    def compute(self,conditions):
        
        #unpack the variables

        Po = conditions.freestream.pressure
        To = conditions.freestream.temperature
    

    
        
        #method
        conditions.freestream.gamma =1.4
        conditions.freestream.Cp =1004.

        conditions.freestream.stagnation_temperature = conditions.freestream.temperature*(1+((conditions.freestream.gamma-1)/2 *conditions.freestream.mach_number**2))
        
        conditions.freestream.stagnation_pressure = conditions.freestream.pressure* ((1+(conditions.freestream.gamma-1)/2 *conditions.freestream.mach_number**2 )**3.5 )  
             
         
        
        #pack outputs
        self.outputs.Tt =conditions.freestream.stagnation_temperature
        self.outputs.Pt =conditions.freestream.stagnation_pressure
        
        

        

        
    __call__ = compute


#--------------------------------------------------------------------------------------

class Nozzle(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Nozzle
        a nozzle component 
        
        this class is callable, see self.__call__
        
    """    
    
    def __defaults__(self):
        

        self.tag = 'Nozzle'

        self.etapold = 1.0
        self.pid = 1.0
        self.inputs
        self.outputs
        self.inputs.Tt=0.
        self.inputs.Pt=0.
        
        self.outputs.Tt=0.
        self.outputs.Pt=0.
        self.outputs.ht=0.
        

        
    def compute(self,conditions):
        
        #unpack the variables
        gamma=conditions.freestream.gamma
        Cp = conditions.freestream.Cp
        Po = conditions.freestream.pressure
        
        Tt_in = self.inputs.Tt
        Pt_in = self.inputs.Pt

    
        
        #method
        Pt_out=Pt_in*self.pid
        Tt_out=Tt_in*self.pid**((gamma-1)/(gamma*self.etapold))
        ht_out=Cp*Tt_out 
        
        

        Mach=np.sqrt((((Pt_out/Po)**((gamma-1)/gamma))-1)*2/(gamma-1))
        T_out=Tt_out/(1+(gamma-1)/2*Mach**2)
        h_out=Cp*T_out
        u_out=np.sqrt(2*(ht_out-h_out))        
         
        
        #pack outputs
        self.outputs.Tt=Tt_out
        self.outputs.Pt=Pt_out
        self.outputs.ht=ht_out
        self.outputs.M = Mach
        self.outputs.T = T_out
        self.outputs.h = h_out
        self.outputs.u = u_out
        
        

        

        
    __call__ = compute
        
        
#--------------------------------------------------------------------------------------
    
class Compressor(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Compressor
        a compressor component
        
        this class is callable, see self.__call__
        
    """    
    
    def __defaults__(self):
        

        self.tag = 'Compressor'

        self.etapold = 1.0
        self.pid = 1.0
        
        self.inputs.Tt=0.
        self.inputs.Pt=0.
        
        self.outputs.Tt=0.
        self.outputs.Pt=0.
        self.outputs.ht=0.        
 
        
        

    def compute(self,conditions):
        
        #unpack the variables
        gamma=conditions.freestream.gamma
        Cp = conditions.freestream.Cp  
        Tt_in = self.inputs.Tt
        Pt_in = self.inputs.Pt        
    
        #method
        Pt_out=Pt_in*self.pid
        Tt_out=Tt_in*self.pid**((gamma-1)/(gamma*self.etapold))
        ht_out=Cp*Tt_out 
        
        #pack outputs
        self.outputs.Tt=Tt_out
        self.outputs.Pt=Pt_out
        self.outputs.ht=ht_out        
        

    __call__ = compute
    
    
#--------------------------------------------------------------------------------------
    
class Fan(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Fan
        a Fan component
        
        this class is callable, see self.__call__
        
    """    
    
    def __defaults__(self):
        

        self.tag ='Fan'

        self.etapold = 1.0
        self.pid = 1.0
        
        self.inputs.Tt=0.
        self.inputs.Pt=0.
        
        self.outputs.Tt=0.
        self.outputs.Pt=0.
        self.outputs.ht=0.        
      
        
        
    def compute(self,conditions):
        
        gamma=conditions.freestream.gamma
        Cp = conditions.freestream.Cp  
        Tt_in = self.inputs.Tt
        Pt_in = self.inputs.Pt        
    
    
        #method
        Pt_out=Pt_in*self.pid
        Tt_out=Tt_in*self.pid**((gamma-1)/(gamma*self.etapold))
        ht_out=Cp*Tt_out    #h(Tt1_8)
        
        
        #pack outputs
        self.outputs.Tt=Tt_out
        self.outputs.Pt=Pt_out
        self.outputs.ht=ht_out   
        

        
        
    __call__ = compute  
        

#--------------------------------------------------------------------------------------


    
class Combustor(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Combustor
        a combustor component
        
        this class is callable, see self.__call__
        
    """    
    
    def __defaults__(self):
        

        self.tag = 'Combustor'
        self.eta = 1.0
        self.tau = 1.0
        self.To = 273.0
        self.alphac = 0.0 
        self.Tt4 = 1.0
        self.fuel = SUAVE.Attributes.Propellants.Jet_A()
        
        self.inputs.Tt = 1.0
        self.inputs.Pt = 1.0  
        self.inputs.nozzle_temp = 1.0     
        
        self.outputs.Tt=1.0
        self.outputs.Pt=1.0
        self.outputs.ht=1.0
        self.outputs.f = 1.0        
        
        

        
        
    def compute(self,conditions):
        
        #unpack the variables
        gamma=conditions.freestream.gamma
        Cp = conditions.freestream.Cp
        To = conditions.freestream.temperature
        
        Tt_in = self.inputs.Tt 
        Pt_in = self.inputs.Pt   
        Tt_n = self.inputs.nozzle_temp   
        
        htf=self.fuel.specific_energy
        
        
        #method        
        tau = htf/(Cp*To)
    
        f=(((self.Tt4/To)-tau*(Tt_in/Tt_n))/(self.eta*tau-(self.Tt4/To)))
        
        
        ht_out=Cp*self.Tt4
        Pt_out=Pt_in*self.pib  
        
        #pack outputs
        self.outputs.Tt=self.Tt4
        self.outputs.Pt=Pt_out
        self.outputs.ht=ht_out 
        self.outputs.f = f
        

        
    __call__ = compute
   
        
#--------------------------------------------------------------------------------------


class Turbine(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Turbine
        a Turbine component
        
        this class is callable, see self.__call__
        
    """    
    
    def __defaults__(self):
        

        self.tag ='Turbine'
        self.eta_mech =1.0
        self.Cp =1004
        self.gamma =1.4
        self.etapolt = 1.0
        
        self.inputs.Tt = 1.0
        self.inputs.Pt = 1.0  
        self.inputs.h_compressor_out = 1.0
        self.inputs.h_compressor_in = 1.0
        self.inputs.f = 1.0
        

        self.outputs.Tt=1.0
        self.outputs.Pt=1.0
        self.outputs.ht=1.0    
        
        
     
        
    def compute(self,conditions):
        
        #unpack inputs
        gamma=conditions.freestream.gamma
        Cp = conditions.freestream.Cp 
        
        Tt_in =self.inputs.Tt 
        Pt_in =self.inputs.Pt   
        h_compressor_out =self.inputs.h_compressor_out 
        h_compressor_in=self.inputs.h_compressor_in
        f =self.inputs.f        
        
        #method 
        deltah_ht=-1/(1+f)*1/self.eta_mech*(h_compressor_out-h_compressor_in)
        
        Tt_out=Tt_in+deltah_ht/Cp
        
        Pt_out=Pt_in*(Tt_out/Tt_in)**(gamma/((gamma-1)*self.etapolt))
        ht_out=Cp*Tt_out   #h(Tt4_5)
        
        #pack outputs
        self.outputs.Tt=Tt_out
        self.outputs.Pt=Pt_out
        self.outputs.ht=ht_out        

        
    __call__ = compute      
    

#--------------------------------------------------------------------------------------


    
class Thrust(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Thrust
        a component that computes the thrust and other output properties
        
        this class is callable, see self.__call__
        
    """    
    
    def __defaults__(self):
        

        self.tag ='Thrust'
        self.alpha=1.0
        self.mdhc =1.0
        self.Tref =1.0
        self.Pref=1.0
        self.no_eng =1.0

        
        self.inputs.fan_exit_velocity =1.0
        self.inputs.core_exit_velocity =1004.
        self.inputs.f = 1.0
        
  

        self.outputs.Thrust=1.0
        self.outputs.sfc=1.0
        self.outputs.Isp=1.0    
        self.outputs.non_dim_thrust=1.0
        self.outputs.mdot_core = 1.0
        self.outputs.fuel_rate=1.0
        self.outputs.mfuel = 1.0
        self.outputs.power = 1.0
        
        
        
     
        
    def compute(self,conditions):
        
        #unpack inputs
        gamma=conditions.freestream.gamma
        Cp = conditions.freestream.Cp 
        
        fan_exit_velocity= self.inputs.fan_exit_velocity 
        core_exit_velocity=self.inputs.core_exit_velocity 
        f= self.inputs.f  
        stag_temp_lpt_exit=self.inputs.stag_temp_lpt_exit
        stag_press_lpt_exit=self.inputs.stag_press_lpt_exit
        
        u0 =  conditions.freestream.velocity
        a0=conditions.freestream.speed_of_sound
        g = conditions.freestream.gravity
        throttle = conditions.propulsion.throttle
        

        #method
        #Engine Properties--------------

        #--specific thrust
        Fsp=((1+f)*core_exit_velocity-u0+self.alpha*(fan_exit_velocity-u0))/((1+self.alpha)*a0)
        

        Isp=Fsp*a0*(1+self.alpha)/(f*g)
    
        TSFC=3600/Isp  
        
        #mass flow sizing
        mdot_core=self.mdhc*np.sqrt(self.Tref/stag_temp_lpt_exit)*(stag_press_lpt_exit/self.Pref)


        fuel_rate=mdot_core*f*self.no_eng
        
        #dimensional thrust
        FD2=Fsp*a0*(1+self.alpha)*mdot_core*self.no_eng*throttle
        
        #--fuel mass flow rate
        mfuel=0.1019715*FD2*TSFC/3600            
        
        power = FD2*u0

        
        #pack outputs

        self.outputs.Thrust=FD2
        self.outputs.sfc=TSFC
        self.outputs.Isp=Isp   
        self.outputs.non_dim_thrust=Fsp
        self.outputs.mdot_core = mdot_core
        self.outputs.fuel_rate= fuel_rate
        self.outputs.mfuel = mfuel     
        self.outputs.power = power  

        
    __call__ = compute         
        
      
        
#--------------------------------------------------------------------------------------


        
# the network
class Network(Data):
    def __defaults__(self):
        
        self.tag = 'Network'
        #self.Nozzle       = SUAVE.Components.Energy.Gas_Turbine.Nozzle()
        #self.Compressor   = SUAVE.Components.Energy.Gas_Turbine.Compressor()
        #self.Combustor    = SUAVE.Components.Energy.Gas_Turbine.Combustor()
        #self.Turbine      = SUAVE.Components.Energy.Gas_Turbine.Turbine()
      

        self.nacelle_dia = 0.0
        self.tag         = 'Network'
        
    _component_root_map = None
        
        
        
    #def __init__(self,*args,**kwarg):
        ## will set defaults
        #super(Network,self).__init__(*args,**kwarg)

        #self._component_root_map = {
            #SUAVE.Components.Energy.Gas_Turbine.Nozzle              : self['Nozzle']              ,
            #SUAVE.Components.Energy.Gas_Turbine.Compressor          : self['Compressor']                  ,
            #SUAVE.Components.Energy.Gas_Turbine.Combustor           : self['Combustor']                ,
            #SUAVE.Components.Energy.Gas_Turbine.Turbine             : self['Turbine']                   ,
                                                  
        #}    
        
        
    
    #def find_component_root(self,component):
        #""" find pointer to component data root.
        #"""

        #component_type = type(component)
        
        

        ## find component root by type, allow subclasses
        #for component_type, component_root in self._component_root_map.iteritems():
            #if isinstance(component,component_type):
                #break
        #else:
            #raise Component_Exception , "Unable to place component type %s" % component.typestring()

        #return component_root        
    
        
    #def append_component(self,component):
        #""" adds a component to network """

        ## assert database type
        #if not isinstance(component,Data):
            #raise Component_Exception, 'input component must be of type Data()'

        ## find the place to store data
        #component_root = self.find_component_root(component)
        

        ## store data
        #component_root.append(component)

        #return            
    
    


           
    
    
    
    # manage process with a driver function
    def evaluate(self,eta,conditions):
    
        # unpack
        
        #conditions.freestream.gamma =1.4
        #conditions.freestream.Cp =1004.

        #conditions.freestream.stagnation_temperature = conditions.freestream.temperature*(1+((conditions.freestream.gamma-1)/2 *conditions.freestream.mach_number**2))
        
        #conditions.freestream.stagnation_pressure = conditions.freestream.pressure* ((1+(conditions.freestream.gamma-1)/2 *conditions.freestream.mach_number**2 )**3.5 )  
        

        self.ram(conditions)
        
        
        
        self.inlet_nozzle.inputs.Tt = self.ram.outputs.Tt #conditions.freestream.stagnation_temperature
        self.inlet_nozzle.inputs.Pt = self.ram.outputs.Pt #conditions.freestream.stagnation_pressure
        
        self.inlet_nozzle(conditions)   
        
        
        

        #---Flow through core------------------------------------------------------
        
        #--low pressure compressor
        self.low_pressure_compressor.inputs.Tt = self.inlet_nozzle.outputs.Tt
        self.low_pressure_compressor.inputs.Pt = self.inlet_nozzle.outputs.Pt
        
        self.low_pressure_compressor(conditions) 
        
        
        #--high pressure compressor
        
        self.high_pressure_compressor.inputs.Tt = self.low_pressure_compressor.outputs.Tt
        self.high_pressure_compressor.inputs.Pt = self.low_pressure_compressor.outputs.Pt        
        
        self.high_pressure_compressor(conditions) 
        
        
        #--Combustor
        self.combustor.inputs.Tt = self.high_pressure_compressor.outputs.Tt
        self.combustor.inputs.Pt = self.high_pressure_compressor.outputs.Pt   
        self.combustor.inputs.nozzle_temp = self.inlet_nozzle.outputs.Tt
        
        self.combustor(conditions)
        
        
        
        #high pressure turbine
        
        self.high_pressure_turbine.inputs.Tt = self.combustor.outputs.Tt
        self.high_pressure_turbine.inputs.Pt = self.combustor.outputs.Pt    
        self.high_pressure_turbine.inputs.h_compressor_out = self.high_pressure_compressor.outputs.ht
        self.high_pressure_turbine.inputs.h_compressor_in = self.low_pressure_compressor.outputs.ht
        self.high_pressure_turbine.inputs.f = self.combustor.outputs.f
        
        self.high_pressure_turbine(conditions)
        
        
        
        #high pressure turbine        
        
        self.low_pressure_turbine.inputs.Tt = self.high_pressure_turbine.outputs.Tt
        self.low_pressure_turbine.inputs.Pt = self.high_pressure_turbine.outputs.Pt    
        self.low_pressure_turbine.inputs.h_compressor_out = self.low_pressure_compressor.outputs.ht
        self.low_pressure_turbine.inputs.h_compressor_in = self.inlet_nozzle.outputs.ht
        self.low_pressure_turbine.inputs.f = self.combustor.outputs.f        
        
        self.low_pressure_turbine(conditions)
        
        
        
        #core nozzle  
        
        self.core_nozzle.inputs.Tt = self.low_pressure_turbine.outputs.Tt
        self.core_nozzle.inputs.Pt = self.low_pressure_turbine.outputs.Pt     
        
        self.core_nozzle(conditions)   
        

        
        #Fan
        
        
        self.fan.inputs.Tt = self.inlet_nozzle.outputs.Tt
        self.fan.inputs.Pt = self.inlet_nozzle.outputs.Pt
        
        self.fan(conditions) 
        
        
        #fan nozzle
        
        self.fan_nozzle.inputs.Tt = self.fan.outputs.Tt
        self.fan_nozzle.inputs.Pt = self.fan.outputs.Pt        
        
        self.fan_nozzle(conditions)   
         
        
        
        #compute thrust
        
        self.thrust.inputs.fan_exit_velocity = self.fan_nozzle.outputs.u
        self.thrust.inputs.core_exit_velocity = self.core_nozzle.outputs.u 
        self.thrust.inputs.f  = self.combustor.outputs.f
        self.thrust.inputs.stag_temp_lpt_exit  = self.low_pressure_turbine.outputs.Tt
        self.thrust.inputs.stag_press_lpt_exit = self.low_pressure_turbine.outputs.Pt
        self.thrust(conditions)
        
 
                
        F = self.thrust.outputs.Thrust
        mdot = self.thrust.outputs.mdot_core
        Isp = self.thrust.outputs.Isp
        P = self.thrust.outputs.power


       # return F,mdot,Isp
        return F[:,0],mdot[:,0],P[:,0]

            
    __call__ = evaluate
    
    
    
    
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------


def test():
    

    #-------Conditions---------------------
    
    # --- Conditions        
    ones_1col = np.ones([1,1])    
    
    
    
    # setup conditions
    conditions = Data()
    conditions.frames       = Data()
    conditions.freestream   = Data()
    conditions.aerodynamics = Data()
    conditions.propulsion   = Data()
    conditions.weights      = Data()
    conditions.energies     = Data()
  #  self.conditions = conditions
    

    # freestream conditions
    conditions.freestream.velocity           = ones_1col*263.
    conditions.freestream.mach_number        = ones_1col*0.8
    conditions.freestream.pressure           = ones_1col*20000.
    conditions.freestream.temperature        = ones_1col*215.
    conditions.freestream.density            = ones_1col* 0.8
    conditions.freestream.speed_of_sound     = ones_1col* 300.
    conditions.freestream.viscosity          = ones_1col* 0.000001
    conditions.freestream.altitude           = ones_1col* 10.
    conditions.freestream.gravity            = ones_1col*9.8
    #conditions.freestream.gamma              =  [1.4]
    #conditions.freestream.Cp                 =  [1004.]    
    #conditions.freestream.stagnation_temperature =  310.
    #conditions.freestream.stagnation_pressure =  22500.        
    #conditions.freestream.reynolds_number    = ones_1col * 0
    #conditions.freestream.dynamic_pressure   = ones_1col * 0
    

    
    # propulsion conditions
    conditions.propulsion.throttle           =  1.0
    #conditions.propulsion.fuel_mass_rate     = ones_1col * 0
    #conditions.propulsion.thrust_breakdown   = Data()
    

    
    
    
    #----------engine propulsion-----------------
    
    
    


    gt_engine = SUAVE.Components.Energy.Gas_Turbine.Network()

    
    #gt_engine = Network('gas_turbine')
    
    #Ram
    ram = SUAVE.Components.Energy.Gas_Turbine.Ram()
    ram.tag = 'ram'
    gt_engine.ram = ram
    
    
    
    
    #inlet nozzle
    inlet_nozzle = SUAVE.Components.Energy.Gas_Turbine.Nozzle()
    inlet_nozzle.tag = 'inlet nozzle'
    gt_engine.inlet_nozzle = inlet_nozzle
    gt_engine.inlet_nozzle.etapold = 1.0
    gt_engine.inlet_nozzle.pid = 1.0
 
    
    
    #low pressure compressor    
    low_pressure_compressor = SUAVE.Components.Energy.Gas_Turbine.Compressor()
    low_pressure_compressor.tag = 'lpc'
    gt_engine.low_pressure_compressor = low_pressure_compressor
    gt_engine.low_pressure_compressor.etapold = 0.94
    gt_engine.low_pressure_compressor.pid = 1.14
    
    

      
    #high pressure compressor  
    high_pressure_compressor = SUAVE.Components.Energy.Gas_Turbine.Compressor()
    high_pressure_compressor.tag = 'hpc'
    gt_engine.high_pressure_compressor = high_pressure_compressor
    gt_engine.high_pressure_compressor.etapold = 0.91
    gt_engine.high_pressure_compressor.pid = 21.4
    
 
    
    
    
    #low pressure turbine  
    low_pressure_turbine = SUAVE.Components.Energy.Gas_Turbine.Turbine()
    low_pressure_turbine.tag='lpt'
    gt_engine.low_pressure_turbine = low_pressure_turbine
    gt_engine.low_pressure_turbine.eta_mech =0.99
    gt_engine.low_pressure_turbine.etapolt = 0.87       
    
    
    #high pressure turbine  
    high_pressure_turbine = SUAVE.Components.Energy.Gas_Turbine.Turbine()
    high_pressure_turbine.tag='hpt'
    gt_engine.high_pressure_turbine = high_pressure_turbine   
    gt_engine.high_pressure_turbine.eta_mech =0.99
    gt_engine.high_pressure_turbine.etapolt = 0.91       
    
    
    #combustor  
    combustor = SUAVE.Components.Energy.Gas_Turbine.Combustor()
    combustor.tag = 'Comb'
    gt_engine.combustor = combustor
    gt_engine.combustor.eta = 0.95
    #gt_engine.combustor.To = To
    gt_engine.combustor.alphac = 1.0     
    gt_engine.combustor.Tt4 =   1400
    gt_engine.combustor.pib =   0.99
    gt_engine.fuel = SUAVE.Attributes.Propellants.Jet_A()
    
    
    #core nozzle
    core_nozzle = SUAVE.Components.Energy.Gas_Turbine.Nozzle()
    core_nozzle.tag = 'core nozzle'
    gt_engine.core_nozzle = core_nozzle
    gt_engine.core_nozzle.etapold = 1.0
    gt_engine.core_nozzle.pid = 1.0
     



    #fan nozzle
    fan_nozzle = SUAVE.Components.Energy.Gas_Turbine.Nozzle()
    fan_nozzle.tag = 'fan nozzle'
    gt_engine.fan_nozzle = fan_nozzle
    gt_engine.fan_nozzle.etapold = 1.0
    gt_engine.fan_nozzle.pid = 1.0

    
    
    #fan    
    fan = SUAVE.Components.Energy.Gas_Turbine.Fan()
    fan.tag = 'fan'
    gt_engine.fan = fan
    gt_engine.fan.etapold = 0.98
    gt_engine.fan.pid = 1.7
    
    #thrust
    thrust = SUAVE.Components.Energy.Gas_Turbine.Thrust()
    thrust.tag ='compute_thrust'
    
    gt_engine.thrust = thrust
    gt_engine.thrust.alpha=6.2
    gt_engine.thrust.mdhc =1.0
    gt_engine.thrust.Tref =273.0
    gt_engine.thrust.Pref=101325
    gt_engine.thrust.no_eng =1.0    

    eta=1.0
    [F,mdot,Isp] = gt_engine(eta,conditions)
    
    print F
    
    



if __name__ == '__main__':   
    test()
    #raise RuntimeError , 'module test failed, not implemented'



# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

#Nozzle.Container = Container
#Compressor.Container = Container
#Combustor.Container = Container
#Turbine.Container = Container
