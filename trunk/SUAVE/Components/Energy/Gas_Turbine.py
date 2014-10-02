# Gas_Turbine.py
# 
# Created:  Anil, July 2014
      
#--put in a folder

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
#from SUAVE.Components.Energy.Gas_Turbine import Network


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

def fm_id(M):

    R=287.87
    g=1.4
    m0=(g+1)/(2*(g-1))
    m1=((g+1)/2)**m0
    m2=(1+(g-1)/2*M**2)**m0
    fm=m1*M/m2
    return fm

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
        
    
        self.outputs.stagnation_temperature=1.0
        self.outputs.stagnation_pressure =1.0

        

        
    def compute(self,conditions):
        
        #unpack the variables

        Po = conditions.freestream.pressure
        To = conditions.freestream.temperature
    

    
        
        #method
        conditions.freestream.gamma =1.4
        conditions.freestream.Cp =1.4*287.87/(1.4-1)
        
        
        #Compute the stagnation quantities from the input static quantities
        conditions.freestream.stagnation_temperature = conditions.freestream.temperature*(1+((conditions.freestream.gamma-1)/2 *conditions.freestream.mach_number**2))
        
        conditions.freestream.stagnation_pressure = conditions.freestream.pressure* ((1+(conditions.freestream.gamma-1)/2 *conditions.freestream.mach_number**2 )**3.5 )  
        conditions.freestream.Cp                 = 1.4*287.87/(1.4-1)
        conditions.freestream.R                  = 287.87             
         
        
        #pack outputs
        self.outputs.stagnation_temperature =conditions.freestream.stagnation_temperature
        self.outputs.stagnation_pressure =conditions.freestream.stagnation_pressure
        
        

        

        
    __call__ = compute


#--------------------------------------------------------------------------------------

class Compression_Nozzle(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Nozzle
        a nozzle component 
        
        this class is callable, see self.__call__
        
    """    
    
    def __defaults__(self):
        

        self.tag = 'Nozzle'

        self.polytropic_efficiency = 1.0
        self.pressure_ratio = 1.0

        self.inputs.stagnation_temperature=0.
        self.inputs.stagnation_pressure=0.
        
        self.outputs.stagnation_temperature=0.
        self.outputs.stagnation_pressure=0.
        self.outputs.stagnation_enthalpy=0.
        

        
    def compute(self,conditions):
        
        #unpack the variables
        gamma=conditions.freestream.gamma
        Cp = conditions.freestream.Cp
        Po = conditions.freestream.pressure
        R = conditions.freestream.R
        Tt_in = self.inputs.stagnation_temperature
        Pt_in = self.inputs.stagnation_pressure
        pid = self.pressure_ratio
        etapold =  self.polytropic_efficiency

    
        
        #Computing the output modules
        
        #--Getting the outptu stagnation quantities
        Pt_out=Pt_in*pid
        Tt_out=Tt_in*pid**((gamma-1)/(gamma*etapold))
        ht_out=Cp*Tt_out 
        
        
        #compute the output Mach number, static quantities and the output velocity
        Mach=np.sqrt((((Pt_out/Po)**((gamma-1)/gamma))-1)*2/(gamma-1))
        T_out=Tt_out/(1+(gamma-1)/2*Mach**2)
        h_out=Cp*T_out
        u_out=np.sqrt(2*(ht_out-h_out))  
        
        

        #pack outputs
        self.outputs.stagnation_temperature=Tt_out
        self.outputs.stagnation_pressure=Pt_out
        self.outputs.stagnation_enthalpy=ht_out
        self.outputs.mach_number = Mach
        self.outputs.static_temperature = T_out
        self.outputs.static_enthalpy = h_out
        self.outputs.velocity = u_out
        
        

        

        
    __call__ = compute
        
        
class Expansion_Nozzle(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Nozzle
        a nozzle component 
        
        this class is callable, see self.__call__
        
    """    
    
    def __defaults__(self):
        

        self.tag = 'Nozzle'

        self.polytropic_efficiency = 1.0
        self.pressure_ratio = 1.0

        self.inputs.stagnation_temperature=0.
        self.inputs.stagnation_pressure=0.
        
        self.outputs.stagnation_temperature=0.
        self.outputs.stagnation_pressure=0.
        self.outputs.stagnation_enthalpy=0.
        

        
    def compute(self,conditions):
        
        #unpack the variables
        gamma=conditions.freestream.gamma
        Cp = conditions.freestream.Cp
        Po = conditions.freestream.pressure
        Pto = conditions.freestream.stagnation_pressure
        Tto = conditions.freestream.stagnation_temperature
        R = conditions.freestream.R
        Mo =  conditions.freestream.mach_number
        Tt_in = self.inputs.stagnation_temperature
        Pt_in = self.inputs.stagnation_pressure
        pid = self.pressure_ratio
        etapold = self.polytropic_efficiency
    
        
        #Computing the output modules
        
        #--Getting the outptu stagnation quantities
        Pt_out=Pt_in*pid
        Tt_out=Tt_in*pid**((gamma-1)/(gamma)*etapold)
        ht_out=Cp*Tt_out 
        
        
        #compute the output Mach number, static quantities and the output velocity
        Mach=np.sqrt((((Pt_out/Po)**((gamma-1)/gamma))-1)*2/(gamma-1))
        T_out=Tt_out/(1+(gamma-1)/2*Mach**2)
        h_out=Cp*T_out
        u_out=np.sqrt(2*(ht_out-h_out))  
        
        
        
        if np.linalg.norm(Mach) < 1.0:
        # nozzle unchoked
        
            P_out=Po
            
            Mach=np.sqrt((((Pt_out/Po)**((gamma-1)/gamma))-1)*2/(gamma-1))
            Tt_out=Tt_out/(1+(gamma-1)/2*Mach**2)
            h_out=Cp*T_out
        
        else:
            Mach=1
            T_out=Tt_out/(1+(gamma-1)/2*Mach**2)
            P_out=Pt_out/(1+(gamma-1)/2*Mach**2)**(gamma/(gamma-1))
            h_out=Cp*T_out
          
        # 
        u_out=np.sqrt(2*(ht_out-h_out))
        rho_out=P_out/(R*T_out)
        
        
        area_ratio=(fm_id(Mo)/fm_id(Mach)*(1/(Pt_out/Pto))*(np.sqrt(Tt_out/Tto)))
        

         
        
        #pack outputs
        self.outputs.stagnation_temperature=Tt_out
        self.outputs.stagnation_pressure=Pt_out
        self.outputs.stagnation_enthalpy=ht_out
        self.outputs.mach_number = Mach
        self.outputs.static_temperature = T_out
        self.outputs.static_enthalpy = h_out
        self.outputs.velocity = u_out
        self.outputs.static_pressure = P_out
        self.outputs.area_ratio = area_ratio
        
        

        

        
    __call__ = compute
        
#--------------------------------------------------------------------------------------
    
class Compressor(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Compressor
        a compressor component
        
        this class is callable, see self.__call__
        
    """    
    
    def __defaults__(self):
        

        self.tag = 'Compressor'

        self.polytropic_efficiency = 1.0
        self.pressure_ratio = 1.0
        
        self.inputs.stagnation_temperature=0.
        self.inputs.stagnation_pressure=0.
        
        self.outputs.stagnation_temperature=0.
        self.outputs.stagnation_pressure=0.
        self.outputs.stagnation_enthalpy=0.        
 
        
        

    def compute(self,conditions):
        
        #unpack the variables
        gamma=conditions.freestream.gamma
        Cp = conditions.freestream.Cp  
        Tt_in = self.inputs.stagnation_temperature
        Pt_in = self.inputs.stagnation_pressure      
        pid = self.pressure_ratio
        etapold =  self.polytropic_efficiency
    
        #Compute the output stagnation quantities based on the pressure ratio of the component
        
        ht_in = Cp*Tt_in
        
        Pt_out=Pt_in*pid
        Tt_out=Tt_in*pid**((gamma-1)/(gamma*etapold))
        ht_out=Cp*Tt_out 
        work_done = ht_out- ht_in
        
        #pack outputs
        self.outputs.stagnation_temperature=Tt_out
        self.outputs.stagnation_pressure=Pt_out
        self.outputs.stagnation_enthalpy= ht_out 
        self.outputs.work_done = work_done
        

    __call__ = compute
    
    
#--------------------------------------------------------------------------------------
    
class Fan(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Fan
        a Fan component
        
        this class is callable, see self.__call__
        
    """    
    
    def __defaults__(self):
        

        self.tag ='Fan'

        self.polytropic_efficiency = 1.0
        self.pressure_ratio = 1.0
        
        self.inputs.stagnation_temperature=0.
        self.inputs.stagnation_pressure=0.
        
        self.outputs.stagnation_temperature=0.
        self.outputs.stagnation_pressure=0.
        self.outputs.stagnation_enthalpy=0.        
      
        
        
    def compute(self,conditions):
        
        gamma=conditions.freestream.gamma
        Cp = conditions.freestream.Cp  
        Tt_in = self.inputs.stagnation_temperature
        Pt_in = self.inputs.stagnation_pressure   
        pid = self.pressure_ratio
        etapold =  self.polytropic_efficiency        
    
    
        #Compute the output stagnation quantities based on the pressure ratio of the component
        ht_in = Cp*Tt_in        
        
        Pt_out=Pt_in*pid
        Tt_out=Tt_in*pid**((gamma-1)/(gamma*etapold))
        ht_out=Cp*Tt_out    #h(Tt1_8)
        
        work_done = ht_out- ht_in
        #pack outputs
        self.outputs.stagnation_temperature=Tt_out
        self.outputs.stagnation_pressure=Pt_out
        self.outputs.stagnation_enthalpy=ht_out 
        
        self.outputs.work_done = work_done
        

        
        
    __call__ = compute  
        

#--------------------------------------------------------------------------------------


    
class Combustor(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Combustor
        a combustor component
        
        this class is callable, see self.__call__
        
    """    
    
    def __defaults__(self):
        

        self.tag = 'Combustor'

        self.alphac = 0.0 
        self.turbine_inlet_temperature = 1.0
        self.fuel_data = SUAVE.Attributes.Propellants.Jet_A()
        
        self.inputs.stagnation_temperature = 1.0
        self.inputs.stagnation_pressure = 1.0  
        #self.inputs.nozzle_temp = 1.0     
        
        self.outputs.stagnation_temperature=1.0
        self.outputs.stagnation_pressure=1.0
        self.outputs.stagnation_enthalpy=1.0
        self.outputs.fuel_to_air_ratio = 1.0        
        
        

        
        
    def compute(self,conditions):
        
        #unpack the variables
        gamma=conditions.freestream.gamma
        Cp = conditions.freestream.Cp
        To = conditions.freestream.temperature
        Tto = conditions.freestream.stagnation_temperature
        
        Tt_in = self.inputs.stagnation_temperature 
        Pt_in = self.inputs.stagnation_pressure   
        Tt_n = self.inputs.nozzle_exit_stagnation_temperature
        Tt4 = self.turbine_inlet_temperature
        pib = self.pressure_ratio
        #Tto = self.inputs.freestream_stag_temp
        
        htf=self.fuel_data.specific_energy
        ht4 = Cp*Tt4
        ho = Cp*To
        
        
        
        
        
        #Using the Turbine exit temperature, the fuel properties and freestream temperature to compute the fuel to air ratio f        
        tau = htf/(Cp*To)
        tau_freestream=Tto/To
        
    
        #f=(((self.Tt4/To)-tau_freestream*(Tt_in/Tt_n))/(self.eta*tau-(self.Tt4/To)))
        
        f = (ht4 - ho)/(htf-ht4)
        #print ((self.Tt4/To)-tau_freestream*(Tt_in/Tt_n))
        
        ht_out=Cp*Tt4
        Pt_out=Pt_in*pib  
        
        #pack outputs
        self.outputs.stagnation_temperature=Tt4
        self.outputs.stagnation_pressure=Pt_out
        self.outputs.stagnation_enthalpy=ht_out 
        self.outputs.fuel_to_air_ratio = f #0.0215328 #f
        

        
    __call__ = compute
   
        
#--------------------------------------------------------------------------------------


class Turbine(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Turbine
        a Turbine component
        
        this class is callable, see self.__call__
        
    """    
    
    def __defaults__(self):
        

        self.tag ='Turbine'
        self.mechanical_efficiency =1.0
        self.polytropic_efficiency = 1.0
        
        self.inputs.stagnation_temperature = 1.0
        self.inputs.stagnation_pressure = 1.0  
        self.inputs.fuel_to_air_ratio = 1.0
        

        self.outputs.stagnation_temperature=1.0
        self.outputs.stagnation_pressure=1.0
        self.outputs.stagnation_enthalpy=1.0    
        
        
     
        
    def compute(self,conditions):
        
        #unpack inputs
        gamma=conditions.freestream.gamma
        Cp = conditions.freestream.Cp 
        
        Tt_in =self.inputs.stagnation_temperature 
        Pt_in =self.inputs.stagnation_pressure   

        alpha =  self.inputs.bypass_ratio
        f =self.inputs.fuel_to_air_ratio  
        compressor_work = self.inputs.compressor.work_done
        fan_work = self.inputs.fan.work_done
        
        eta_mech =  self.mechanical_efficiency
        etapolt =  self.polytropic_efficiency
        
        #Using the stagnation enthalpy drop across the corresponding turbine and the fuel to air ratio to compute the energy drop across the turbine
        deltah_ht=-1/(1+f)*1/eta_mech*((compressor_work)+ alpha*(fan_work))
        
        #Compute the output stagnation quantities from the inputs and the energy drop computed above 
        Tt_out=Tt_in+deltah_ht/Cp
        Pt_out=Pt_in*(Tt_out/Tt_in)**(gamma/((gamma-1)*etapolt))
        ht_out=Cp*Tt_out   #h(Tt4_5)
        
        
        
        #pack outputs
        self.outputs.stagnation_temperature=Tt_out
        self.outputs.stagnation_pressure=Pt_out
        self.outputs.stagnation_enthalpy=ht_out        

        
    __call__ = compute      
    

#--------------------------------------------------------------------------------------


   


    
class Thrust(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Thrust
        a component that computes the thrust and other output properties
        
        this class is callable, see self.__call__
        
    """    
    
    def __defaults__(self):
        

        self.tag ='Thrust'
        self.bypass_ratio=1.0
        self.compressor_nondimensional_massflow =1.0
        self.reference_temperature =1.0
        self.reference_pressure=1.0
        self.number_of_engines =1.0

        

        self.inputs.fuel_to_air_ratio = 1.0
        
  

        self.outputs.thrust=1.0
        self.outputs.thrust_specific_fuel_consumption=1.0
        self.outputs.specific_impulse=1.0    
        self.outputs.non_dimensional_thrust=1.0
        self.outputs.core_mass_flow_rate = 1.0
        self.outputs.fuel_flow_rate=1.0
        self.outputs.fuel_mass = 1.0
        self.outputs.power = 1.0
        
        
        
     
        
    def compute(self,conditions):
        
        #unpack inputs
        gamma=conditions.freestream.gamma
        Cp = conditions.freestream.Cp 
        
 
        f= self.inputs.fuel_to_air_ratio  
        stag_temp_lpt_exit=self.inputs.stag_temp_lpt_exit
        stag_press_lpt_exit=self.inputs.stag_press_lpt_exit
      
        core_nozzle = self.inputs.core_nozzle
        fan_nozzle = self.inputs.fan_nozzle
        
        fan_exit_velocity= self.inputs.fan_nozzle.velocity 
        core_exit_velocity=self.inputs.core_nozzle.velocity   
        
        fan_area_ratio = self.inputs.fan_nozzle.area_ratio
        core_area_ratio = self.inputs.core_nozzle.area_ratio         

        u0 = conditions.freestream.velocity
        a0=conditions.freestream.speed_of_sound
        M0 = conditions.freestream.mach_number
        p0 = conditions.freestream.pressure
        
        g = conditions.freestream.gravity
        throttle = conditions.propulsion.throttle
        
        bypass_ratio =  self.bypass_ratio
        Tref = self.reference_temperature
        Pref = self.reference_pressure
        no_eng = self.number_of_engines
        mdhc =  self.compressor_nondimensional_massflow
        


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

        Ae_b_Ao=1/(1+bypass_ratio)*core_area_ratio
        
        print 'Ae_b_Ao',Ae_b_Ao        
        
        A1e_b_A1o=bypass_ratio/(1+bypass_ratio)*fan_area_ratio
         
         
        print 'A1e_b_A1o',A1e_b_A1o          
         
         
        Thrust_nd=gamma*M0**2*(1/(1+bypass_ratio)*(core_nozzle.velocity/u0-1)+(bypass_ratio/(1+bypass_ratio))*(fan_nozzle.velocity/u0-1))+Ae_b_Ao*(core_nozzle.static_pressure/p0-1)+A1e_b_A1o*(fan_nozzle.static_pressure/p0-1)
        
        
        
        
        ##calculate actual value of thrust 
        
        Fsp=1/(gamma*M0)*Thrust_nd
        
        print 'Fsp ',Fsp

      ##overall engine quantities
        
        Isp=Fsp*a0*(1+bypass_ratio)/(f*g)
        TSFC=3600/Isp  # for the test case 


        print 'TSFC ',TSFC
        
        #mass flow sizing
        mdot_core=mdhc*np.sqrt(Tref/stag_temp_lpt_exit)*(stag_press_lpt_exit/Pref)
        
        ##mdot_core=FD/(Fsp*ao*(1+aalpha))
        ##print mdot_core
        print 'mdot_core ',stag_temp_lpt_exit
      
        ##-------if areas specified-----------------------------
        fuel_rate=mdot_core*f*no_eng
        
        FD2=Fsp*a0*(1+bypass_ratio)*mdot_core*no_eng*throttle
        mfuel=0.1019715*FD2*TSFC/3600
        ###State.config.A_engine=A22
        
        print 'Thrust' , FD2        
        
        power = FD2*u0
        
        #pack outputs

        self.outputs.thrust= FD2 #thrust
        self.outputs.thrust_specific_fuel_consumption=TSFC
        self.outputs.specific_impulse=Isp   
        self.outputs.non_dimensional_thrust=Fsp #specific_thrust
        self.outputs.core_mass_flow_rate = mdot_core
        self.outputs.fuel_flow_rate= fuel_rate
        self.outputs.fuel_mass = mfuel     
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
  


        ram(conditions)
        

        
        
        inlet_nozzle.inputs.stagnation_temperature = ram.outputs.stagnation_temperature #conditions.freestream.stagnation_temperature
        inlet_nozzle.inputs.stagnation_pressure = ram.outputs.stagnation_pressure #conditions.freestream.stagnation_pressure
        
    
        print 'ram out temp ', ram.outputs.stagnation_temperature
        print 'ram out press', ram.outputs.stagnation_pressure    
        
        
        inlet_nozzle(conditions)   
        
        
        print 'inlet nozzle out temp ', inlet_nozzle.outputs.stagnation_temperature
        print 'inlet nozzle out press', inlet_nozzle.outputs.stagnation_pressure         
        print 'inlet nozzle out h', inlet_nozzle.outputs.stagnation_enthalpy         

        #---Flow through core------------------------------------------------------
        
        #--low pressure compressor
        low_pressure_compressor.inputs.stagnation_temperature = inlet_nozzle.outputs.stagnation_temperature
        low_pressure_compressor.inputs.stagnation_pressure = inlet_nozzle.outputs.stagnation_pressure
        
        low_pressure_compressor(conditions) 
        
        print 'low_pressure_compressor out temp ', low_pressure_compressor.outputs.stagnation_temperature
        print 'low_pressure_compressor out press', low_pressure_compressor.outputs.stagnation_pressure 
        print 'low_pressure_compressor out h', low_pressure_compressor.outputs.stagnation_enthalpy
        #--high pressure compressor
        
        high_pressure_compressor.inputs.stagnation_temperature = low_pressure_compressor.outputs.stagnation_temperature
        high_pressure_compressor.inputs.stagnation_pressure = low_pressure_compressor.outputs.stagnation_pressure        
        
        high_pressure_compressor(conditions) 
        
        print 'high_pressure_compressor out temp ', high_pressure_compressor.outputs.stagnation_temperature
        print 'high_pressure_compressor out press', high_pressure_compressor.outputs.stagnation_pressure            
        print 'high_pressure_compressor out h', high_pressure_compressor.outputs.stagnation_enthalpy
        
        
        #Fan
        
        
        fan.inputs.stagnation_temperature = inlet_nozzle.outputs.stagnation_temperature
        fan.inputs.stagnation_pressure = inlet_nozzle.outputs.stagnation_pressure
        
        fan(conditions) 
        
        print 'fan out temp ', fan.outputs.stagnation_temperature
        print 'fan out press', fan.outputs.stagnation_pressure     
        print 'fan out h', fan.outputs.stagnation_enthalpy
        

        
        #--Combustor
        combustor.inputs.stagnation_temperature = high_pressure_compressor.outputs.stagnation_temperature
        combustor.inputs.stagnation_pressure = high_pressure_compressor.outputs.stagnation_pressure   
        combustor.inputs.nozzle_exit_stagnation_temperature = inlet_nozzle.outputs.stagnation_temperature
        
        combustor(conditions)
        
        print 'combustor out temp ', combustor.outputs.stagnation_temperature
        print 'combustor out press', combustor.outputs.stagnation_pressure          
        print 'combustor out f', combustor.outputs.fuel_to_air_ratio
        print 'combustor out h', combustor.outputs.stagnation_enthalpy
        
        #high pressure turbine
        
        high_pressure_turbine.inputs.stagnation_temperature = combustor.outputs.stagnation_temperature
        high_pressure_turbine.inputs.stagnation_pressure = combustor.outputs.stagnation_pressure    
        high_pressure_turbine.inputs.compressor = high_pressure_compressor.outputs
        high_pressure_turbine.inputs.fuel_to_air_ratio = combustor.outputs.fuel_to_air_ratio
        high_pressure_turbine.inputs.fan =  fan.outputs
        high_pressure_turbine.inputs.bypass_ratio =0.0    
        
        high_pressure_turbine(conditions)
        
        print 'high_pressure_turbine out temp ', high_pressure_turbine.outputs.stagnation_temperature
        print 'high_pressure_turbine out press', high_pressure_turbine.outputs.stagnation_pressure       
        print 'high_pressure_turbine out h', high_pressure_turbine.outputs.stagnation_enthalpy
        
        #low pressure turbine        
        
        low_pressure_turbine.inputs.stagnation_temperature = high_pressure_turbine.outputs.stagnation_temperature
        low_pressure_turbine.inputs.stagnation_pressure = high_pressure_turbine.outputs.stagnation_pressure    
        low_pressure_turbine.inputs.compressor = low_pressure_compressor.outputs
        low_pressure_turbine.inputs.fuel_to_air_ratio = combustor.outputs.fuel_to_air_ratio   
        low_pressure_turbine.inputs.fan =  fan.outputs
        low_pressure_turbine.inputs.bypass_ratio =  thrust.bypass_ratio   
        
        low_pressure_turbine(conditions)
        
        print 'low_pressure_turbine out temp ', low_pressure_turbine.outputs.stagnation_temperature
        print 'low_pressure_turbine out press', low_pressure_turbine.outputs.stagnation_pressure        
        print 'low_pressure_turbine out h', low_pressure_turbine.outputs.stagnation_enthalpy        
        
        #core nozzle  
        
        core_nozzle.inputs.stagnation_temperature = low_pressure_turbine.outputs.stagnation_temperature
        core_nozzle.inputs.stagnation_pressure = low_pressure_turbine.outputs.stagnation_pressure     
        
        core_nozzle(conditions)   
        
        print 'core_nozzle out temp ', core_nozzle.outputs.stagnation_temperature
        print 'core_nozzle out press', core_nozzle.outputs.stagnation_pressure        
        print 'core_nozzle out h', core_nozzle.outputs.stagnation_enthalpy        

        

        
        
        #fan nozzle
        
        fan_nozzle.inputs.stagnation_temperature = fan.outputs.stagnation_temperature
        fan_nozzle.inputs.stagnation_pressure = fan.outputs.stagnation_pressure        
        
        fan_nozzle(conditions)   
         
        print 'fan_nozzle out temp ', fan_nozzle.outputs.stagnation_temperature
        print 'fan_nozzle out press', fan_nozzle.outputs.stagnation_pressure        
        print 'fan_nozzle out h', fan_nozzle.outputs.stagnation_enthalpy         
        
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


       # return F,mdot,Isp
        return F[:,0],mdot[:,0],P[:,0]  #return the 2d array instead of the 1D array

            
    __call__ = evaluate
    
    
    
    
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
#make 2 test scripts
#one like below
#one full mission

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
    conditions.freestream.velocity           = ones_1col*223.
    conditions.freestream.mach_number        = ones_1col*0.8
    conditions.freestream.pressure           = ones_1col*20000.
    conditions.freestream.temperature        = ones_1col*215.
    conditions.freestream.density            = ones_1col* 0.8
    conditions.freestream.speed_of_sound     = ones_1col* 300.
    conditions.freestream.viscosity          = ones_1col* 0.000001
    conditions.freestream.altitude           = ones_1col* 10.
    conditions.freestream.gravity            = ones_1col*9.8

    

    
    # propulsion conditions
    conditions.propulsion.throttle           =  1.0

    

    
    
    
    #----------engine propulsion-----------------
    
    
    


    gt_engine = SUAVE.Components.Energy.Gas_Turbine.Network()

    

    
    #Ram
    ram = SUAVE.Components.Energy.Gas_Turbine.Ram()
    ram.tag = 'ram'
    gt_engine.ram = ram
    
    
    
    
    #inlet nozzle
    inlet_nozzle = SUAVE.Components.Energy.Gas_Turbine.Compression_Nozzle()
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
    gt_engine.high_pressure_compressor.pid = 13.2
    
 

    
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
    core_nozzle = SUAVE.Components.Energy.Gas_Turbine.Expansion_Nozzle()
    core_nozzle.tag = 'core nozzle'
    gt_engine.core_nozzle = core_nozzle
    gt_engine.core_nozzle.etapold = 1.0
    gt_engine.core_nozzle.pid = 1.0
     



    #fan nozzle
    fan_nozzle = SUAVE.Components.Energy.Gas_Turbine.Expansion_Nozzle()
    fan_nozzle.tag = 'fan nozzle'
    gt_engine.fan_nozzle = fan_nozzle
    gt_engine.fan_nozzle.etapold = 1.0
    gt_engine.fan_nozzle.pid = 1.0

    
    #power out as an output
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


    #byoass ratio  closer to fan
    
    numerics = Data()
    
    eta=1.0
    [F,mdot,Isp] = gt_engine(conditions,numerics)
    
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
