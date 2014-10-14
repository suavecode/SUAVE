# test_gasturbine_network.py
# 
# Created:  Anil Variyar, August 2014
# Modified: 
#        

""" create and evaluate a gas turbine network
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Attributes import Units

import numpy as np
import pylab as plt

import copy, time

from SUAVE.Structure import (
Data, Container, Data_Exception, Data_Warning,
)

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components import Component_Exception
#from SUAVE.Components.Energy.Gas_Turbine import Network
from SUAVE.Components.Energy.Networks.Turbofan_Network import Turbofan_Network




# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    
    # call the network function
    energy_network()
    

    return


def energy_network():
    
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
    conditions.freestream.mach_number        = ones_1col*0.8
    conditions.freestream.pressure           = ones_1col*20000.
    conditions.freestream.temperature        = ones_1col*215.
    conditions.freestream.density            = ones_1col* 0.8

    conditions.freestream.viscosity          = ones_1col* 0.000001475
    conditions.freestream.altitude           = ones_1col* 10.
    conditions.freestream.gravity            = ones_1col*9.81
    conditions.freestream.gamma              = ones_1col*1.4
    conditions.freestream.Cp                 = 1.4*287.87/(1.4-1)
    conditions.freestream.R                  = 287.87
    conditions.M = conditions.freestream.mach_number 
    conditions.T = conditions.freestream.temperature
    conditions.p = conditions.freestream.pressure
    conditions.freestream.speed_of_sound     = ones_1col* np.sqrt(conditions.freestream.Cp/(conditions.freestream.Cp-conditions.freestream.R)*conditions.freestream.R*conditions.freestream.temperature) #300.
    conditions.freestream.velocity           = conditions.M * conditions.freestream.speed_of_sound
    conditions.velocity = conditions.M * conditions.freestream.speed_of_sound
    conditions.q = 0.5*conditions.freestream.density*conditions.velocity**2
    conditions.g0 = conditions.freestream.gravity
    
    # propulsion conditions
    conditions.propulsion.throttle           =  ones_1col*1.0

    

    
    
    
    #----------engine propulsion-----------------
    
    
    

    #initialize the gas turbine network
    gt_engine = SUAVE.Components.Energy.Networks.Turbofan_Network()
    #gt_engine = SUAVE.Components.Energy.Gas_Turbine.Network()
    
    working_fluid = SUAVE.Attributes.Gases.Air
    gt_engine.working_fluid = working_fluid
#GT_ENGINE.WORKING_FLUID
    

    
    #create a ram component to convert the freestream quantities to stagnation quantities

    ram = SUAVE.Components.Energy.Converters.Ram()
    #ram = SUAVE.Components.Energy.Gas_Turbine.Ram()
    ram.tag = 'ram'
    gt_engine.ram = ram
    
    
    
    
    #create the inlet nozzle to the engine 
    inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    #inlet_nozzle = SUAVE.Components.Energy.Gas_Turbine.Compression_Nozzle()
    inlet_nozzle.tag = 'inlet nozzle'
    #gt_engine.inlet_nozzle = inlet_nozzle
    # input the pressure ratio and polytropic effeciency of the nozzle
    inlet_nozzle.polytropic_efficiency = 0.98
    inlet_nozzle.pressure_ratio = 0.98
    gt_engine.inlet_nozzle = inlet_nozzle
    
    
    #low pressure compressor 
    low_pressure_compressor = SUAVE.Components.Energy.Converters.Compressor()    
    #low_pressure_compressor = SUAVE.Components.Energy.Gas_Turbine.Compressor()
    low_pressure_compressor.tag = 'lpc'
    # input the pressure ratio and polytropic effeciency of the compressor
    low_pressure_compressor.polytropic_efficiency = 0.91
    low_pressure_compressor.pressure_ratio = 1.14    
    gt_engine.low_pressure_compressor = low_pressure_compressor

    
    

      
    #high pressure compressor  
    high_pressure_compressor = SUAVE.Components.Energy.Converters.Compressor()    
    #high_pressure_compressor = SUAVE.Components.Energy.Gas_Turbine.Compressor()
    high_pressure_compressor.tag = 'hpc'
    # input the pressure ratio and polytropic effeciency of the compressor
    high_pressure_compressor.polytropic_efficiency = 0.91
    high_pressure_compressor.pressure_ratio = 13.415    
    gt_engine.high_pressure_compressor = high_pressure_compressor

    
 

    
    #low pressure turbine  
    low_pressure_turbine = SUAVE.Components.Energy.Converters.Turbine()   
    #low_pressure_turbine = SUAVE.Components.Energy.Gas_Turbine.Turbine()
    low_pressure_turbine.tag='lpt'
    # input the pressure ratio and polytropic effeciency of the turbine
    low_pressure_turbine.mechanical_efficiency =0.99
    low_pressure_turbine.polytropic_efficiency = 0.93     
    gt_engine.low_pressure_turbine = low_pressure_turbine
      
    
    
    #high pressure turbine 
    high_pressure_turbine = SUAVE.Components.Energy.Converters.Turbine()   
    #high_pressure_turbine = SUAVE.Components.Energy.Gas_Turbine.Turbine()
    high_pressure_turbine.tag='hpt'
    # input the pressure ratio and polytropic effeciency of the turbine
    high_pressure_turbine.mechanical_efficiency =0.99
    high_pressure_turbine.polytropic_efficiency = 0.93     
    gt_engine.high_pressure_turbine = high_pressure_turbine 
      
    
    
    #combustor  
    combustor = SUAVE.Components.Energy.Converters.Combustor()   
    #combustor = SUAVE.Components.Energy.Gas_Turbine.Combustor()
    combustor.tag = 'Comb'
    # input the effeciency, pressure ratio and the turbine inlet temperature
    combustor.efficiency = 0.99 
    combustor.alphac = 1.0     
    combustor.turbine_inlet_temperature =   1450
    combustor.pressure_ratio =   0.95
    fuel_data = SUAVE.Attributes.Propellants.Jet_A()    
    gt_engine.combustor = combustor

    
    
    #core nozzle
    core_nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    #core_nozzle = SUAVE.Components.Energy.Gas_Turbine.Expansion_Nozzle()
    core_nozzle.tag = 'core nozzle'
    # input the pressure ratio and polytropic effeciency of the nozzle
    core_nozzle.polytropic_efficiency = 0.95
    core_nozzle.pressure_ratio = 0.99    
    gt_engine.core_nozzle = core_nozzle

     



    #fan nozzle
    fan_nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    #fan_nozzle = SUAVE.Components.Energy.Gas_Turbine.Expansion_Nozzle()
    fan_nozzle.tag = 'fan nozzle'
    # input the pressure ratio and polytropic effeciency of the nozzle
    fan_nozzle.polytropic_efficiency = 0.95
    fan_nozzle.pressure_ratio = 0.99    
    gt_engine.fan_nozzle = fan_nozzle


    
    #power out as an output
    #fan    
    fan = SUAVE.Components.Energy.Converters.Fan()   
    #fan = SUAVE.Components.Energy.Gas_Turbine.Fan()
    fan.tag = 'fan'
    # input the pressure ratio and polytropic effeciency of the fan
    fan.polytropic_efficiency = 0.93
    fan.pressure_ratio = 1.7    
    gt_engine.fan = fan

    
    
    #create a thrust component which computes the thrust of the engine
    thrust = SUAVE.Components.Energy.Processes.Thrust()       
    #thrust = SUAVE.Components.Energy.Gas_Turbine.Thrust()
    thrust.tag ='compute_thrust'
    thrust.bypass_ratio=5.4
    thrust.compressor_nondimensional_massflow = 49.7272495725 #1.0
    thrust.reference_temperature =288.15
    thrust.reference_pressure=1.01325*10**5
    thrust.number_of_engines =1.0     
    gt_engine.thrust = thrust
   
    gt_engine.number_of_engines = thrust.number_of_engines

    #bypass ratio  closer to fan
    
    numerics = Data()
    
    eta=1.0
    [F,mdot,Isp] = gt_engine(conditions,numerics)
    
    print 'thrust', F
    print '\n'

    
    
    ##--------------Turbofan based validation----------
    
    
    ### ------------------------------------------------------------------
    ###  Turbofan
    ### ------------------------------------------------------------------    
    
    #turbofan = SUAVE.Components.Propulsors.TurboFanPASS()
    #turbofan.tag = 'Turbo Fan'
    
    #turbofan.propellant = SUAVE.Attributes.Propellants.Jet_A()
    
    #turbofan.analysis_type                 = '1D'     #
    #turbofan.diffuser_pressure_ratio       = 0.98     #
    #turbofan.fan_pressure_ratio            = 1.7      #
    #turbofan.fan_nozzle_pressure_ratio     = 0.99     #
    #turbofan.lpc_pressure_ratio            = 1.14     #
    #turbofan.hpc_pressure_ratio            = 13.415   #
    #turbofan.burner_pressure_ratio         = 0.95     #
    #turbofan.turbine_nozzle_pressure_ratio = 0.99     #
    #turbofan.Tt4                           = 1450.0   #
    #turbofan.bypass_ratio                  = 5.4      #
    #turbofan.design_thrust                 = 25000.0  #
    #turbofan.no_of_engines                 = 1.0      #
    #turbofan.number_of_engines             = turbofan.no_of_engines
   ## turbofan.thrust.design                 = 25000.0  #
    
    ## turbofan sizing conditions
    #sizing_segment = SUAVE.Components.Propulsors.Segments.Segment()
    
    #sizing_segment.M   = 0.8          #
    #sizing_segment.alt = 10.0         #
    #sizing_segment.T   = 218.0        #
    #sizing_segment.p   = 0.239*10**5  # 
    
 
    

    
    ## size the turbofan
    #turbofan.engine_sizing_1d(sizing_segment)  
    
    ##sls_thrust = turbofan.sea_level_static()
    
    #CF, Isp, eta_Pe = turbofan(conditions.propulsion.throttle,conditions)
    
    
    
    

    
if __name__ == '__main__':
    
    main()