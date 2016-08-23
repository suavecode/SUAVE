#Turbofan_Network.py
# 
# Created:  Anil Variyar, Feb 2016
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

#import VyPy
#from VyPy import *
#from regression import gpr

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn
import copy


from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components import Component_Exception
from SUAVE.Components.Propulsors.Propulsor import Propulsor
from SUAVE.Plugins.VyPy.regression import gpr
#from Turbofan_Jacobian import Turbofan_Jacobian
from Turbofan_TASOPTc import Turbofan_TASOPTc


# ----------------------------------------------------------------------
#  Turbofan Network
# ----------------------------------------------------------------------

class Turbofan_TASOPTc_wrap(Propulsor):
    
    def __defaults__(self):
        
        #setting the default values
        self.tag = 'Turbo_Fan'
        self.number_of_engines = 1.0
        self.nacelle_diameter  = 1.0
        self.engine_length     = 1.0
        self.bypass_ratio      = 1.0   
        
        self.design_params     = None
        self.offdesign_params  = None
        self.max_iters         = 1
        self.newton_relaxation = 0.3 #0.8*np.ones(8)     
        #self.newton_relaxation[6] = 1.0
        self.compressor_map_file = "Compressor_map.txt"
        self.cooling_flow = 0
        self.no_of_turbine_stages = 0
        self.c_model = None
        #self.an_jacobian = None
    
    
    _component_root_map = None
        
    
    def unpack(self):

        self.c_model = Turbofan_TASOPTc()
        self.design_params    = Data()
        self.offdesign_params = Data()
        #self.an_jacobian = Turbofan_Jacobian()
        
#        eng_params = np.zeros(23)
#        
#        eng_params[0] = self.bypass_ratio  
#        
#        eng_params[1] = self.inlet_nozzle.pressure_ratio
#        eng_params[2] = self.inlet_nozzle.polytropic_efficiency
#        
#        eng_params[3] = self.fan.pressure_ratio
#        eng_params[4] = self.fan.polytropic_efficiency
#        
#        eng_params[5] = self.fan_nozzle.pressure_ratio
#        eng_params[6] = self.fan_nozzle.polytropic_efficiency
#        
#        eng_params[7] = self.low_pressure_compressor.pressure_ratio
#        eng_params[8] = self.low_pressure_compressor.polytropic_efficiency
#        
#        eng_params[9] = self.high_pressure_compressor.pressure_ratio
#        eng_params[10] = self.high_pressure_compressor.polytropic_efficiency
#        
#        eng_params[11] = self.combustor.turbine_inlet_temperature
#        eng_params[12] = self.combustor.pressure_ratio
#        eng_params[13] = self.combustor.efficiency
#        eng_params[14] = self.combustor.fuel_data.specific_energy 
#        
#        eng_params[15] = self.low_pressure_turbine.polytropic_efficiency
#        eng_params[16] = self.low_pressure_turbine.mechanical_efficiency
#        
#        eng_params[17] = self.high_pressure_turbine.polytropic_efficiency
#        eng_params[18] = self.high_pressure_turbine.mechanical_efficiency
#        
#        eng_params[19] = self.core_nozzle.pressure_ratio
#        eng_params[20] = self.core_nozzle.polytropic_efficiency
#        
#        eng_params[21] = self.fan.hub_to_tip_ratio
#        eng_params[22] = self.high_pressure_compressor.hub_to_tip_ratio



        self.c_model.dp.aalpha = self.bypass_ratio
        
        self.c_model.dp.pi_d = self.inlet_nozzle.pressure_ratio
        self.c_model.dp.eta_d = self.inlet_nozzle.polytropic_efficiency
        
        self.c_model.dp.pi_f = self.fan.pressure_ratio
        self.c_model.dp.eta_f = self.fan.polytropic_efficiency
        
        self.c_model.dp.pi_fn = self.fan_nozzle.pressure_ratio
        self.c_model.dp.eta_fn = self.fan_nozzle.polytropic_efficiency
        
        self.c_model.dp.pi_lc = self.low_pressure_compressor.pressure_ratio
        self.c_model.dp.eta_lc = self.low_pressure_compressor.polytropic_efficiency
        
        self.c_model.dp.pi_hc = self.high_pressure_compressor.pressure_ratio
        self.c_model.dp.eta_hc = self.high_pressure_compressor.polytropic_efficiency
        
        self.c_model.dp.Tt4 = self.combustor.turbine_inlet_temperature
        self.c_model.dp.pi_b = self.combustor.pressure_ratio
        self.c_model.dp.eta_b = self.combustor.efficiency
        self.c_model.dp.htf = self.combustor.fuel_data.specific_energy
        
        self.c_model.dp.eta_lt = self.low_pressure_turbine.polytropic_efficiency
        self.c_model.dp.etam_lt = self.low_pressure_turbine.mechanical_efficiency
        
        self.c_model.dp.eta_ht = self.high_pressure_turbine.polytropic_efficiency
        self.c_model.dp.etam_ht = self.high_pressure_turbine.mechanical_efficiency
        
        self.c_model.dp.pi_tn = self.core_nozzle.pressure_ratio
        self.c_model.dp.eta_tn = self.core_nozzle.polytropic_efficiency
        
        self.c_model.dp.HTR_f = self.fan.hub_to_tip_ratio
        self.c_model.dp.HTR_hc = self.high_pressure_compressor.hub_to_tip_ratio
        
        #print self.c_model.max_iters
        
        self.c_model.unpack()
    
    
    def size(self,mach_number,altitude,delta_isa = 0.):  
        
        #Unpack components
        atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        p,T,rho,a,mu = atmosphere.compute_values(altitude,delta_isa)
        
             
        #print mach_number,p[0][0],T[0][0],1.0,self.thrust.total_design,self.number_of_engines
             
             
        self.c_model.size(mach_number,p[0][0],T[0][0],1.0,self.thrust.total_design,int(self.number_of_engines))
            
        
        self.sealevel_static_thrust = self.c_model.sls_thrust
        self.df = self.c_model.fan_diameter
        
        results = Data()
        
        return results






    def evaluate_thrust(self,state):
        #results_offdesign = self.offdesign(state)
        
        #imports
        conditions = state.conditions
        numerics   = state.numerics
        #reference = self.reference
        throttle = conditions.propulsion.throttle
        
        local_state = copy.deepcopy(state)
        local_throttle = copy.deepcopy(throttle)
        
        #throttle = 0.6 + conditions.propulsion.throttle*(1.0-0.6)/1.0
        local_throttle[throttle<0.6] = 0.6
        local_throttle[throttle>1.0] = 1.0
        
        local_state.conditions.propulsion.throttle = local_throttle
        
        
        #freestream properties
        T0 = conditions.freestream.temperature
        p0 = conditions.freestream.pressure
        M0 = conditions.freestream.mach_number  
        
        
        F = np.zeros([len(T0),3])
        mdot0 = np.zeros([len(T0),1])
        S  = np.zeros(len(T0))
        F_mdot0 = np.zeros(len(T0))          
        
        local_scale = throttle/local_throttle
        
        for iseg in range(0,len(M0)):
            M = M0[iseg][0]
            p = p0[iseg][0]
            T = T0[iseg][0]
            throttle_s = 1.0 #local_throttle[iseg][0]
            
            #print M,p,T,throttle_s
            
            
            
            self.c_model.offdesign(M,p,T,throttle_s)
            
        
            F[iseg,0] = self.c_model.Thrust_fin*throttle[iseg][0] #local_scale[iseg]
            mdot0[iseg,0] = self.c_model.mdot_fin*throttle[iseg][0] #local_scale[iseg]
            S[iseg] = self.c_model.sfc_fin
        

        
        results = Data()
        results.thrust_force_vector = F*self.number_of_engines
        results.vehicle_mass_rate   = mdot0*self.number_of_engines
        results.sfc                 = S
        #results.thrust_non_dim      = F_mdot0
        #results.offdesigndata = results_eval.offdesigndata        
        
        
        
        return results




    def engine_out(self,state):
        
        
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
        
        #return    



    __call__ = evaluate_thrust



