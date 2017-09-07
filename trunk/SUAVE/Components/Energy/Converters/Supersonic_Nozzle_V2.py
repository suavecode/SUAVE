# Supersonic_Nozzle_V2.py
#
# Created:  May 2017, P. Goncalves

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

import SUAVE

from SUAVE.Core import Units
from scipy.optimize import fsolve

# package imports
import numpy as np


from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Methods.Propulsion.fm_id import fm_id


# ----------------------------------------------------------------------
#  Expansion Nozzle Component
# ----------------------------------------------------------------------

def exit_Mach_shock(area_ratio, gamma, Pt_out, P0):
        func = lambda Me : (Pt_out/P0)*(1/area_ratio)-(((gamma+1)/2)**((gamma+1)/(2*(gamma-1))))*Me*((1+(gamma-1)/2*Me**2)**0.5)
        Me_initial_guess = 0.01
        Me_solution = fsolve(func,Me_initial_guess)
        
        return Me_solution
        
        
def mach_area(area_ratio, gamma, subsonic):
        func = lambda Me : area_ratio**2 - ((1/Me)**2)*(((2/(gamma+1))*(1+((gamma-1)/2)*Me**2))**((gamma+1)/((gamma-1))))
        if subsonic:
            Me_initial_guess = 0.01
        else:
            Me_initial_guess = 2.0         
        
        Me_solution = fsolve(func,Me_initial_guess)

        return Me_solution

    
def normal_shock(M1, gamma):  
    M2 = np.sqrt((((gamma-1)*M1**2)+2)/(2*gamma*M1**2-(gamma-1)))
    
    return M2
    
def pressure_ratio_isentropic(area_ratio, gamma, subsonic):
    #yields pressure ratio for isentropic conditions given area ratio
    Me = mach_area(area_ratio,gamma, subsonic)
    
    pr_isentropic = (1+((gamma-1)/2)*Me**2)**(-gamma/(gamma-1))
    
    return pr_isentropic

def pressure_ratio_shock_in_nozzle(area_ratio, gamma):
    #yields maximium pressure ratio where shock takes place inside the nozzle, given area ratio
    Me = mach_area(area_ratio, gamma, False)
    M2 = normal_shock(Me, gamma)
    
    pr_shock_in_nozzle = ((area_ratio)*(((gamma+1)/2)**((gamma+1)/(2*(gamma-1))))*M2*((1+((gamma-1)/2)*M2**2)**0.5))**(-1)
    return pr_shock_in_nozzle
    
    
class Supersonic_Nozzle_V2(Energy_Component):
    """This is a variable geometry nozzle component that allows 
    for supersonic outflow. all possible nozzle conditions, including 
    overexpansion and underexpansion.
    Calling this class calls the compute function. 
    T
    
    Assumptions:
    Pressure ratio and efficiency do not change with varying conditions.
    
    Source:
    https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/
    
    Cantwell, Fundamentals of Compressible Flow, Chapter 10
    """
    
    def __defaults__(self):
        
        #set the defaults
        self.tag = 'Nozzle'
        self.polytropic_efficiency           = 1.0
        self.pressure_ratio                  = 1.0
        self.inputs.stagnation_temperature   = 1.0
        self.inputs.stagnation_pressure      = 0.
        self.outputs.stagnation_temperature  = 1.0
        self.outputs.stagnation_pressure     = 0.
        self.outputs.stagnation_enthalpy     = 0.
    
    
    
    def compute(self,conditions):
        
        #unpack the values
        
        #unpack from conditions
        gamma    = conditions.freestream.isentropic_expansion_factor
        Cp       = conditions.freestream.specific_heat_at_constant_pressure
        Po       = conditions.freestream.pressure
        Pto      = conditions.freestream.stagnation_pressure
        Tto      = conditions.freestream.stagnation_temperature
        R        = conditions.freestream.universal_gas_constant
        Mo       = conditions.freestream.mach_number
        To       = conditions.freestream.temperature
        
        #unpack from inputs
        Tt_in    = self.inputs.stagnation_temperature
        Pt_in    = self.inputs.stagnation_pressure
        
        
        #unpack from self
        pid           = self.pressure_ratio
        etapold       = self.polytropic_efficiency
        #area_ratio    = self.area_ratio
        
        
        #Method for computing the nozzle properties
        
        #--Getting the output stagnation quantities
        Pt_out   = Pt_in*pid
        Tt_out   = Tt_in*pid**((gamma-1)/(gamma)*etapold)
        ht_out   = Cp*Tt_out

        
        # Method for computing the nozzle properties
        
        #-- Compute the output Mach number guess with freestream pressure
        M_out          = np.sqrt((((Pt_out/Po)**((gamma-1)/gamma))-1)*2/(gamma-1))

        #-- Initializing arrays
        P_out         = 1.0 *M_out/M_out
        Pt_out        = Pt_out * M_out/M_out
        area_ratio = 1.5
        max_area_ratio = 1.70
        min_area_ratio = 1.4
        
#        
        #-- Compute limits of each possible regime 
        
        subsonic_pressure_ratio = pressure_ratio_isentropic(area_ratio, gamma, True)
        nozzle_shock_pressure_ratio = pressure_ratio_shock_in_nozzle(area_ratio, gamma)
        supersonic_pressure_ratio = pressure_ratio_isentropic(area_ratio, gamma, False)
        
        supersonic_max_Area = pressure_ratio_isentropic(max_area_ratio, gamma, False)
        supersonic_min_Area = pressure_ratio_isentropic(min_area_ratio, gamma, False)

        choked_state = True
        #-- Subsonic regime
        if np.any(Po/Pt_out > subsonic_pressure_ratio):
            #print 'Subsonic regime'
            P_out = Po
            M_out = np.sqrt((((Pt_out/P_out)**((gamma-1)/gamma))-1)*2/(gamma-1))
            choked_state = False
            
            
        #-- Sonic regime
        elif np.any(Po/Pt_out == subsonic_pressure_ratio):
            #print 'Sonic throat'
            P_out = Po
            M_out = np.sqrt((((Pt_out/P_out)**((gamma-1)/gamma))-1)*2/(gamma-1))
            
        #-- Supersonic nozzle AND shock inside nozzle
        elif np.any(Po/Pt_out >= nozzle_shock_pressure_ratio):
            #print 'Shock inside nozzle'
            P_out = Po
            M_out = exit_Mach_shock(area_ratio, gamma, Pt_out, Po[0])
            
        #-- Supersonic nozzle AND shock outside nozzle
        elif np.any(Po/Pt_out > supersonic_pressure_ratio):
            #print 'Overexpanded flow'
            if np.any(Po/Pt_out <= supersonic_min_Area):
                #print 'Variable geometry, reduce area to match P0'
                P_out = Po
                M_out = np.sqrt((((Pt_out/P_out)**((gamma-1)/gamma))-1)*2/(gamma-1))
                area_ratio = np.sqrt((1/M_out)**2)*(((2/(gamma+1))*(1+((gamma-1)/2)*M_out**2))**((gamma+1)/((gamma-1))))
                
            else:       
                P_out = supersonic_pressure_ratio * Pt_out
                M_out = np.sqrt((((Pt_out/P_out)**((gamma-1)/gamma))-1)*2/(gamma-1))
            
        #-- Supersonic nozzle
        elif np.any(Po/Pt_out == supersonic_pressure_ratio):
            #print 'Isentropic supersonic flow'
            P_out = Po
            M_out = np.sqrt((((Pt_out/P_out)**((gamma-1)/gamma))-1)*2/(gamma-1))
            
            
        #-- Supersonic nozzle AND expansion outside nozzle
        elif np.any(Po/Pt_out < supersonic_pressure_ratio):
            #print 'Underexpanded flow'
            
            if np.any(Po/Pt_out >= supersonic_max_Area):
                #print 'Variable geometry, increase area to match Po'
                P_out = Po
                M_out = np.sqrt((((Pt_out/P_out)**((gamma-1)/gamma))-1)*2/(gamma-1))
                area_ratio = fm_id(M_out)
            else:
                P_out = supersonic_pressure_ratio * Pt_out
                M_out = np.sqrt((((Pt_out/P_out)**((gamma-1)/gamma))-1)*2/(gamma-1))
                area_ratio = fm_id(M_out)

       #-- Calculate other flow properties

        T_out = Tt_out/(1+(gamma-1)/2*M_out**2)
        h_out         = Cp*T_out
        u_out = M_out*np.sqrt(gamma*R*T_out)
        rho_out       = P_out/(R*T_out)


        
        #pack computed quantities into outputs
        self.outputs.stagnation_temperature  = Tt_out
        self.outputs.stagnation_pressure     = Pt_out
        self.outputs.stagnation_enthalpy     = ht_out
        self.outputs.mach_number             = M_out
        self.outputs.static_temperature      = T_out
        self.outputs.rho                     = rho_out
        self.outputs.static_enthalpy         = h_out
        self.outputs.velocity                = u_out
        self.outputs.static_pressure         = P_out
        self.outputs.area_ratio              = area_ratio

    __call__ = compute
    
   
