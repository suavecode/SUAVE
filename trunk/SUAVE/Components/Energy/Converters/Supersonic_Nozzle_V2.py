# Supersonic_Nozzle_V2.py
#
# Created:  May 2017, P. Goncalves

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

import SUAVE

from SUAVE.Core import Units


# package imports
import numpy as np
from scipy.optimize import fsolve



from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Methods.Propulsion.nozzle_calculations import exit_Mach_shock, pressure_ratio_isentropic, pressure_ratio_shock_in_nozzle
from SUAVE.Methods.Propulsion.fm_id import fm_id



# ----------------------------------------------------------------------
#  Expansion Nozzle Component
# ----------------------------------------------------------------------
    
    
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
        """This sets the default values for the component to function.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """
        #set the defaults
        self.tag = 'Nozzle'
        self.polytropic_efficiency           = 1.0
        self.pressure_ratio                  = 1.0
        self.inputs.stagnation_temperature   = 1.0
        self.inputs.stagnation_pressure      = 0.
        self.outputs.stagnation_temperature  = 1.0
        self.outputs.stagnation_pressure     = 0.
        self.outputs.stagnation_enthalpy     = 0.
        self.max_area_ratio                  = 2.
        self.min_area_ratio                  = 1.8
    
    
    
    def compute(self,conditions):
        
        """This computes the output values from the input values according to
        equations from the source.
        
        Assumptions:
        Constant polytropic efficiency and pressure ratio
        
        Source:
        https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/
        
        Inputs:
        conditions.freestream.
          isentropic_expansion_factor         [-]
          specific_heat_at_constant_pressure  [J/(kg K)]
          pressure                            [Pa]
          stagnation_pressure                 [Pa]
          stagnation_temperature              [K]
          universal_gas_constant              [J/(kg K)] (this is misnamed - actually refers to the gas specific constant)
          mach_number                         [-]
        self.inputs.
          stagnation_temperature              [K]
          stagnation_pressure                 [Pa]
                   
        Outputs:
        self.outputs.
          stagnation_temperature              [K]  
          stagnation_pressure                 [Pa]
          stagnation_enthalpy                 [J/kg]
          mach_number                         [-]
          static_temperature                  [K]
          static_enthalpy                     [J/kg]
          velocity                            [m/s]
          static_pressure                     [Pa]
                
        Properties Used:
        self.
          pressure_ratio                      [-]
          polytropic_efficiency               [-]
          max_area_ratio                      [-]
          min_area_ratio                      [-]
        """           
        
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
        pid             = self.pressure_ratio
        etapold         = self.polytropic_efficiency
        max_area_ratio  = self.max_area_ratio
        min_area_ratio  = self.min_area_ratio
        
        
        # Method for computing the nozzle properties
        
        #--Getting the output stagnation quantities
        Pt_out   = Pt_in*pid
        Tt_out   = Tt_in*pid**((gamma-1)/(gamma)*etapold)
        ht_out   = Cp*Tt_out
  
        # Method for computing the nozzle properties
  
        #-- Initial estimate for exit area
        area_ratio = (max_area_ratio + min_area_ratio) / 2
        
        #-- Compute limits of each possible flow condition       
        subsonic_pressure_ratio     = pressure_ratio_isentropic(area_ratio, gamma, True)
        nozzle_shock_pressure_ratio = pressure_ratio_shock_in_nozzle(area_ratio, gamma)
        supersonic_pressure_ratio   = pressure_ratio_isentropic(area_ratio, gamma, False) 
        supersonic_max_Area         = pressure_ratio_isentropic(max_area_ratio, gamma, False)
        supersonic_min_Area         = pressure_ratio_isentropic(min_area_ratio, gamma, False)

        #-- Compute the output Mach number guess with freestream pressure

        #-- Initializing arrays
        P_out       = 1.0 *Pt_out/Pt_out
        A_ratio     = area_ratio*Pt_out/Pt_out
        M_out       = 1.0 *Pt_out/Pt_out


        # Establishing a correspondence between real pressure ratio and limits of each flow condition
        i_sub               = Po/Pt_out >= subsonic_pressure_ratio     
        i2                  = Po/Pt_out < subsonic_pressure_ratio
        i3                  = Po/Pt_out >= nozzle_shock_pressure_ratio    
        i_shock             = np.logical_and(i2,i3)      
        i4                  = Po/Pt_out < nozzle_shock_pressure_ratio
        i5                  = Po/Pt_out > supersonic_min_Area
        i_over              = np.logical_and(i4,i5)            
        i6                  = Po/Pt_out <= supersonic_min_Area
        i7                  = Po/Pt_out >= supersonic_max_Area
        i_sup               = np.logical_and(i6,i7)            
        i_und               = Po/Pt_out < supersonic_max_Area
        
        #-- Subsonic and sonic flow
        P_out[i_sub]        = Po[i_sub]
        M_out[i_sub]        = np.sqrt((((Pt_out[i_sub]/P_out[i_sub])**((gamma-1)/gamma))-1)*2/(gamma-1))
        A_ratio[i_sub]      = fm_id(M_out[i_sub])
        
        #-- Shock inside nozzle
        P_out[i_shock]      = Po[i_shock]
        M_out[i_shock]      = np.sqrt((((Pt_out[i_shock]/P_out[i_shock])**((gamma-1)/gamma))-1)*2/(gamma-1))
        #M_out[i_shock]      = exit_Mach_shock(A_ratio[i_shock], gamma, Pt_out[i_shock], Po[i_shock])    
        A_ratio[i_shock]    = area_ratio
        #-- Overexpanded flow
        P_out[i_over]       = supersonic_pressure_ratio*Pt_out[i_over] 
        M_out[i_over]       = np.sqrt((((Pt_out[i_over]/P_out[i_over])**((gamma-1)/gamma))-1)*2/(gamma-1))
        A_ratio[i_over]     = fm_id(M_out[i_over])
        #-- Isentropic supersonic flow, with variable area adjustments
        P_out[i_sup]        = Po[i_sup]
        M_out[i_sup]        = np.sqrt((((Pt_out[i_sup]/P_out[i_sup])**((gamma-1)/gamma))-1)*2/(gamma-1))    
        A_ratio[i_sup]      = fm_id(M_out[i_sup])
        #-- Underexpanded flow
        P_out[i_und]        = supersonic_pressure_ratio*Pt_out[i_und] 
        M_out[i_und]        = np.sqrt((((Pt_out[i_und]/P_out[i_und])**((gamma-1)/gamma))-1)*2/(gamma-1))
        A_ratio[i_und]      = fm_id(M_out[i_und])
        
       #-- Calculate other flow properties
        T_out   = Tt_out/(1+(gamma-1)/2*M_out**2)
        h_out   = Cp*T_out
        u_out   = M_out*np.sqrt(gamma*R*T_out)
        rho_out = P_out/(R*T_out)
        
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
        self.outputs.area_ratio              = A_ratio

    __call__ = compute
    
   
