## @ingroup Components-Energy-Converters
# Supersonic_Nozzle.py
#
# Created:  May 2015, T. MacDonald
# Modified: Jan 2016, T. MacDonald
#           Jun 2017, P. Goncalves
#           Sep 2017, E. Botero
#           Jan 2018, W. Maier
#           Aug 2018, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

import SUAVE

from SUAVE.Core import Units

# package imports
import numpy as np

from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Methods.Propulsion.fm_id import fm_id
from SUAVE.Methods.Propulsion.nozzle_calculations import exit_Mach_shock, pressure_ratio_isentropic, pressure_ratio_shock_in_nozzle

# ----------------------------------------------------------------------
#  Expansion Nozzle Component
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Converters
class Supersonic_Nozzle(Energy_Component):
    """This is a nozzle component that allows for supersonic outflow.
    Calling this class calls the compute function.
    
    Assumptions:
    Pressure ratio and efficiency do not change with varying conditions.
    
    Source:
    https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/
    """
    
    def __defaults__(self):
        """ This sets the default values for the component to function.
        
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
        self.pressure_recovery               = 1.0
        self.inputs.stagnation_temperature   = 0.
        self.inputs.stagnation_pressure      = 0.
        self.outputs.stagnation_temperature  = 0.
        self.outputs.stagnation_pressure     = 0.
        self.outputs.stagnation_enthalpy     = 0.
        self.max_area_ratio                  = 1000.
        self.min_area_ratio                  = 0.    
    
    
    
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
          gas_specific_constant               [J/(kg K)]
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
          area_ratio                          [-]
                
        Properties Used:
        self.
          pressure_ratio                      [-]
          polytropic_efficiency               [-]
          pressure_recovery                   [-]
        """           
        
        #unpack the values
        
        #unpack from conditions
        gamma    = conditions.freestream.isentropic_expansion_factor
        Cp       = conditions.freestream.specific_heat_at_constant_pressure
        Po       = conditions.freestream.pressure
        Pto      = conditions.freestream.stagnation_pressure
        Tto      = conditions.freestream.stagnation_temperature
        R        = conditions.freestream.gas_specific_constant
        Mo       = conditions.freestream.mach_number
        
        #unpack from inputs
        Tt_in    = self.inputs.stagnation_temperature
        Pt_in    = self.inputs.stagnation_pressure
        
        #unpack from self
        pid      = self.pressure_ratio
        etapold  = self.polytropic_efficiency
        eta_rec =  self.pressure_recovery
        
        #Method for computing the nozzle properties
        
        #--Getting the output stagnation quantities
        Pt_out   = Pt_in*pid*eta_rec
        Tt_out   = Tt_in*(pid*eta_rec)**((gamma-1)/(gamma)*etapold)
        ht_out   = Cp*Tt_out
        
        
        #compute the output Mach number, static quantities and the output velocity
        Mach          = np.sqrt((((Pt_out/Po)**((gamma-1)/gamma))-1)*2/(gamma-1))
        
        #Remove check on mach numbers from expansion nozzle
        i_low         = Mach < 10.0
        
        #initializing the Pout array
        P_out         = 1.0 *Mach/Mach
        
        #Computing output pressure and Mach number for the case Mach <1.0
        P_out[i_low]  = Po[i_low]
        Mach[i_low]   = np.sqrt((((Pt_out[i_low]/Po[i_low])**((gamma[i_low]-1.)/gamma[i_low]))-1.)*2./(gamma[i_low]-1.))
        
        #Computing the output temperature,enthalpy, velocity and density
        T_out         = Tt_out/(1.+(gamma-1.)/2.*Mach*Mach)
        h_out         = Cp*T_out
        u_out         = np.sqrt(2.*(ht_out-h_out))
        rho_out       = P_out/(R*T_out)
        
        #Computing the freestream to nozzle area ratio (mainly from thrust computation)
        area_ratio    = (fm_id(Mo,gamma)/fm_id(Mach,gamma)*(1/(Pt_out/Pto))*(np.sqrt(Tt_out/Tto)))
        
        #pack computed quantities into outputs
        self.outputs.stagnation_temperature  = Tt_out
        self.outputs.stagnation_pressure     = Pt_out
        self.outputs.stagnation_enthalpy     = ht_out
        self.outputs.mach_number             = Mach
        self.outputs.static_temperature      = T_out
        self.outputs.density                 = rho_out
        self.outputs.static_enthalpy         = h_out
        self.outputs.velocity                = u_out
        self.outputs.static_pressure         = P_out
        self.outputs.area_ratio              = area_ratio
        
    def compute_limited_geometry(self,conditions):
        
        """This is a variable geometry nozzle component that allows 
        for supersonic outflow. all possible nozzle conditions, including 
        overexpansion and underexpansion.
        
        Assumptions:
        Constant polytropic efficiency and pressure ratio
        
        Source:
        https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/
        https://web.stanford.edu/~cantwell/AA210A_Course_Material/AA210A_Course_Notes/
        
        Inputs:
        conditions.freestream.
          isentropic_expansion_factor         [-]
          specific_heat_at_constant_pressure  [J/(kg K)]
          pressure                            [Pa]
          stagnation_pressure                 [Pa]
          stagnation_temperature              [K]
          gas_specific_constant               [J/(kg K)] 
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
        R        = conditions.freestream.gas_specific_constant
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
        Tt_out   = Tt_in*pid**((gamma-1.)/(gamma)*etapold)
        ht_out   = Cp*Tt_out
  
        # Method for computing the nozzle properties
        #-- Initial estimate for exit area
        area_ratio = (max_area_ratio + min_area_ratio)/2.
        
        #-- Compute limits of each possible flow condition       
        subsonic_pressure_ratio     = pressure_ratio_isentropic(area_ratio, gamma, True)
        nozzle_shock_pressure_ratio = pressure_ratio_shock_in_nozzle(area_ratio, gamma)
        supersonic_pressure_ratio   = pressure_ratio_isentropic(area_ratio, gamma, False) 
        supersonic_max_Area         = pressure_ratio_isentropic(max_area_ratio, gamma, False)
        supersonic_min_Area         = pressure_ratio_isentropic(min_area_ratio, gamma, False)

        #-- Compute the output Mach number guess with freestream pressure
        #-- Initializing arrays
        P_out       = np.ones_like(Pt_out)
        A_ratio     = area_ratio*np.ones_like(Pt_out)
        M_out       = np.ones_like(Pt_out)

        # Establishing a correspondence between real pressure ratio and limits of each flow condition
        
        # Determine if flow is within subsonic/sonic range
        i_sub               = Po/Pt_out >= subsonic_pressure_ratio 
        
        # Detemine if there is a shock in nozzle
        i2                  = Po/Pt_out < subsonic_pressure_ratio
        i3                  = Po/Pt_out >= nozzle_shock_pressure_ratio
        i_shock             = np.logical_and(i2,i3)      
        
        # Determine if flow is overexpanded
        i4                  = Po/Pt_out < nozzle_shock_pressure_ratio
        i5                  = Po/Pt_out > supersonic_min_Area
        i_over              = np.logical_and(i4,i5)  
        
        # Determine if flow is supersonic
        i6                  = Po/Pt_out <= supersonic_min_Area
        i7                  = Po/Pt_out >= supersonic_max_Area
        i_sup               = np.logical_and(i6,i7) 
        
        # Determine if flow is underexpanded
        i_und               = Po/Pt_out < supersonic_max_Area
        
        #-- Subsonic and sonic flow
        P_out[i_sub]        = Po[i_sub]
        M_out[i_sub]        = np.sqrt((((Pt_out[i_sub]/P_out[i_sub])**((gamma[i_sub]-1.)/gamma[i_sub]))-1.)*2./(gamma[i_sub]-1.))
        A_ratio[i_sub]      = 1./fm_id(M_out[i_sub],gamma[i_sub])
        
        #-- Shock inside nozzle
        P_out[i_shock]      = Po[i_shock]
        M_out[i_shock]      = np.sqrt((((Pt_out[i_shock]/P_out[i_shock])**((gamma[i_shock]-1.)/gamma[i_shock]))-1.)*2./(gamma[i_shock]-1.))   
        A_ratio[i_shock]    = 1./fm_id(M_out[i_shock],gamma[i_shock])
        
        #-- Overexpanded flow
        P_out[i_over]       = supersonic_min_Area[i_over]*Pt_out[i_over] 
        M_out[i_over]       = np.sqrt((((Pt_out[i_over]/P_out[i_over])**((gamma[i_over]-1.)/gamma[i_over]))-1.)*2./(gamma[i_over]-1.))
        A_ratio[i_over]     = 1./fm_id(M_out[i_over],gamma[i_over])
        
        #-- Isentropic supersonic flow, with variable area adjustments
        P_out[i_sup]        = Po[i_sup]
        M_out[i_sup]        = np.sqrt((((Pt_out[i_sup]/P_out[i_sup])**((gamma[i_sup]-1.)/gamma[i_sup]))-1.)*2./(gamma[i_sup]-1.))    
        A_ratio[i_sup]      = 1./fm_id(M_out[i_sup],gamma[i_sup])
        
        #-- Underexpanded flow
        P_out[i_und]        = supersonic_max_Area[i_und]*Pt_out[i_und] 
        M_out[i_und]        = np.sqrt((((Pt_out[i_und]/P_out[i_und])**((gamma[i_und]-1.)/gamma[i_und]))-1.)*2./(gamma[i_und]-1.))
        A_ratio[i_und]      = 1./fm_id(M_out[i_und],gamma[i_und])
        
        #-- Calculate other flow properties
        T_out   = Tt_out/(1.+(gamma-1.)/2.*M_out*M_out)
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
        
    def compute_scramjet(self,conditions): 
        """This computes exit conditions of a scramjet. 
        
        Assumptions: 
        Fixed output Cp and Gamma 
        
        Source: 
        Heiser, William H., Pratt, D. T., Daley, D. H., and Unmeel, B. M.,  
        "Hypersonic Airbreathing Propulsion", 1994  
        Chapter 4 - pgs. 175-180
        
        Inputs:  
        conditions.freestream.  
           isentropic_expansion_factor         [-]  
           specific_heat_at_constant_pressure  [J/(kg K)]  
           pressure                            [Pa]  
           stagnation_pressure                 [Pa]  
           stagnation_temperature              [K]  
           gas_specific_constant               [J/(kg K)]  
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
           area_ratio                          [-]  
        
        Properties Used:  
        self.  
           polytropic_efficiency               [-]  
           pressure_expansion_ratio            [-]                    
        """  
        
        # unpack values  
        
        # unpack from conditions 
        Po         = conditions.freestream.pressure   
        Vo         = conditions.freestream.velocity 
        To         = conditions.freestream.temperature 
        R          = conditions.freestream.gas_specific_constant 
        
        # unpack from inputs 
        Tt_in      = self.inputs.stagnation_temperature 
        Pt_in      = self.inputs.stagnation_pressure 
        T_in       = self.inputs.static_temperature 
        P_in       = self.inputs.static_pressure 
        u_in       = self.inputs.velocity 
        f          = self.inputs.fuel_to_air_ratio   
        Cpe        = self.inputs.specific_heat_constant_pressure 
        gamma      = self.inputs.isentropic_expansion_factor                 

        # unpack from self 
        eta        = self.polytropic_efficiency 
        p10_p0     = self.pressure_expansion_ratio
        
        # compute output properties 
        P_out      = Po*p10_p0
        T_out      = T_in*(1.-eta*(1.-((P_out/Po)*(Po/P_in))**(R/Cpe))) 
        u_out      = np.sqrt(u_in*u_in+2.*Cpe*(T_in-T_out)) 
        A_ratio    = (1.+f)*(Po/P_out)*(T_out/To)*(Vo/u_out) 
        M_out      = u_out/np.sqrt(gamma*R*T_out) 
        
        #pack computed quantities into outputs          
        self.outputs.stagnation_temperature  = Tt_in  
        self.outputs.stagnation_pressure     = Pt_in        
        self.outputs.temperature             = T_out         
        self.outputs.pressure                = P_out      
        self.outputs.velocity                = u_out   
        self.outputs.static_pressure         = P_out     
        self.outputs.area_ratio              = A_ratio  
        self.outputs.mach_number             = M_out        
        
    __call__ = compute