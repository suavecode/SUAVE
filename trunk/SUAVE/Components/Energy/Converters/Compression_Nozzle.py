## @ingroup Components-Energy-Converters
# Compression_Nozzle.py
#
# Created:  Jul 2014, A. Variyar
# Modified: Jan 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE

# python imports
from warnings import warn

# package imports
import numpy as np

from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Compression Nozzle Component
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
class Compression_Nozzle(Energy_Component):
    """This is a nozzle component intended for use in compression.
    Calling this class calls the compute function.
    
    Assumptions:
    Pressure ratio and efficiency do not change with varying conditions.
    Subsonic or choked output.
    
    Source:
    https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/
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
        #setting the default values 
        self.tag = 'Nozzle'
        self.polytropic_efficiency           = 1.0
        self.pressure_ratio                  = 1.0
        self.inputs.stagnation_temperature   = 0.
        self.inputs.stagnation_pressure      = 0.
        self.outputs.stagnation_temperature  = 0.
        self.outputs.stagnation_pressure     = 0.
        self.outputs.stagnation_enthalpy     = 0.
    

    def compute(self,conditions):
        """ This computes the output values from the input values according to
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
          universal_gas_constant              [J/(kg K)] (this is misnamed - actually refers to the gas specific constant)
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

        Properties Used:
        self.
          pressure_ratio                      [-]
          polytropic_efficiency               [-]
        """           
        #unpack the values
        
        #unpack from conditions
        gamma   = conditions.freestream.isentropic_expansion_factor
        Cp      = conditions.freestream.specific_heat_at_constant_pressure
        Po      = conditions.freestream.pressure
        Mo      = conditions.freestream.mach_number
        R       = conditions.freestream.universal_gas_constant
        
        #unpack from inpust
        Tt_in   = self.inputs.stagnation_temperature
        Pt_in   = self.inputs.stagnation_pressure
        
        #unpack from self
        pid     =  self.pressure_ratio
        etapold =  self.polytropic_efficiency
        
        #Method to compute the output variables
        
        #--Getting the output stagnation quantities
        Tt_out  = Tt_in*pid**((gamma-1)/(gamma*etapold))
        ht_out  = Cp*Tt_out
        
        #initializing the arrays
        Mach     = 1.0 * Pt_in/Pt_in
        T_out    = 1.0 * Pt_in/Pt_in
        Mo       = Mo * Pt_in/Pt_in
        Pt_out   = 1.0 * Pt_in/Pt_in
        P_out    = 1.0 * Pt_in/Pt_in
        
        i_low  = Mo <= 1.0
        i_high = Mo > 1.0
        
        
        #compute the output Mach number, static quantities and the output velocity
        
        #-- Inlet Mach <= 1.0, isentropic relations
        Mach[i_low]    = np.sqrt( (((Pt_out[i_low]/Po[i_low])**((gamma-1.)/gamma))-1.) *2./(gamma-1.) )
        T_out[i_low]   = Tt_out[i_low]/(1+(gamma-1)/2*Mach[i_low]*Mach[i_low])
        Pt_out[i_low]  = Pt_in[i_low]*pid
        
        #-- Inlet Mach > 1.0, normal shock
        Mach[i_high]   = np.sqrt((1+(gamma-1)/2*Mo[i_high]**2)/(gamma*Mo[i_high]**2-(gamma-1)/2))
        T_out[i_high]  = Tt_out[i_high]/(1+(gamma-1)/2*Mach[i_high]*Mach[i_high])
        Pt_out[i_high] = Pt_in[i_high]*((((gamma+1)*(Mo[i_high]**2))/((gamma-1)*Mo[i_high]**2+2))**(gamma/(gamma-1)))*((gamma+1)/(2*gamma*Mo[i_high]**2-(gamma-1)))**(1/(gamma-1))
        P_out[i_high]  = Pt_out[i_high]*(1+(2*gamma/(gamma+1))*(Mach[i_high]**2-1))
        
        #-- Compute exit velocity and enthalpy
        h_out   = Cp*T_out
        u_out   = np.sqrt(2*(ht_out-h_out))
          
        #pack computed quantities into outputs
        self.outputs.stagnation_temperature  = Tt_out
        self.outputs.stagnation_pressure     = Pt_out
        self.outputs.stagnation_enthalpy     = ht_out
        self.outputs.mach_number             = Mach
        self.outputs.static_temperature      = T_out
        self.outputs.static_enthalpy         = h_out
        self.outputs.velocity                = u_out
    

    __call__ = compute