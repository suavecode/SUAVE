## @ingroup Components-Energy-Converters
# Combustor.py
#
# Created:  Oct 2014, A. Variyar
# Modified: Jan 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE

import numpy as np
from scipy.optimize import fsolve

from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Methods.Propulsion.rayleigh import rayleigh
from SUAVE.Methods.Propulsion.fm_solver import fm_solver

# ----------------------------------------------------------------------
#  Combustor Component
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
class Combustor(Energy_Component):
    """This is provides output values for a combustor
    Calling this class calls the compute function.
    
    Assumptions:
    None
    
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
        
        self.tag = 'Combustor'
        
        #-----setting the default values for the different components
        self.fuel_data                      = SUAVE.Attributes.Propellants.Jet_A()
        self.alphac                         = 0.0
        self.turbine_inlet_temperature      = 1.0
        self.inputs.stagnation_temperature  = 1.0
        self.inputs.stagnation_pressure     = 1.0
        self.inputs.static_pressure         = 1.0
        self.inputs.mach_number             = 0.1
        self.outputs.stagnation_temperature = 1.0
        self.outputs.stagnation_pressure    = 1.0
        self.outputs.static_pressure        = 1.0
        self.outputs.stagnation_enthalpy    = 1.0
        self.outputs.fuel_to_air_ratio      = 1.0
        self.fuel_data                      = Data()
        self.rayleigh_analyses              = False
        self.area_ratio                     = 1.9
    
    def compute(self,conditions):
        """ This computes the output values from the input values according to
        equations from the source.

        Assumptions:
        Constant efficiency and pressure ratio

        Source:
        https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/

        Inputs:
        conditions.freestream.
          isentropic_expansion_factor         [-]
          specific_heat_at_constant_pressure  [J/(kg K)]
          temperature                         [K]
          stagnation_temperature              [K]
        self.inputs.
          stagnation_temperature              [K]
          stagnation_pressure                 [Pa]

        Outputs:
        self.outputs.
          stagnation_temperature              [K]  
          stagnation_pressure                 [Pa]
          stagnation_enthalpy                 [J/kg]
          fuel_to_air_ratio                   [-]

        Properties Used:
        self.
          turbine_inlet_temperature           [K]
          pressure_ratio                      [-]
          efficiency                          [-]
          area_ratio                          [-]
        """         
        # unpack the values

        # unpacking the values from conditions
        gamma  = conditions.freestream.isentropic_expansion_factor 
        Cp     = conditions.freestream.specific_heat_at_constant_pressure
        To     = conditions.freestream.temperature
        Tto    = conditions.freestream.stagnation_temperature
        
        # unpacking the values form inputs
        Tt_in  = self.inputs.stagnation_temperature
        Pt_in  = self.inputs.stagnation_pressure
        Mach   = self.inputs.mach_number
        Tt4    = self.turbine_inlet_temperature
        pib    = self.pressure_ratio
        eta_b  = self.efficiency
        
        # unpacking values from self
        htf             = self.fuel_data.specific_energy
        ray_analysis    = self.rayleigh_analyses
        ar              = self.area_ratio
        
        # Rayleigh flow analysis, constant pressure burner
        if ray_analysis:
            
            # Initialize arrays
            M_out  = 1*Pt_in/Pt_in
            Ptr   = 1*Pt_in/Pt_in

            # Make i_rayleigh the size of output arrays
            i_rayleigh = Pt_in < 2*Pt_in
            
            # Isentropic decceleration through divergent nozzle
            Mach[i_rayleigh]    = fm_solver(ar,Mach[i_rayleigh],gamma)  
            
            # Determine max stagnation temperature to thermally choke flow                                     
            Tt4_ray = Tt_in*(1+gamma*Mach**2)**2/((2*(1+gamma)*Mach**2)*(1+(gamma-1)/2*Mach**2)) 

            # Checking if Tt4 is limited by Rayleigh
            i_low = Tt4_ray <= Tt4
            i_high = Tt4_ray > Tt4
            
            # Choose Tt4 for fuel calculations
            
            # --Material limitations define Tt4
            Tt4 = Tt4*Tt4_ray/Tt4_ray
            
            # --Rayleigh limitations define Tt4
            Tt4[i_low] = Tt4_ray[i_low]
            
            #Rayleigh calculations
            M_out[i_rayleigh], Ptr[i_rayleigh] = rayleigh(gamma,Mach[i_rayleigh],Tt4[i_rayleigh]/Tt_in[i_rayleigh])
            Pt_out     = Ptr*Pt_in
            
        else:
            Pt_out      = Pt_in*pib


        # method to compute combustor properties

        # method - computing the stagnation enthalpies from stagnation temperatures
        ht4     = Cp*Tt4
        ho      = Cp*To
        ht_in   = Cp*Tt_in
        
        # Using the Turbine exit temperature, the fuel properties and freestream temperature to compute the fuel to air ratio f
        f       = (ht4 - ht_in)/(eta_b*htf-ht4)

        # Computing the exit static and stagnation conditions
        ht_out  = Cp*Tt4
        
        # pack computed quantities into outputs
        self.outputs.stagnation_temperature  = Tt4
        self.outputs.stagnation_pressure     = Pt_out
        self.outputs.stagnation_enthalpy     = ht_out
        self.outputs.fuel_to_air_ratio       = f 
    
    
    
    __call__ = compute
    
