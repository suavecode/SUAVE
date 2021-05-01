## @ingroup Components-Energy-Converters
# Combustor.py
#
# Created:  Oct 2014, A. Variyar
# Modified: Jan 2016, T. MacDonald
#           Sep 2017, P. Goncalves
#           Jan 2018, W. Maier
#           Aug 2018, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE

import numpy as np

from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Methods.Propulsion.rayleigh import rayleigh
from SUAVE.Methods.Propulsion.fm_solver import fm_solver

# ----------------------------------------------------------------------
#  Combustor Component
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
class Combustor(Energy_Component):
    """This provides output values for a combustor
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
        self.fuel_data                       = SUAVE.Attributes.Propellants.Jet_A()
        self.alphac                          = 0.0
        self.turbine_inlet_temperature       = 1.0
        self.inputs.stagnation_temperature   = 1.0
        self.inputs.stagnation_pressure      = 1.0
        self.inputs.static_pressure          = 1.0
        self.inputs.mach_number              = 0.1
        self.outputs.stagnation_temperature  = 1.0
        self.outputs.stagnation_pressure     = 1.0
        self.outputs.static_pressure         = 1.0
        self.outputs.stagnation_enthalpy     = 1.0
        self.outputs.fuel_to_air_ratio       = 1.0
        self.fuel_data                       = Data()
        self.area_ratio                      = 1.0
        self.axial_fuel_velocity_ratio       = 0.0
        self.fuel_velocity_ratio             = 0.0
        self.burner_drag_coefficient         = 0.0
        self.absolute_sensible_enthalpy      = 0.0
        self.fuel_equivalency_ratio          = 1.0        
        self.inputs.nondim_mass_ratio        = 1.0 # allows fuel already burned to be added to the flow
    
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
          nondim_mass_ratio                   [-]

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
          fuel_data.specific_energy           [J/kg]
        """         
        # unpack the values

        # unpacking the values from conditions
        gamma  = conditions.freestream.isentropic_expansion_factor 
        Cp     = conditions.freestream.specific_heat_at_constant_pressure
        To     = conditions.freestream.temperature
        Tto    = conditions.freestream.stagnation_temperature
        
        # unpacking the values form inputs
        Tt_in    = self.inputs.stagnation_temperature
        Pt_in    = self.inputs.stagnation_pressure
        Tt4      = self.turbine_inlet_temperature
        pib      = self.pressure_ratio
        eta_b    = self.efficiency
        nondim_r = self.inputs.nondim_mass_ratio
        
        # unpacking values from self
        htf    = self.fuel_data.specific_energy
        ar     = self.area_ratio
        
        # compute pressure
        Pt_out = Pt_in*pib


        # method to compute combustor properties

        # method - computing the stagnation enthalpies from stagnation temperatures
        ht4     = Cp*Tt4*nondim_r
        ht_in   = Cp*Tt_in*nondim_r
        ho      = Cp*To
        
        # Using the Turbine exit temperature, the fuel properties and freestream temperature to compute the fuel to air ratio f
        f       = (ht4 - ht_in)/(eta_b*htf-ht4)

        # Computing the exit static and stagnation conditions
        ht_out  = Cp*Tt4
        
        # pack computed quantities into outputs
        self.outputs.stagnation_temperature  = Tt4
        self.outputs.stagnation_pressure     = Pt_out
        self.outputs.stagnation_enthalpy     = ht_out
        self.outputs.fuel_to_air_ratio       = f 
    
    def compute_rayleigh(self,conditions):
        """ This combutes the temperature and pressure change across the
        the combustor using Rayleigh Line flow; it checks for themal choking.

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
          fuel_data.specific_energy           [J/kg]
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
        htf    = self.fuel_data.specific_energy
        ar     = self.area_ratio
        
        # Rayleigh flow analysis, constant pressure burner
            
        # Initialize arrays
        M_out  = 1*Pt_in/Pt_in
        Ptr    = 1*Pt_in/Pt_in

        # Isentropic decceleration through divergent nozzle
        Mach   = fm_solver(ar,Mach[:,0],gamma[:,0])  
        
        # Determine max stagnation temperature to thermally choke flow                                     
        Tt4_ray = Tt_in*(1.+gamma*Mach*Mach)**2./((2.*(1.+gamma)*Mach*Mach)*(1.+(gamma-1.)/2.*Mach*Mach))

        # Rayleigh limitations define Tt4, taking max temperature before choking
        Tt4 = Tt4 * np.ones_like(Tt4_ray)
        Tt4[Tt4_ray <= Tt4] = Tt4_ray[Tt4_ray <= Tt4]
        
        #Rayleigh calculations
        M_out[:,0], Ptr[:,0] = rayleigh(gamma[:,0],Mach,Tt4[:,0]/Tt_in[:,0]) 
        Pt_out     = Ptr*Pt_in
            
        # method to compute combustor properties
        # method - computing the stagnation enthalpies from stagnation temperatures
        ht4     = Cp*Tt4
        ho      = Cp*To
        ht_in   = Cp*Tt_in
        
        # Using the Turbine exit temperature, the fuel properties and freestream temperature to compute the fuel to air ratio f
        f       = (ht4 - ht_in)/(eta_b*htf-ht4)

        # Computing the exit static and stagnation conditions
        ht_out  = Cp*Tt4   #May be double counting here.....no need (maybe)
        
        # pack computed quantities into outputs
        self.outputs.stagnation_temperature  = Tt4
        self.outputs.stagnation_pressure     = Pt_out
        self.outputs.stagnation_enthalpy     = ht_out
        self.outputs.fuel_to_air_ratio       = f    
        self.outputs.mach_number             = M_out
        
        
    def compute_supersonic_combustion(self,conditions): 
        """ This function computes the output values for supersonic  
        combustion (Scramjet).  This will be done using stream thrust 
        analysis. 
        
        Assumptions: 
        Constant Pressure Combustion      
        Flow is in axial direction at all times 
        Flow properities at exit are 1-Da averages 

        Source: 
        Heiser, William H., Pratt, D. T., Daley, D. H., and Unmeel, B. M., 
        "Hypersonic Airbreathing Propulsion", 1994 
        Chapter 4 - pgs. 175-180
        
        Inputs: 
        conditions.freestream. 
           isentropic_expansion_factor          [-] 
           specific_heat_at_constant_pressure   [J/(kg K)] 
           temperature                          [K] 
           stagnation_temperature               [K]
           universal_gas_constant               [J/(kg K)]  
        self.inputs. 
           stagnation_temperature               [K] 
           stagnation_pressure                  [Pa] 
           inlet_nozzle                         [-] 
  
        Outputs: 
        self.outputs. 
           stagnation_temperature               [K] 
           stagnation_pressure                  [Pa] 
           stagnation_enthalpy                  [J/kg] 
           fuel_to_air_ratio                    [-] 
           static_temperature                   [K] 
           static_pressure                      [Pa] 
           velocity                             [m/s] 
           mach_number                          [-]          
        
       Properties Used: 
          self.fuel_data.specific_energy       [J/kg] 
          self.efficiency                      [-]
          self.axial_fuel_velocity_ratio       [-] 
          self.fuel_velocity_ratio             [-] 
          self.burner_drag_coefficient         [-] 
          self.temperature_reference           [K] 
          self.absolute_sensible_enthalpy      [J/kg] 
          self.specific_heat_constant_pressure [J/(kg K)] 
          """ 
        # unpack the values 
  
        # unpacking the values from conditions 
        R      = conditions.freestream.gas_specific_constant 
        Tref   = conditions.freestream.temperature
        
        # unpacking the values from inputs 
        nozzle  = self.inputs.inlet_nozzle 
        Pt_in   = self.inputs.stagnation_pressure 
        Cp_c    = nozzle.specific_heat_at_constant_pressure
        
        # unpacking the values from self 
        htf     = self.fuel_data.specific_energy 
        eta_b   = self.efficiency 
        Vfx_V3  = self.axial_fuel_velocity_ratio 
        Vf_V3   = self.fuel_velocity_ratio 
        Cfb     = self.burner_drag_coefficient 
        hf      = self.absolute_sensible_enthalpy 
        phi     = self.fuel_equivalency_ratio
        
        # compute gamma and Cp over burner 
        Cpb     = Cp_c*1.45          # Estimate from Heiser and Pratt
        gamma_b = (Cpb/R)/(Cpb/R-1.)  
        
        # unpack nozzle input values 
        T_in = nozzle.static_temperature 
        V_in = nozzle.velocity 
        P_in = nozzle.static_pressure 
        
        # setting stoichiometric fuel-to-air  
        f_st = self.fuel_data.stoichiometric_fuel_to_air
        f    = phi*f_st
        
        # compute output velocity, mach and temperature 
        V_out  = V_in*(((1.+f*Vfx_V3)/(1.+f))-(Cfb/(2.*(1.+f)))) 
        T_out  = ((T_in/(1.+f))*(1.+(1./(Cpb*T_in ))*(eta_b*f*htf+f*hf+f*Cpb*Tref+(1.+f*Vf_V3*Vf_V3)*V_in*V_in/2.))) - V_out*V_out/(2.*Cpb) 
        M_out  = V_out/(np.sqrt(gamma_b*R*T_out)) 
        Tt_out = T_out*(1.+(gamma_b-1.)/2.)*M_out*M_out
        
        # compute the exity static and stagnation conditions 
        ht_out = Cpb*Tt_out 
        P_out  = P_in 
        Pt_out = Pt_in*((((gamma_b+1.)*(M_out**2.))/((gamma_b-1.)*M_out**2.+2.))**(gamma_b/(gamma_b-1.)))*((gamma_b+1.)/(2.*gamma_b*M_out**2.-(gamma_b-1.)))**(1./(gamma_b-1.))  
        
        # pack computed quantities into outputs    
        self.outputs.stagnation_temperature          = Tt_out  
        self.outputs.stagnation_pressure             = Pt_out        
        self.outputs.stagnation_enthalpy             = ht_out        
        self.outputs.fuel_to_air_ratio               = f        
        self.outputs.static_temperature              = T_out  
        self.outputs.static_pressure                 = P_out         
        self.outputs.velocity                        = V_out  
        self.outputs.mach_number                     = M_out 
        self.outputs.specific_heat_constant_pressure = Cpb
        self.outputs.isentropic_expansion_factor     = gamma_b
        
    __call__ = compute
    
