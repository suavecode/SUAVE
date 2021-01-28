## @ingroup Components-Energy-Converters
# Compression_Nozzle.py
#
# Created:  Jul 2014, A. Variyar
# Modified: Jan 2016, T. MacDonald
#           Sep 2017, P. Goncalves
#           Jan 2018, W. Maier
#           Aug 2018, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE

# python imports
from warnings import warn

# package imports
import numpy as np

from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Methods.Propulsion.shock_train import shock_train

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
        self.pressure_recovery               = 1.0
        self.compressibility_effects         = False
        self.inputs.stagnation_temperature   = 0.0
        self.inputs.stagnation_pressure      = 0.0
        self.outputs.stagnation_temperature  = 0.0
        self.outputs.stagnation_pressure     = 0.0
        self.outputs.stagnation_enthalpy     = 0.0
        self.compression_levels              = 0.0
        self.theta                           = 0.0
       
    def compute(self,conditions):
        """ This computes the output values from the input values according to
        equations from the source.

        Assumptions:
        Constant polytropic efficiency and pressure ratio
        Adiabatic

        Source:
        https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/

        Inputs:
        conditions.freestream.
          isentropic_expansion_factor         [-]
          specific_heat_at_constant_pressure  [J/(kg K)]
          pressure                            [Pa]
          gas_specific_constant               [J/(kg K)]
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
          pressure_recovery                   [-]
        """

        #unpack from conditions
        gamma   = conditions.freestream.isentropic_expansion_factor
        Cp      = conditions.freestream.specific_heat_at_constant_pressure
        Po      = conditions.freestream.pressure
        Mo      = conditions.freestream.mach_number
        R       = conditions.freestream.gas_specific_constant

        #unpack from inpust
        Tt_in   = self.inputs.stagnation_temperature
        Pt_in   = self.inputs.stagnation_pressure

        #unpack from self
        pid     =  self.pressure_ratio
        etapold =  self.polytropic_efficiency
        eta_rec =  self.pressure_recovery
        compressibility_effects =  self.compressibility_effects

        #Method to compute the output variables

        #--Getting the output stagnation quantities
        Pt_out  = Pt_in*pid*eta_rec
        Tt_out  = Tt_in*(pid*eta_rec)**((gamma-1)/(gamma*etapold))
        ht_out  = Cp*Tt_out

        if compressibility_effects :

            # Checking from Mach numbers below, above 1.0
            i_low  = Mo <= 1.0
            i_high = Mo > 1.0

            #initializing the arrays
            Mach     = np.ones_like(Pt_in)
            T_out    = np.ones_like(Pt_in)
            Mo       = Mo * np.ones_like(Pt_in)
            Pt_out   = np.ones_like(Pt_in)
            P_out    = np.ones_like(Pt_in)

            #-- Inlet Mach <= 1.0, isentropic relations
            Pt_out[i_low]  = Pt_in[i_low]*pid
            Mach[i_low]    = np.sqrt( (((Pt_out[i_low]/Po[i_low])**((gamma[i_low]-1.)/gamma[i_low]))-1.) *2./(gamma[i_low]-1.) ) 
            T_out[i_low]   = Tt_out[i_low]/(1.+(gamma[i_low]-1.)/2.*Mach[i_low]*Mach[i_low])

            #-- Inlet Mach > 1.0, normal shock
            Mach[i_high]   = np.sqrt((1.+(gamma[i_high]-1.)/2.*Mo[i_high]**2.)/(gamma[i_high]*Mo[i_high]**2-(gamma[i_high]-1.)/2.))
            T_out[i_high]  = Tt_out[i_high]/(1.+(gamma[i_high]-1.)/2*Mach[i_high]*Mach[i_high])
            Pt_out[i_high] = pid*Pt_in[i_high]*((((gamma[i_high]+1.)*(Mo[i_high]**2.))/((gamma[i_high]-1.)*Mo[i_high]**2.+2.))**(gamma[i_high]/(gamma[i_high]-1.)))*((gamma[i_high]+1.)/(2.*gamma[i_high]*Mo[i_high]**2.-(gamma[i_high]-1.)))**(1./(gamma[i_high]-1.))
            P_out[i_high]  = Pt_out[i_high]*(1.+(gamma[i_high]-1.)/2.*Mach[i_high]**2.)**(-gamma[i_high]/(gamma[i_high]-1.))
        else:
            Pt_out  = Pt_in*pid*eta_rec
            
            # in case pressures go too low
            if np.any(Pt_out<Po):
                warn('Pt_out goes too low',RuntimeWarning)
                Pt_out[Pt_out<Po] = Po[Pt_out<Po]

            Mach    = np.sqrt( (((Pt_out/Po)**((gamma-1.)/gamma))-1.) *2./(gamma-1.) )
            T_out  = Tt_out/(1.+(gamma-1.)/2.*Mach*Mach)


        #-- Compute exit velocity and enthalpy
        h_out   = Cp*T_out
        u_out   = np.sqrt(2.*(ht_out-h_out))

        #pack computed quantities into outputs
        self.outputs.stagnation_temperature  = Tt_out
        self.outputs.stagnation_pressure     = Pt_out
        self.outputs.stagnation_enthalpy     = ht_out
        self.outputs.mach_number             = Mach
        self.outputs.static_temperature      = T_out
        self.outputs.static_enthalpy         = h_out
        self.outputs.velocity                = u_out
                
    def compute_scramjet(self,conditions): 
        """This function computes the compression of a scramjet 
        using shock trains.  
    
        Assumptions: 
    
        Source: 
        Heiser, William H., Pratt, D. T., Daley, D. H., and Unmeel, B. M.,  
        "Hypersonic Airbreathing Propulsion", 1994  
        Chapter 4 - pgs. 175-180
        
        Inputs: 
           conditions.freestream. 
           isentropic_expansion_factor        [-] 
           specific_heat_at_constant_pressure [J/(kg K)] 
           pressure                           [Pa] 
           gas_specific_constant              [J/(kg K)] 
           temperature                        [K] 
           mach_number                        [-] 
           velocity                           [m/s] 
    
        self.inputs. 
           stagnation_temperature             [K] 
           stagnation_pressure                [Pa] 
    
        Outputs: 
        self.outputs. 
           stagnation_temperature             [K] 
           stagnation_pressure                [Pa] 
           stagnation_enthalpy                [J/kg] 
           mach_number                        [-] 
           static_temperature                 [K] 
           static_enthalpy                    [J/kg] 
           velocity                           [m/s] 
           specific_heat_at_constant_pressure [J/(kg K)] 
    
        Properties Used: 
        self. 
           efficiency                         [-] 
           shock_count                        [-] 
           theta                              [Rad] 
        """ 

        # unpack the values 
    
        # unpack from conditions 
        gamma       = conditions.freestream.isentropic_expansion_factor 
        Cp          = conditions.freestream.specific_heat_at_constant_pressure 
        P0          = conditions.freestream.pressure 
        T0          = conditions.freestream.temperature 
        M0          = conditions.freestream.mach_number 
        U0          = conditions.freestream.velocity
        R           = conditions.freestream.gas_specific_constant
        
        # unpack from inputs 
        Tt_in       = self.inputs.stagnation_temperature 
        Pt_in       = self.inputs.stagnation_pressure 
        
        # unpack from self 
        eta         = self.polytropic_efficiency 
        shock_count = self.compression_levels 
        theta       = self.theta
        
        # compute compressed flow variables  
        
        # compute inlet conditions, based on geometry and number of shocks 
        psi, Ptr    = shock_train(M0,gamma,shock_count,theta) 
        
        # Compute/Look Up New gamma and Cp values (Future Work)
        gamma_c     = gamma
        Cp_c        = Cp
        
        # compute outputs 
        T_out       = psi*T0 
        P_out       = P0*(psi/(psi*(1.-eta)+eta))**(Cp_c/R) 
        Pt_out      = Ptr*Pt_in 
        Mach        = np.sqrt((2./(gamma_c-1.))*((T0/T_out)*(1.+(gamma_c-1.)/2.*M0*M0)-1.)) 
        u_out       = np.sqrt(U0*U0-2.*Cp_c*T0*(psi-1.)) 
        h_out       = Cp_c*T_out 
        Tt_out      = Tt_in 
        ht_out      = Cp_c*Tt_out 
        
        # packing output values  
        self.outputs.stagnation_temperature             = Tt_out              
        self.outputs.stagnation_pressure                = Pt_out                
        self.outputs.stagnation_enthalpy                = ht_out         
        self.outputs.mach_number                        = Mach        
        self.outputs.static_temperature                 = T_out        
        self.outputs.static_enthalpy                    = h_out          
        self.outputs.static_pressure                    = P_out
        self.outputs.specific_heat_at_constant_pressure = Cp_c
        self.outputs.velocity                           = u_out
        
    __call__ = compute
