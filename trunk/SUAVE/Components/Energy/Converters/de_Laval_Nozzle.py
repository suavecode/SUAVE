## @ingroup Components-Energy-Converters
# de_Laval_Nozzle.py
#
# Created:  Jan 2018, W. Maier
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
import SUAVE

from SUAVE.Core import Units

# package imports
import numpy as np

# suave imports
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Methods.Propulsion.fm_solver       import fm_solver

# ----------------------------------------------------------------------
#  de Laval Rocket Nozzle Component
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
class de_Laval_Nozzle(Energy_Component):
    """This is a nozzle component that allows for supersonic outflow. 
    This is a converging-diverging nozzle used primarilty in rockets.
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
        self.polytropic_efficiency           = None
        self.pressure_ratio_converge         = None
        self.pressure_ratio_diverge          = None
        self.expansion_ratio                 = None    
        self.area_throat                     = None
        
    def compute(self,conditions):
        """This computes the output values from the input values according to
        equations from the source.
        
        Assumptions:
        Constant polytropic efficiency and pressure ratio
        Isentropic Process from chamber to throat (Pt_throat = Pt2)
        Supersonic/Throat is Choked
        
        Source:
        Chapter 7
        https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/
        
        Inputs:
        self.inputs.
          combustion_temperature              [k]
          combustion_pressure                 [Pa]
          isentropic_expansion_factor         [-]
          gas_specific_constant               [J/(kg-K)]
                             
        Outputs:
        self.outputs.
          stagnation_pressure                 [Pa]
          stagnation_temperature              [K]  
          static_temperature                  [K]
          static_pressure                     [Pa]
          mach_number                         [-]
          velocity                            [m/s]
                
        Properties Used:
        self.
          pressure_ratio                      [-]
          polytropic_efficiency               [-]
          area_throat                         [m^2]
          expansion_ratio                     [-]
        """           

        #--Unpack the values--

        # unpack from inputs
        Pt_combustion  = self.inputs.combustion_pressure
        Tt_combustion  = self.inputs.combustion_temperature
        gamma          = self.inputs.isentropic_expansion_factor
        R              = self.inputs.gas_specific_constant     
        
        # unpack from self
        pi              = self.pressure_ratio
        etapold         = self.polytropic_efficiency
        expansion_ratio = self.expansion_ratio
        area_throat     = self.area_throat 
        
        
        #--Method for computing the nozzle properties--
        
        #--Converging Nozzle --
        # getting stagnation quantities at throat
        Pt_throat   = Pt_combustion*pi
        Tt_throat   = Tt_combustion*pi**((gamma-1.)/(gamma)*etapold)        
        
        # getting static quantities at throat
        T_throat   = Tt_throat/(1.+(gamma-1.)/2.)
        P_throat   = Pt_throat/((1.+(gamma-1.)/2.)**(gamma/(gamma-1.)))
        U_throat   = np.sqrt(gamma*R*T_throat)
        rho_throat = P_throat/(R*T_throat)             
        
        #--Diverging Nozzle--
        # getting stagnation quantities at exit
        Pt_out   = Pt_throat*pi
        Tt_out   = Tt_throat*pi**((gamma-1.)/(gamma)*etapold) 
        
        # calculation exit mach number
        Me       = fm_solver(expansion_ratio,1.0, gamma)
        
        # computing the output temperature, pressure, density, and velocity
        T_out    = Tt_out/(1.+(gamma-1.)/2.*Me*Me)
        P_out    = Pt_out/((1.+(gamma-1.)/2.*Me*Me)**(gamma/(gamma-1.)))
        U_out    = Me*np.sqrt(gamma*R*T_out)
        rho_out  = P_out/(R*T_out)
        
        #--pack computed quantities into outputs--
        self.outputs.stagnation_temperature      = Tt_out
        self.outputs.stagnation_pressure         = Pt_out
        self.outputs.combustion_pressure         = Pt_combustion
        self.outputs.static_pressure             = P_out
        self.outputs.static_temperature          = T_out
        self.outputs.mach_number                 = Me
        self.outputs.expansion_ratio             = expansion_ratio
        self.outputs.exhaust_velocity            = U_out 
        self.outputs.area_throat                 = area_throat
        self.outputs.isentropic_expansion_factor = gamma
        self.outputs.density                     = rho_out
        
    __call__ = compute