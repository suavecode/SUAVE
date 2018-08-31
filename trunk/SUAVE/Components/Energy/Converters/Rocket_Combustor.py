## @ingroup Components-Energy-Converters
# Rocket_Combustor.py
#
# Created:  Feb 2018, W. Maier
# Modified: 

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
class Rocket_Combustor(Energy_Component):
    """This provides output values for a rocket combustion chamber.
    Calling this class calls the compute function.
    
    Assumptions:
    None
    
    Source:
    Chapter 7
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
        
        self.tag = 'Rocket_Combustor'
        
        # setting the default values for the different components
        self.propellant_data                    = None
        self.propellant_data                    = Data()
        self.inputs.combustion_pressure         = None
        self.inputs.throttle                    = None
        self.efficiency                         = None
    
    def compute(self,conditions):
        """ This computes the output values from the input values according to
        equations from the source. This function is currently a shell until 
        CEA surrogate is include. This functions similiarly to RAM.py at the
        moment.

        Assumptions:
        Combustion happens as M<<<1, assumed to be Mach = 0.
        Combustion needs CEA surrogate.
        
        Source:
        Chapter 7
        https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/

        Inputs:
        self.inputs.
          combustion_pressure           [Pa]

        Outputs:
        self.outputs.
          combustion_temperature        [K]  
          combustion_pressure           [Pa]
          isentropic_expansion_factor   [-]
          gas_specific_constant         [J/(Kg-K)]

        Properties Used:
       
        """         
        #--Unpack the values--
                
        # unpacking the values form inputs
        Pc     = self.inputs.combustion_pressure
        thr    = conditions.propulsion.throttle 
                      
        # unpacking values from self
        gamma = self.propellant_data.isentropic_expansion_factor
        Tc    = self.propellant_data.combustion_temperature
        R     = self.propellant_data.gas_specific_constant
                      
        #--Pack outputs--
        self.outputs.combustion_temperature      = Tc
        self.outputs.combustion_pressure         = Pc
        self.outputs.isentropic_expansion_factor = gamma
        self.outputs.gas_specific_constant       = R
        
    __call__ = compute    