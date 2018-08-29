## @ingroup Attributes-Liquids
# Water.py

# Created:  Dec 2013, SUAVE TEAM
# Modified: Jan, 2016, M. Vegh
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .Liquid import Liquid
from SUAVE.Core import Data


# ----------------------------------------------------------------------
#  Water Liquid Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Liquids
class Water(Liquid):
    """Holds constants and functions that compute properties for water.
    
    Assumptions:
    None
    
    Source:
    None
    """

    def __defaults__(self):
        """This sets the default values.

        Assumptions:
        None

        Source:
        Values commonly available

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """             
        self.molecular_mass         = 18.             # kg/kmol
        self.composition.H2O        = 1.0
        self.heat_of_vaporization   = 2260.*1000                #heat of vaporization of water [J/kg]
        self.density                = 1000.                       #density of water
        self.boiling_point          = 373.15                    #boiling point (K)


    def compute_cp(self,temperature=300,pressure=101325):
        """Computes Cp by 4th-order polynomial data fit:
        cp(T) = c1*T^4 + c2*T^3 + c3*T^2 + c4*T+c
            
        Assumptions:
        278 K < T < 373.15 K

        Source:
        Unknown

        Inputs:
        temperature    [K]

        Outputs:
        cp             [J/kg-K]

        Properties Used:
        None
        """   
        T = temperature
        
        c  = [3E-6, -.0036,  1.8124, -406.5, 38390]
        cp = c[0]*T**4 + c[1]*T**3 + c[2]*T**2 + c[3]*T+c[4]

        return cp


    def compute_absolute_viscosity(self,temperature=300,pressure=101325):
        """Compute the absolute (dynamic) viscosity. Model accurate to within 
        2.5% between 0 C and 370C
            
        Assumptions:
        None

        Source:
        Engineering Fluid Mechanics, Al-Shemmeri, pg. 18

        Inputs:
        temperature  [K]

        Outputs:
        mu           [kg/(m-s)]

        Properties Used:
        None
        """         
        T  = temperature
        mu = (2.414E-5)*10**(247.8/(T-140))

        return mu