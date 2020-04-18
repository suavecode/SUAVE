## @ingroup Attributes-Gases
# Steam.py

# Created:  Mar 2014, SUAVE Team
# Modified: Jan 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .Gas import Gas
from SUAVE.Core import Data

# modules
from math import sqrt


# ----------------------------------------------------------------------
#  Steam Gas Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Gases
class Steam(Gas):
    """Holds constants and functions that compute gas properties for steam.
    
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
        self.molecular_mass = 18.                  # kg/kmol
        self.gas_specific_constant = 461.889       # m^2/s^2-K, specific gas constant
        self.composition.H2O = 1.0

    def compute_density(self,T=300,p=101325):
        """Computes air density given temperature and pressure

        Assumptions:
        Ideal gas

        Source:
        Common equation

        Inputs:
        T         [K]  - Temperature
        p         [Pa] - Pressure

        Outputs:
        density   [kg/m^3]

        Properties Used:
        self.gas_specific_constant
        """        
        return p/(self.gas_specific_constant*T)

    def compute_speed_of_sound(self,T=300,p=101325,variable_gamma=False):
        """Computes speed of sound given temperature and pressure

        Assumptions:
        Ideal gas with gamma = 1.33 if variable gamma is False

        Source:
        Common equation

        Inputs:
        T              [K]       - Temperature
        p              [Pa]      - Pressure
        variable_gamma <boolean> - Determines if gamma is computed

        Outputs:
        speed of sound [m/s]

        Properties Used:
        self.compute_gamma() (if variable gamma is True)
        self.gas_specific_constant
        """        
        if variable_gamma:
            g = self.compute_gamma(T,p)
        else:
            g = 1.33

        return sqrt(g*self.gas_specific_constant*T)

    def compute_cv(self,T=300,p=101325):
        """Stub for computing Cv - not functional
        """  

        raise NotImplementedError

    def compute_cp(self,T=300,p=101325):
        """Computes Cp by 3rd-order polynomial data fit:
        cp(T) = c1*T^3 + c2*T^2 + c3*T + c4
            
        Assumptions:
        300 K < T < 1500 K

        Source:
        Unknown, possibly Combustion Technologies for a Clean Environment 
        (Energy, Combustion and the Environment), Jun 15, 1995, Carvalhoc

        Inputs:
        T              [K]       - Temperature
        p              [Pa]      - Pressure

        Outputs:
        cp             [J/kg-K]

        Properties Used:
        None
        """  

        c = [5E-9, -.0001,  .9202, 1524.7]
        cp = c[0]*T**3 + c[1]*T**2 + c[2]*T + c[3]

        return cp

    def compute_gamma(self,T=300,p=101325):
        """Gives constant gamma of 1.33
            
        Assumptions:
        233 K < T < 1273 K 

        Source:
        Common value

        Inputs:
        None

        Outputs:
        g              [-]

        Properties Used:
        None
        """           
        g=1.33                      #use constant value for now; will add a better model later
        
        return g

    def compute_absolute_viscosity(self,T=300,p=101325):
        """Gives constant absolute viscosity of 1e-6
        WARNING: this value appears to be incorrect
            
        Assumptions:
        Constant value

        Source:
        Common value

        Inputs:
        None

        Outputs:
        mu0        [kg/(m-s)]

        Properties Used:
        None
        """  
        mu0=1E-6

        return mu0