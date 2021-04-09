## @ingroup Attributes-Gases
# Air.py

# Created:  Mar. 2014, SUAVE Team
# Modified: Jan. 2016, M. Vegh
#           Dec. 2017, W. Maier 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .Gas import Gas
from SUAVE.Core import Data
import numpy as np

# ----------------------------------------------------------------------
#  Air Gas Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Gases
class Air(Gas):
    """Holds constants and functions that compute gas properties for air.
    
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

        self.molecular_mass = 28.96442        # kg/kmol
        self.gas_specific_constant = 287.0528742                 # m^2/s^2-K, specific gas constant
        self.composition.O2 = 0.20946
        self.composition.Ar = 0.00934
        self.composition.CO2 = 0.00036
        self.composition.N2  = 0.78084
        self.composition.other = 0.00

    def compute_density(self,T=300.,p=101325.):
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

    def compute_speed_of_sound(self,T=300.,p=101325.,variable_gamma=False):
        """Computes speed of sound given temperature and pressure

        Assumptions:
        Ideal gas with gamma = 1.4 if variable gamma is False

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
            g = 1.4*np.ones_like(T)
            
        return np.sqrt(g*self.gas_specific_constant*T)

    def compute_cp(self,T=300.,p=101325.):
        """Computes Cp by 3rd-order polynomial data fit:
        cp(T) = c1*T^3 + c2*T^2 + c3*T + c4

        Coefficients (with 95% confidence bounds):
        c1 = -7.357e-007  (-9.947e-007, -4.766e-007)
        c2 =    0.001307  (0.0009967, 0.001617)
        c3 =     -0.5558  (-0.6688, -0.4429)
        c4 =        1074  (1061, 1086) 
            
        Assumptions:
        123 K < T < 673 K 

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

        c = [-7.357e-007, 0.001307, -0.5558, 1074.0]
        cp = c[0]*T*T*T + c[1]*T*T + c[2]*T + c[3]

        return cp

    def compute_gamma(self,T=300.,p=101325.):
        """Computes Cp by 3rd-order polynomial data fit:
        gamma(T) = c1*T^3 + c2*T^2 + c3*T + c4

        Coefficients (with 95% confidence bounds):
        c1 =  1.629e-010  (1.486e-010, 1.773e-010)
        c2 = -3.588e-007  (-3.901e-007, -3.274e-007)
        c3 =   0.0001418  (0.0001221, 0.0001614)
        c4 =       1.386  (1.382, 1.389)
            
        Assumptions:
        233 K < T < 1273 K 

        Source:
        Unknown

        Inputs:
        T              [K]       - Temperature
        p              [Pa]      - Pressure

        Outputs:
        g              [-]

        Properties Used:
        None
        """   

        c = [1.629e-010, -3.588e-007, 0.0001418, 1.386]
        g = c[0]*T*T*T + c[1]*T*T + c[2]*T + c[3]

        return g

    def compute_absolute_viscosity(self,T=300.,p=101325.):
        """Compute the absolute (dynamic) viscosity
            
        Assumptions:
        Ideal gas

        Source:
        https://www.cfd-online.com/Wiki/Sutherland's_law

        Inputs:
        T                  [K]       - Temperature

        Outputs:
        absolute viscosity [kg/(m-s)]

        Properties Used:
        None
        """           

        S = 110.4                   # constant in deg K (Sutherland's Formula)
        C1 = 1.458e-6               # kg/m-s-sqrt(K), constant (Sutherland's Formula)

        return C1*(T**(1.5))/(T + S)