""" Steam.py: Physical description of steam """

# Created by:     M. Vegh 12/12/13
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# classes
from Gas import Gas
from SUAVE.Attributes.Constants import Universe, Composition
from SUAVE.Structure import Data, Data_Exception, Data_Warning

# modules
#import numpy as np
from math import sqrt

# initialize local constants
Universe = Universe()

# ----------------------------------------------------------------------
#  Air
# ----------------------------------------------------------------------

class Steam(Gas):

    """ Physical constants specific to Air """

    def __defaults__(self):

        self.molecular_mass = 18.             # kg/kmol
        self.gas_specific_constant = 461.889                     # m^2/s^2-K, specific gas constant
        self.composition = Data()
        self.composition.H2O = 1.0

    def compute_density(self,T=300,p=101325):

            return p/(self.gas_specific_constant*T)

    def compute_speed_of_sound(self,T=300,p=101325,variable_gamma=False):

        if variable_gamma:
            g = self.compute_gamma(T,p)
        else:
            g = 1.33

        return sqrt(g*self.gas_specific_constant*T)

    def compute_cv(self,T=300,p=101325):

        # placeholder 

        raise NotImplementedError

    def compute_cp(self,T=300,p=101325):

        """  3rd-order polynomial data fit:
            cp(T) = c1*T^3 + c2*T^2 + c3*T + c4


            cp in J/kg-K, T in K
            Valid for 300 K < T < 1500 K """

        c = [5E-9, -.0001,  .9202, 1524.7]
        cp = c[0]*T**3 + c[1]*T**2 + c[2]*T + c[3]

        return cp

    def compute_gamma(self,T=300,p=101325):
        g=1.33                      #use constant value for now; will add a better model later
        
        return g

    def compute_absolute_viscosity(self,T=300,p=101325):

        mu0=1E-6

        return mu0