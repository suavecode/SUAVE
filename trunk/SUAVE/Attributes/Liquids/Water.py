#Water.py: Physical description of water
# Created:  Dec 2013, SUAVE TEAM
# Modified: Apr 2015, SUAVE TEAM
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# classes
from Liquid import Liquid
from SUAVE.Attributes.Constants import Composition
from SUAVE.Core import Data, Data_Exception, Data_Warning


# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Water(Liquid):

    """ Physical constants specific to Air """

    def __defaults__(self):

        self.molecular_mass         = 18.             # kg/kmol
        self.composition            = Data()
        self.composition.H2O        = 1.0
        self.heat_of_vaporization   =2260.*1000                #heat of vaporization of water [J/kg]
        self.density                =1000.                       #density of water
        self.boiling_point          =373.15                    #boiling point (K)


    def compute_cp(self,temperature=300,pressure=101325):

        """  4th-order polynomial data fit:
            cp(T) = c1*T^4 + c2*T^3 + c3*T^2 + c4*T+c5

            cp in J/kg-K, T in K
            Valid for 278 K < T < 373.15 K """
        T=temperature
        
        c = [3E-6, -.0036,  1.8124, -406.5, 38390]
        cp = c[0]*T**4 + c[1]*T**3 + c[2]*T**2 + c[3]*T+c[4]

        return cp


    def compute_absolute_viscosity(self,temperature=300,pressure=101325):
        """
        model accurate to within 2.5% between 0 C and 370C
        """
        T=temperature
        mu=(2.414E-5)*10**(247.8/(T-140))

        return mu