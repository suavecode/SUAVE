## @ingroup Attributes-Propellants
#Gaseous_H2.py
#
# Created:  Unk 2013, SUAVE TEAM
# Modified: Feb 2016, M. Vegh
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from .Propellant import Propellant
from SUAVE.Attributes.Constants import Composition
from SUAVE.Core import Data
# ----------------------------------------------------------------------
#  Gaseous_H2 Propellant Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Propellants
class Gaseous_H2(Propellant):
    """Holds values for this propellant
    
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
        self.tag                       = 'H2 Gas'
        self.reactant                  = 'O2'
        self.specific_energy           = 141.86e6                           # J/kg
        self.energy_density            = 5591.13e6                          # J/m^3
        self.max_mass_fraction         = Data({'Air' : 0.013197, 'O2' : 0.0630})  # kg propellant / kg oxidizer
    

        # gas properties
        self.composition               = Composition( H2 = 1.0 )
        self.molecular_mass            = 2.016                             # kg/kmol
        self.gas_constant              = 4124.0                            # J/kg-K              
        self.pressure                  = 700e5                             # Pa
        self.temperature               = 293.0                             # K
        self.compressibility_factor    = 1.4699                            # compressibility factor
        self.density                   = 39.4116                           # kg/m^3
