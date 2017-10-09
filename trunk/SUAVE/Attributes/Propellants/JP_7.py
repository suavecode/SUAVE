## @ingroup Attributes-Propellants
#Jet A
#
# Created:  Unk 2013, SUAVE TEAM
# Modified: Feb 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Propellant import Propellant

# ----------------------------------------------------------------------
#  Jet_A Propellant Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Propellants
class JP_7(Propellant):
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
        self.tag                       = 'JP_7'
        self.reactant                  = 'O2'
        self.density                   = 820.0                          # kg/m^3 (15 C, 1 atm)
        self.specific_energy           = 43.90325e6                       # J/kg
        self.energy_density            = 35276.4e6                      # J/m^3
        self.max_mass_fraction         = {'Air' : 0.0633, \
                                          'O2' : 0.3022}                # kg propellant / kg oxidizer

        # critical temperatures
        self.temperatures.flash        = 311.15                 # K
        self.temperatures.autoignition = 483.15                 # K
        self.temperatures.freeze       = 233.15                 # K
        self.temperatures.boiling      = 0.0                    # K