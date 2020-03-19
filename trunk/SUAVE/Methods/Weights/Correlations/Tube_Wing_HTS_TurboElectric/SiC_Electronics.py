## @ingroup Methods-Weights-Correlations-Tube_Wing_HTS_TurboElectric
# SiC_Electronics.py
#
# Created:  Mar 2020,   K. Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  SiC Electronics mass estimation
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Tube_Wing_HTS_TurboElectric
def SiC_mass(HTS_DC_Supply):
    """ Basic mass estimation for silicon carbide power electronics.

        Assumptions:
        Mass scales linearly with power and current

        Source:
        Siemens aviation electronic speed controllers
        http://sustainableskies.org/two-siemens-powered-electric-aircraft-debut/
        900g of power electronics reported to supply 104kVA

        Inputs:
        current             [A]
        power_out           [W]

        Outputs:
        mass                [kg]

    """

    # Unpack inputs
    power           = HTS_DC_Supply.rated_power
    current         = HTS_DC_Supply.rated_current

    # Calculate SiC specific power
    specific_power  = 104000/0.9     # W/kg

    # Estimate mass
    mass = power / specific_power

    # Pack results
    HTS_DC_Supply.mass_properties.mass  = mass          # [kg]

    # Return result.
    return mass