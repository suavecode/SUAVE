## @ingroup Methods-Weights-Correlations-Tube_Wing_HTS_TurboElectric
# Current_Supply.py
#
# Created:  May 2020,   K. Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Current supply mass estimation
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Tube_Wing_HTS_TurboElectric
def current_supply_mass(HTS_Current_Supply):
    """ Basic mass estimation for silicon based high current supply.

        Assumptions:
        Mass scales linearly with power and current according to Mass = 0.0035*Power + 5.5

        Source:
        Survey of high current (100A+) supplies available from i-Sunam (NEOS series) and Ametek (SGE series). Note these units are typically ~80-85% efficient according to the i-Sunam and Amemtek datasheets.

        Inputs:
        current             [A]
        power_out           [W]

        Outputs:
        mass                [kg]

    """

    # Unpack inputs
    power           = HTS_Current_Supply.rated_power    # [W]
    current         = HTS_Current_Supply.rated_current  # [A]

    # Estimate mass
    mass = 0.0035 * power + 5.5

    # Pack results
    HTS_Current_Supply.mass_properties.mass  = mass          # [kg]

    # Return result.
    return mass