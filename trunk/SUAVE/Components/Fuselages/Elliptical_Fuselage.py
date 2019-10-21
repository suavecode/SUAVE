## @ingroup components-fuselages
# Fuselage.py
#
# Created:  Oct 2019, J. Smart
# Modified:

#-----------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------

from SUAVE.Components.Fuselages import Fuselage

#-----------------------------------------------------------------------
# Elliptical Fuselage
#-----------------------------------------------------------------------

## @ingroup components-fuselages
class Elliptical_Fuselage(Fuselage):
    """Elliptical body fuselage for eVTOL and other small aircraft.

    Assumptions:
    Rounded Body Characterized by Three Major Axes (Maximum Length, Width, Height)

    Source:
    N/A
    """

    def __defaults__(self):
        """ This sets the default values for the component to function.

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

        return
