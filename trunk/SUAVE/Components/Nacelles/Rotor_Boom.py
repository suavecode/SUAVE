## @ingroup Components-Nacelles
# Rotor_Boom.py
# 
# Created:    Dec 2022, J. Smart
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

from SUAVE.Components.Nacelles import Nacelle

## @ingroup Components-Nacelles
class Rotor_Boom(Nacelle):
    """
    ADD CLASS DESCRIPTION 

    Assumptions:
    None

    Source:
    None

    Properties Used:
    N/A	
    """

    def __defaults__(self, *args, **kwargs):
        """
        Sets the defaults for construction the Rotor_Boom.
        """

        self.number_of_rotors = None

    def __init__(self, *args, **kwargs):
        """
        Initialization that builds the Rotor_Boom by running the initialization
        of the super class and then setting extra defaults.
        """

        super().__init__()
        self.__defaults__()
