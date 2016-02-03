# International_Standard.py: 

# Created:  Mar, 2014, SUAVE Team
# Modified: Jan, 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Attributes.Atmospheres import Atmosphere
from SUAVE.Attributes.Gases import Air
from SUAVE.Attributes.Atmospheres.Earth import US_Standard_1976

# ----------------------------------------------------------------------
#  Classes
# ----------------------------------------------------------------------

# from background research, this appears true
class International_Standard(US_Standard_1976):
    """ International_Standard.py: International Standard Atmosphere (ISO2533:1975) """
    def __defaults__(self):
        self.tag = 'International Standard Atmosphere'    

# ----------------------------------------------------------------------
