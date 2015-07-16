# Geoemtry.py
#

""" SUAVE Methods for Geoemtry Generation
"""


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy
from math import pi, sqrt
from SUAVE.Core  import Data
#from SUAVE.Attributes import Constants

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
def horizontal_tail_planform(Wing):
    """ results = SUAVE.Methods.Geometry.horizontal_tail_planform(Wing)
    
        see SUAVE.Methods.Geometry.wing_planform()
    """
    wing_planform(Wing)
    return 0