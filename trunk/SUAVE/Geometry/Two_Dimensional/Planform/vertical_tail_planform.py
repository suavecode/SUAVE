# Geoemtry.py
#

""" SUAVE Methods for Geoemtry Generation
"""


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy
from math import pi, sqrt
from SUAVE.Structure  import Data
from SUAVE.Geometry.Two_Dimensional.Planform.wing_planform import wing_planform
#from SUAVE.Attributes import Constants

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def vertical_tail_planform(Wing):
    """ results = SUAVE.Geometry.vertical_tail_planform(Wing)
        
        see SUAVE.Geometry.wing_planform()
    """
    wing_planform(Wing)
    return 0
    