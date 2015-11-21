# Geoemtry.py
#

""" SUAVE Methods for Geometry Generation
"""


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy
from math import pi, sqrt
from SUAVE.Core  import Data
from SUAVE.Methods.Geometry.Two_Dimensional.Planform  import wing_planform
#from SUAVE.Attributes import Constants

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
def horizontal_tail_planform_raymer(Htail, Wing,  l_ht,c_ht):
    """
    by M. Vegh
    Based on a tail sizing correlation from Raymer
    inputs:
    Htail =horizontal stabilizer
    Wing  =main wing
    l_ht  =length from wing mac to htail mac [m]
    c_ht  =horizontal tail coefficient
    
    sample c_ht values: .5=Sailplane, .5=homebuilt, .7=GA single engine, .8 GA twin engine
    .5=agricultural, .9=twin turboprop, .7=flying boat, .7=jet trainer, .4=jet fighter
    1.= military cargo/bomber, 1.= jet transport
    """
    
    Htail.areas.reference=Wing.chords.mean_aerodynamic*c_ht*Wing.areas.reference/l_ht
    #wing_planform(Htail)
    return 0    