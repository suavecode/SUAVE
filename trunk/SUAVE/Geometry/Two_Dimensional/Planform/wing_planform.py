# Geoemtry.py
#

""" SUAVE Methods for Geoemtry Generation
"""

# TODO:
# object placement, wing location
# tail: placed at end of fuselage, or pull from volume
# engines: number of engines, position by 757

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy
from math import pi, sqrt
from SUAVE.Structure  import Data
#from SUAVE.Attributes import Constants

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------


def wing_planform(Wing):
    """ err = SUAVE.Geometry.wing_planform(Wing)
    
        basic wing planform calculation
        
        Assumptions:
            trapezoidal wing
            no leading/trailing edge extensions
            
        Inputs:
            Wing.sref
            Wing.ar
            Wing.taper
            Wing.sweep
            
        Outputs:
            Wing.chord_root
            Wing.chord_tip
            Wing.chord_mac
            Wing.area_wetted
            Wing.span
        
    """
    
    # unpack
    span  = Wing.Spans.projected
    sref  = Wing.Areas.reference
    #ar    = Wing.ar
    taper = Wing.taper
    sweep = Wing.sweep
    
    # calculate
    ar = span**2. / sref
    #span = sqrt(ar*sref)
    
    chord_root = 2*sref/span/(1+taper)
    chord_tip  = taper * chord_root
    
    swet = 2*span/2*(chord_root+chord_tip)

    mac = 2./3.*( chord_root+chord_tip - chord_root*chord_tip/(chord_root+chord_tip) )
    
    # update
    Wing.Chords.root  = chord_root
    Wing.Chords.tip   = chord_tip
    Wing.Chords.mean_aerodynamic   = mac
    Wing.Areas.wetted = swet
    #Wing.span        = span
    Wing.aspect_ratio          = ar
    
    return 0
