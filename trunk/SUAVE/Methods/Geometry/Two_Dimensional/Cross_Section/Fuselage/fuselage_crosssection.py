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
from SUAVE.Core  import Data
#from SUAVE.Attributes import Constants

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------


def fuselage_crosssection(Fuselage):
    """ err = SUAVE.Methods.Geometry.fuselage_planform(Fuselage)
        calculate relevent fuselage crosssection dimensions
        
        Assumptions:
            wall_thickness = 0.04 * seat_width
        
        Inputs:
            Fuselage.seat_width
            Fuselage.seat_layout_lower
            Fuselage.aisle_width
            Fuselage.fuse_hw
            
        Outputs:
            Fuselage.wall_thickness
            Fuselage.height
            Fusealge.width
        
    """
    
    # assumptions
    wall_thickness_ratio = 0.04 # ratio of inner diameter to wall thickness
    
    # unpack main floor
    seat_width     = Fuselage.seat_width
    layout         = Fuselage.seat_layout_lower
    aisle_width    = Fuselage.aisle_width
    fuselage_hw    = Fuselage.fuse_hw
    
    # calculate
    total_seat_width = sum(layout)*seat_width + \
                      (len(layout)-1)*aisle_width
    
    # needs verification
    wall_thickness  = total_seat_width * wall_thickness_ratio
    fuselage_width  = total_seat_width + 2*wall_thickness
    fuselage_height = fuselage_hw * fuselage_width
    
    # update
    Fuselage.wall_thickness  = wall_thickness
    Fuselage.width  = fuselage_width
    Fuselage.height = fuselage_height
    
    return 0
