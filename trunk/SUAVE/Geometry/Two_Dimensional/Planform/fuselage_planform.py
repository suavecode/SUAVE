# Geoemtry.py
#
# Last Modified: Tim MacDonald 7/10/14
# Added Deff to class parameters

""" SUAVE Methods for Geometry Generation
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
from SUAVE.Geometry.Two_Dimensional.Planform import fuselage_tube_wing_planform

#from SUAVE.Attributes import Constants

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------


def fuselage_planform(fuselage):
    """ err = SUAVE.Geometry.fuselage_planform(fuselage)
    
        Assumptions:
            fuselage cross section is an ellipse
            ellipse circumference approximated
            
        Inputs:
            fuselage.num_coach_seats
            fuselage.seat_pitch
            fuselage.fineness_nose
            fuselage.fineness_tail
            fuselage.fore_space
            fuselage.aft_space
            fuselage.width
            fuselage.height            
            
        Outputs:
            fuselage.length_nose
            fuselage.length_tail
            fuselage.length_cabin
            fuselage.length_total
            fuselage.area_wetted
            
    """
    
    # unpack
    number_seats    = fuselage.number_coach_seats
    seat_pitch      = fuselage.seat_pitch
    seats_abreast   = fuselage.seats_abreast
    forward_extra   = fuselage.lengths.fore_space
    aft_extra       = fuselage.lengths.aft_space

    # process
    fuselage.length_constant_section = number_seats * seat_pitch / seats_abreast + \
                   forward_extra + aft_extra

    # TODO: there seems to be a problem with the __init__.py: fuselage_tube_wing_planform should be imported
    fuselage_tube_wing_planform.fuselage_tube_wing_planform(fuselage)

    return 0
