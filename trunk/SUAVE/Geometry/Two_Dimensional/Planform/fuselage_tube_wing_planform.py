# Geoemtry.py
#
# Last Modified: Jia Xu
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
#from SUAVE.Attributes import Constants

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------


def fuselage_tube_wing_planform(fuselage):
    """ err = SUAVE.Geometry.fuselage_tube_wing_planform(fuselage)
    
        Assumptions:
            fuselage cross section is an ellipse
            ellipse circumference approximated
            
        Inputs:
            fuselage.fineness_nose
            fuselage.fineness_tail
            fuselage.length_constant_section
            fuselage.width
            fuselage.height            
            
        Outputs:
            fuselage.length_nose
            fuselage.length_tail
            fuselage.length_cabin
            fuselage.area_wetted
            
    """
    
    # unpack
    nose_fineness   = fuselage.fineness.nose
    tail_fineness   = fuselage.fineness.tail
    fuselage_width  = fuselage.width
    fuselage_height = fuselage.heights.maximum
    constant_section_length = fuselage.length_constant_section
    
    # process
    nose_length  = nose_fineness * fuselage_width
    tail_length  = tail_fineness * fuselage_width

    fuselage_length = constant_section_length + nose_length + tail_length
    
    wetted_area = 0.0
    
    # model constant fuselage cross section as an ellipse
    # approximate circumference http://en.wikipedia.org/wiki/Ellipse#Circumference
    a = fuselage_width/2.
    b = fuselage_height/2.
    cross_section_area = pi * a * b  # area
    r_effective = (a-b)/(a+b) # effective radius
    circumference = pi*(a+b)*(1.+ ( 3*r_effective**2 )/( 10+sqrt(4.-3.*r_effective**2) )) # circumfrence
    wetted_area += circumference * constant_section_length

    # approximate nose and tail wetted area
    # http://adg.stanford.edu/aa241/drag/wettedarea.html
    diameter_effective = (a+b)*(64.-3.*r_effective**4)/(64.-16.*r_effective**2)
    wetted_area += 0.75*pi*diameter_effective * (nose_length + tail_length)
    
    # reference area approximated with
    reference_area = cross_section_area

    # update
    fuselage.lengths.nose  = nose_length
    fuselage.lengths.tail  = tail_length
    fuselage.lengths.cabin = constant_section_length # this needs a more generic name
    fuselage.lengths.total = fuselage_length
    fuselage.areas.wetted  = wetted_area
    fuselage.areas.front_projected = cross_section_area
    fuselage.effective_diameter = diameter_effective
    
    return 0
