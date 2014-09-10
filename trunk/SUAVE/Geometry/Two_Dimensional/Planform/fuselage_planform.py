# Geoemtry.py
#
# Last Modified: Tim MacDonald 7/10/14
# Added Deff to class parameters

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


def fuselage_planform(Fuselage):
    """ err = SUAVE.Geometry.fuselage_planform(Fuselage)
    
        Assumptions:
            fuselage cross section is an ellipse
            ellipse circumference approximated
            
        Inputs:
            Fuselage.num_coach_seats
            Fuselage.seat_pitch
            Fuselage.fineness_nose
            Fuselage.fineness_tail
            Fuselage.fwdspace
            Fuselage.aftspace
            Fuselage.width
            Fuselage.height            
            
        Outputs:
            Fuselage.length_nose
            Fuselage.length_tail
            Fuselage.length_cabin
            Fuselage.length_total
            Fuselage.area_wetted
            
    """
    
    # unpack
    number_seats    = Fuselage.number_coach_seats
    seat_pitch      = Fuselage.seat_pitch
    seats_abreast   = Fuselage.seats_abreast
    nose_fineness   = Fuselage.fineness.nose
    tail_fineness   = Fuselage.fineness.tail
    forward_extra   = Fuselage.lengths.fore_space
    aft_extra       = Fuselage.lengths.aft_space
    fuselage_width  = Fuselage.width
    fuselage_height = Fuselage.heights.maximum
    
    # process
    nose_length  = nose_fineness * fuselage_width
    tail_length  = tail_fineness * fuselage_width
    cabin_length = number_seats * seat_pitch / seats_abreast + \
                   forward_extra + aft_extra
    fuselage_length = cabin_length + nose_length + tail_length
    
    wetted_area = 0.0
    
    # model constant fuselage cross section as an ellipse
    # approximate circumference http://en.wikipedia.org/wiki/Ellipse#Circumference
    a = fuselage_width/2.
    b = fuselage_height/2.
    A = pi * a * b  # area
    R = (a-b)/(a+b) # effective radius
    C = pi*(a+b)*(1.+ ( 3*R**2 )/( 10+sqrt(4.-3.*R**2) )) # circumfrence
    wetted_area += C * cabin_length
    cross_section_area = A
    
    # approximate nose and tail wetted area
    # http://adg.stanford.edu/aa241/drag/wettedarea.html
    Deff = (a+b)*(64.-3.*R**4)/(64.-16.*R**2)
    wetted_area += 0.75*pi*Deff * (nose_length + tail_length)
    
    # reference area approximated with
    reference_area = cross_section_area
    
    # update
    Fuselage.lengths.nose  = nose_length
    Fuselage.lengths.tail  = tail_length
    Fuselage.lengths.cabin = cabin_length
    Fuselage.lengths.total = fuselage_length
    Fuselage.areas.wetted  = wetted_area
    Fuselage.areas.front_projected = cross_section_area
    #Fuselage.Areas.front_projected     = reference_area # ?? CHECK
    Fuselage.effective_diameter         = Deff
    
    return 0
