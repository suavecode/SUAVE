# fuselage_planform.py
#
# Created:  Jul 2014, T. MacDonald
# Modified: Jan 2016, E. Botero

# TODO:
# object placement, wing location
# tail: placed at end of fuselage, or pull from volume
# engines: number of engines, position by 757

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from math import pi, sqrt

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------


def fuselage_planform(fuselage):
    """ err = SUAVE.Methods.Geometry.fuselage_planform(fuselage)
    
        Assumptions:
            fuselage cross section is an ellipse
            ellipse circumference approximated
            
        Inputs:
            fuselage.num_coach_seats
            fuselage.seat_pitch
            fuselage.fineness_nose
            fuselage.fineness_tail
            fuselage.fwdspace
            fuselage.aftspace
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
    nose_fineness   = fuselage.fineness.nose
    tail_fineness   = fuselage.fineness.tail
    forward_extra   = fuselage.lengths.fore_space
    aft_extra       = fuselage.lengths.aft_space
    fuselage_width  = fuselage.width
    fuselage_height = fuselage.heights.maximum
    
    # process
    nose_length     = nose_fineness * fuselage_width
    tail_length     = tail_fineness * fuselage_width
    cabin_length    = number_seats * seat_pitch / seats_abreast + \
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

    # estimating side projected area
    side_projected_area = fuselage_height * (cabin_length + 0.75 * (nose_length + tail_length))

    # reference area approximated with
    reference_area = cross_section_area
    
    # update
    fuselage.lengths.nose          = nose_length
    fuselage.lengths.tail          = tail_length
    fuselage.lengths.cabin         = cabin_length
    fuselage.lengths.total         = fuselage_length
    fuselage.areas.wetted          = wetted_area
    fuselage.areas.front_projected = cross_section_area
    fuselage.areas.side_projected  = side_projected_area
    fuselage.effective_diameter    = Deff
    
    return 0
