## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Planform
# fuselage_planform.py
#
# Created:  Jul 2014, T. MacDonald
# Modified: Jan 2016, E. Botero
#           May 2020, E. Botero


# TODO:
# object placement, wing location
# tail: placed at end of fuselage, or pull from volume
# engines: number of engines, position by 757

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Planform
def fuselage_planform(fuselage):
    """Calculates fuselage geometry values

    Assumptions:
    None

    Source:
    http://adg.stanford.edu/aa241/drag/wettedarea.html

    Inputs:
    fuselage.
      num_coach_seats       [-]
      seat_pitch            [m]
      seats_abreast         [-]
      fineness.nose         [-]
      fineness.tail         [-]
      lengths.fore_space    [m]
      lengths.aft_space     [m]
      width                 [m]
      heights.maximum       [m]

    Outputs:
    fuselage.
      lengths.nose          [m]
      lengths.tail          [m]
      lengths.cabin         [m]
      lengths.total         [m]
      areas.wetted          [m]
      areas.front_projected [m]
      effective_diameter    [m]

    Properties Used:
    N/A
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
    length          = fuselage.lengths.total
    
    if length ==0.:    
        # process
        nose_length     = nose_fineness * fuselage_width
        tail_length     = tail_fineness * fuselage_width
        cabin_length    = number_seats * seat_pitch / seats_abreast + \
                       forward_extra + aft_extra
        fuselage_length = cabin_length + nose_length + tail_length
    else:
        fuselage_length = fuselage.lengths.total
        nose_length     = nose_fineness * fuselage_width
        tail_length     = tail_fineness * fuselage_width      
        cabin_length    = fuselage_length - nose_length - tail_length
        
        if fuselage_length <= 0:
            fuselage_length = 1.
        if nose_length <= 0.:
            nose_length = 1.
        if tail_length <= 0:
            tail_length = 1.
        if cabin_length <= 1.:
            cabin_length = 1.
        
        # Now we can calculate the number of passengers
        number_seats    = np.round(cabin_length * seats_abreast / seat_pitch)
        if number_seats <=0: number_seats=0
    
    wetted_area = 0.0
    
    # model constant fuselage cross section as an ellipse
    # approximate circumference http://en.wikipedia.org/wiki/Ellipse#Circumference
    a = fuselage_width/2.
    b = fuselage_height/2.
    A = np.pi * a * b  # area
    R = (a-b)/(a+b) # effective radius
    C = np.pi*(a+b)*(1.+ ( 3*R**2 )/( 10+np.sqrt(4.-3.*R**2) )) # circumference
    
    wetted_area += C * cabin_length
    cross_section_area = A
    
    # approximate nose and tail wetted area
    # http://adg.stanford.edu/aa241/drag/wettedarea.html
    Deff = (a+b)*(64.-3.*R**4)/(64.-16.*R**2)
    wetted_area += 0.75*np.pi*Deff * (nose_length + tail_length)
    
    # update
    fuselage.lengths.nose          = nose_length
    fuselage.lengths.tail          = tail_length
    fuselage.lengths.cabin         = cabin_length
    fuselage.lengths.total         = fuselage_length
    fuselage.areas.wetted          = wetted_area
    fuselage.areas.front_projected = cross_section_area
    fuselage.effective_diameter    = Deff
    fuselage.number_coach_seats    = number_seats
    
    return fuselage
