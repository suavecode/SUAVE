# wing_planform.py
#
# Created:  Apr 2014, T. Orra
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
def wing_planform(wing):
    """Computes standard wing planform values.

    Assumptions:
    Trapezoidal wing with no leading/trailing edge extensions

    Source:
    None

    Inputs:
    wing.
      areas.reference          [m^2]
      taper                    [-]
      sweeps.quarter_chord     [radians]
      aspect_ratio             [-]
      thickness_to_chord       [-]
      dihedral                 [radians]
      vertical                 <boolean> Determines if wing is vertical
      symmetric                <boolean> Determines if wing is symmetric
      origin                   [m]       x, y, and z position
      high_lift                <boolean> Determines if wing is in a high lift configuration
      flaps.                             Flap values are only used if high lift is True
        span_start             [-]       Span start position (.1 is 10% span)
        span_end               [-]       Span end position (.1 is 10% span)
        chord                  [-]       Portion of wing chord used (.1 is 10% chord)

    Outputs:
    wing.
      chords.root              [m]
      chords.tip               [m]
      chords.mean_aerodynamics [m]
      areas.wetted             [m^2]
      areas.affected           [m^2]
      spans.projected          [m]
      aerodynamic_center       [m]      x, y, and z location
      flaps.chord_dimensional  [m]
      flaps.area               [m^2]
        

    Properties Used:
    N/A
    """      
    
    # unpack
    sref        = wing.areas.reference
    taper       = wing.taper
    sweep       = wing.sweeps.quarter_chord
    ar          = wing.aspect_ratio
    t_c_w       = wing.thickness_to_chord
    dihedral    = wing.dihedral 
    vertical    = wing.vertical
    symmetric   = wing.symmetric
    origin      = wing.origin
    
    # calculate
    span       = (ar*sref)**.5
    chord_root = 2*sref/span/(1+taper)
    chord_tip  = taper * chord_root
    
    swet = 2.*span/2.*(chord_root+chord_tip) *  (1.0 + 0.2*t_c_w)

    mac = 2./3.*( chord_root+chord_tip - chord_root*chord_tip/(chord_root+chord_tip) )
    
    # calculate leading edge sweep
    le_sweep = np.arctan( np.tan(sweep) - (4./ar)*(0.-0.25)*(1.-taper)/(1.+taper) )
    
    # estimating aerodynamic center coordinates
    y_coord = span / 6. * (( 1. + 2. * taper ) / (1. + taper))
    x_coord = mac * 0.25 + y_coord * np.tan(le_sweep)
    z_coord = y_coord * np.tan(dihedral)
        
    if vertical:
        temp    = y_coord * 1.
        y_coord = z_coord * 1.
        z_coord = temp

    if symmetric:
        y_coord = 0    
        
    # Computing flap geometry
    affected_area = 0.
    if wing.high_lift:
        flap = wing.flaps
        #compute wing chords at flap start and end
        delta_chord = chord_tip - chord_root
        
        wing_chord_flap_start = chord_root + delta_chord * flap.span_start 
        wing_chord_flap_end   = chord_root + delta_chord * flap.span_end  
        wing_mac_flap = 2./3.*( wing_chord_flap_start+wing_chord_flap_end - \
                                wing_chord_flap_start*wing_chord_flap_end/  \
                                (wing_chord_flap_start+wing_chord_flap_end) )
        
        flap.chord_dimensional = wing_mac_flap * flap.chord
        flap_chord_start = wing_chord_flap_start * flap.chord
        flap_chord_end   = wing_chord_flap_end * flap.chord
        flap.area        = (flap_chord_start + flap_chord_end) * (flap.span_end - flap.span_start)*span / 2.    
        affected_area    = (wing_chord_flap_start + wing_chord_flap_end) * (flap.span_end - flap.span_start)*span / 2.          
        
    # update
    wing.chords.root                = chord_root
    wing.chords.tip                 = chord_tip
    wing.chords.mean_aerodynamic    = mac
    wing.areas.wetted               = swet
    wing.areas.affected             = affected_area
    wing.spans.projected            = span
    wing.aerodynamic_center         = [x_coord , y_coord, z_coord]
    
    return wing


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':

    from SUAVE.Core import Data,Units
    from SUAVE.Components.Wings import Wing
        
    #imports
    wing = Wing()
    
    wing.areas.reference        =  10.
    wing.taper                  =  0.50
    wing.sweeps.quarter_chord   =  45.  * Units.deg
    wing.aspect_ratio           =  10.
    wing.thickness_to_chord     =  0.13
    wing.dihedral               =  45.  * Units.deg
    wing.vertical               =  1
    wing.symmetric              =  0
    
    wing.flaps.chord = 0.28
    wing.flaps.span_start = 0.50
    wing.flaps.span_end   = 1.00

    wing_planform(wing)
    print(wing)
