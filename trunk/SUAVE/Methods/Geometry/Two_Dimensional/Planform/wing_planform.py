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
    """ err = SUAVE.Methods.Geometry.wing_planform(Wing)
    
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
    if wing.flaps.chord:     
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
    
    # update
    wing.chords.root                = chord_root
    wing.chords.tip                 = chord_tip
    wing.chords.mean_aerodynamic    = mac
    wing.areas.wetted               = swet
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
    
    wing.areas.reference        = 10.
    wing.taper                  =  0.50
    wing.sweeps.quarter_chord   =  45.  * Units.deg
    wing.aspect_ratio           = 10.
    wing.thickness_to_chord     =  0.13
    wing.dihedral               =  45.  * Units.deg
    wing.vertical               =  1
    wing.symmetric              =  0
    
    wing.flaps.chord = 0.28
    wing.flaps.span_start = 0.50
    wing.flaps.span_end   = 1.00

    wing_planform(wing)
    print wing
