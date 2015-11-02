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

from SUAVE.Core  import Data
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
    sweep       = wing.sweep
    ar          = wing.aspect_ratio
    t_c_w       = wing.thickness_to_chord
    dihedral    = wing.dihedral 
    vertical    = wing.vertical
    symmetric   = wing.symmetric
    origin      = wing.origin
    
    # calculate
    span = (ar*sref)**.5
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
        
    # move AC to be in reference to the vehicle:
    #x_coord = x_coord + origin[0]
    #y_coord = y_coord + origin[1]
    #z_coord = z_coord + origin[2]
                    
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
    
    #imports
    wing = Data()
    wing.areas = Data()
    wing.chords = Data()
    wing.areas = Data()
    wing.spans = Data()
    
    wing.areas.reference        = 10.
    wing.taper                  =  1.0
    wing.sweep                  =  45.  * Units.deg
    wing.aspect_ratio           = 10.
    wing.thickness_to_chord     =  0.13
    wing.dihedral               =  45.  * Units.deg
    wing.vertical               =  1
    wing.symmetric              =  0

    wing_planform(wing)
    print wing