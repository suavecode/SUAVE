## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Planform
# horizontal_tail_planform_raymer.py
#
# Created:  ### ####, M. Vegh
# Modified: Feb 2016, E. Botero
#           Jan 2016, E. Botero

from .wing_planform import wing_planform
# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Planform
def horizontal_tail_planform_raymer(horizontal_stabilizer, wing,  l_ht,c_ht):
    """Adjusts reference area before calling generic wing planform function to compute wing planform values.

    Assumptions:
    None

    Source:
    Raymer

    Inputs:
    horizontal_stabilizer                  [SUAVE data structure]
    wing                                   [SUAVE data structure]  (should be the main wing)
    l_ht                                   [m] length from wing mean aerodynamic chord (MAC) to horizontal stabilizer MAC
    c_ht                                   [-] horizontal tail coefficient (Raymer specific) .5 = Sailplane, .5 = homebuilt, 
                                               .7 = GA single engine, .8 = GA twin engine .5 = agricultural, .9 = twin turboprop, 
                                               .7 = flying boat, .7 = jet trainer, .4 = jet fighter, 1. = military cargo/bomber, 
                                               1. = jet transport

    Outputs:
    horizontal_stabilier.areas.reference   [m^2]
    Other changes to horizontal_stabilizer (see wing_planform)

    Properties Used:
    N/A
    """       
    
    horizontal_stabilizer.areas.reference = wing.chords.mean_aerodynamic*c_ht*wing.areas.reference/l_ht
    wing_planform(horizontal_stabilizer)
    
    return 0    
