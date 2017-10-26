## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Planform
# vertical_tail_planform_raymer.py
#
# Created:  ### ####, M. Vegh
# Modified: Jan 2016, E. Botero


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods.Geometry.Two_Dimensional.Planform  import wing_planform

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Planform
def vertical_tail_planform_raymer(vertical_stabilizer, wing,  l_vt,c_vt):
    """Adjusts reference area before calling generic wing planform function to compute wing planform values.

    Assumptions:
    None

    Source:
    Raymer

    Inputs:
    vertical_stabilizer                    [SUAVE data structure]
    wing                                   [SUAVE data structure]  (should be the main wing)
    l_vt                                   [m] length from wing mean aerodynamic chord (MAC) to horizontal stabilizer MAC
    c_vt                                   [-] horizontal tail coefficient (Raymer specific) .02 = Sailplane, .04 = homebuilt, 
                                               .04 = GA single engine, .07 = GA twin engine, .04 = agricultural, 
                                               .08 = twin turboprop, .06 = flying boat, .06 = jet trainer, .07 = jet fighter
                                               .08 = military cargo/bomber, .09 = jet transport

    Outputs:
    vertical_stabilier.areas.reference     [m^2]
    Other changes to vertical_stabilizer (see wing_planform)

    Properties Used:
    N/A
    """           

    vertical_stabilizer.areas.reference = wing.spans.projected*c_vt*wing.areas.reference/l_vt
  
    wing_planform(vertical_stabilizer)
    
    return 0
    
