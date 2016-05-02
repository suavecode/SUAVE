# horizontal_tail_planform_raymer.py
#
# Created:  ### ####, M. Vegh
# Modified: Feb 2016, E. Botero
#           Jan 2016, E. Botero

from wing_planform import wing_planform
# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
def horizontal_tail_planform_raymer(horizontal_stabilizer, wing,  l_ht,c_ht):
    """
    by M. Vegh
    Based on a tail sizing correlation from Raymer
    inputs:
    Htail =horizontal stabilizer
    Wing  =main wing
    l_ht  =length from wing mac to htail mac [m]
    c_ht  =horizontal tail coefficient
    
    sample c_ht values: .5=Sailplane, .5=homebuilt, .7=GA single engine, .8 GA twin engine
    .5=agricultural, .9=twin turboprop, .7=flying boat, .7=jet trainer, .4=jet fighter
    1.= military cargo/bomber, 1.= jet transport
    """
    
    horizontal_stabilizer.areas.reference = wing.chords.mean_aerodynamic*c_ht*wing.areas.reference/l_ht
    wing_planform(horizontal_stabilizer)
    
    return 0    
