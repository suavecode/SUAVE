# compute_aircraft_drag.py
# 
# Created:  Your Name, Dec 2013
# Modified:         

""" compute_aircraft_drag(aircraft, segment,Sref,cl_w,cd_w)
    """

# ----------------------------------------------------------------------
#  Imports
#
import SUAVE
# suave imports
#from SUAVE.Attributes.Gases.Air import compute_speed_of_sound

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp



#def cdp_wing(l_w,t_c_w,sweep_w, S_exposed_w,Sref,Mc,roc,muc,Tc):
def compute_aircraft_drag(aircraft, segment,Sref,cl_w,cd_w):
    """ SUAVE.Methods.compute_aircraft_drag(Wing,segment)
        computes the total drag associated with an aircraft 
        
        Inputs:
            aircraft- A wing object is passed in
            segment - the segment object contains information regarding the mission segment
            Sref - drag associated with an aircraft
        Outpus:
            drag_aircraft - returns the parasite drag assoicated with the wing
            
            >> try to minimize outputs
            >> pack up outputs into Data() if needed
        
        Assumptions:
            if needed
        
    """
    
    # unpack inputs
    
      ###cd_i_w=cdi(Cl_w, arw_w, cdi_w,cdp_w,d_fus/span_w)
    #SUAVE.Methods.Aerodynamics.Drag.Correlations
    #parasite_drag_total=parasite_drag_aircraft(aircraft,segment,Cl,cdi_inv,cdp,fd_ws)
    
    #parasite_drag_total=SUAVE.Methods.Aerodynamics.Pass_fidelity.parasite_drag_aircraft(aircraft,segment)    
    #[cd_i,induced_drag_total] =SUAVE.Methods.Aerodynamics.Pass_fidelity.induced_drag_wing(aircraft,segment,cl_w,cd_w,parasite_drag_total)
    #[cd_c,compressibility_drag_total]=SUAVE.Methods.Aerodynamics.Pass_fidelity.compressibility_drag_wing(aircraft,segment,cl_w)
    
    parasite_drag_total=SUAVE.Methods.Aerodynamics.Drag.Correlations.parasite_drag_aircraft(aircraft,segment)    
    [cd_i,induced_drag_total] =SUAVE.Methods.Aerodynamics.Drag.Correlations.induced_drag_wing(aircraft,segment,cl_w,cd_w,parasite_drag_total)
    [cd_c,compressibility_drag_total]=SUAVE.Methods.Aerodynamics.Drag.Correlations.compressibility_drag_wing(aircraft,segment,cl_w)    

    
    
    drag_aircraft_untrimmed=parasite_drag_total+induced_drag_total+compressibility_drag_total
    
    drag_aircraft=1.02*drag_aircraft_untrimmed  # correction for trim
    
    #print 'cd_i', cd_i
    #print 'cd_c', cd_c
    #print 'cd_p', parasite_drag_total
    
    
    return drag_aircraft