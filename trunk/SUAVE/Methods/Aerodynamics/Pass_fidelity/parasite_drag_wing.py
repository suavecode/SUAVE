# parasite_drag_wing.py
# 
# Created:  Your Name, Dec 2013
# Modified:         

""" parasite_drag_wing(wing,segment)
    """

# ----------------------------------------------------------------------
#  Imports
#

# suave imports
import SUAVE
from SUAVE.Attributes.Gases import Air # you should let the user pass this as input
air = Air()
compute_speed_of_sound = air.compute_speed_of_sound

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp



#def cdp_wing(l_w,t_c_w,sweep_w, S_exposed_w,Sref,Mc,roc,muc,Tc):
def parasite_drag_wing(wing, segment, Sref):
    """ SUAVE.Methods.parasite_drag_wing(Wing,segment)
        computes the parastite drag associated with a wing 
        
        Inputs:
            Wing- A wing object is passed in
            segment - the segment object contains information regarding the mission segment
            
        Outpus:
            cd_p_w - returns the parasite drag assoicated with the wing
            
            >> try to minimize outputs
            >> pack up outputs into Data() if needed
        
        Assumptions:
            if needed
        
    """
    
    # unpack inputs
    
    #mac_w=wing.mac_w
    mac_w=wing.chord_mac
    t_c_w=wing.t_c
    sweep_w=wing.sweep
    S_exposed_w=wing.S_exposed
    S_affected_w=wing.S_affected
    arw_w=wing.ar
    span_w=wing.span    
    
    Mc=segment.M
    roc=segment.rho
    muc=segment.mew
    Tc=segment.T    
    pc=segment.p
    
    # process
    #----------------wing drag----------------------------
    V=Mc*compute_speed_of_sound(Tc,pc) #numpy.sqrt(1.4*287*Tc)  #input gamma and R
    Re_w=roc*V*mac_w/muc

    cf_inc_w=0.455/(np.log10(Re_w))**2.58  #for turbulent part
    #--effect of mach number on cf for turbulent flow
        
    Tw=Tc*(1+0.178*Mc**2)
    Td=Tc*(1 + 0.035*Mc**2 + 0.45*(Tw/Tc -1))
    Rd_w=Re_w*(Td/Tc)**1.5 *(Td+216/Tc+216)
    cf_w=(Tc/Td)*(Re_w/Rd_w)**0.2*cf_inc_w       

    #---for airfoils-----------------------------------------
    C=1.1  #best
    k_w=1+ 2*C*t_c_w*(np.cos(sweep_w))**2/(np.sqrt(1- Mc**2*(np.cos(sweep_w))**2)) + C**2*(np.cos(sweep_w))**2*t_c_w**2 *(1+5*(np.cos(sweep_w)**2))/(2*(1-(Mc*np.cos(sweep_w))**2))       

    Swet_w=2*(1+ 0.2*t_c_w)*S_exposed_w
    cd_p_w =k_w*cf_w*Swet_w /Sref 

    return cd_p_w