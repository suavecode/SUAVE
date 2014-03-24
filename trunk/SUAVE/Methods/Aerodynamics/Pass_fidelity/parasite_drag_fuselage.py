# parasite_drag_fuselage.py
# 
# Created:  Your Name, Dec 2013
# Modified:         

""" parasite_drag_fuselage(wing,segment)

    Computes the parasite drag based on a set of fits
    """

# ----------------------------------------------------------------------
#  Imports
#
import SUAVE
# suave imports

from SUAVE.Attributes.Gases import Air
#import SUAVE.Attributes.Gases.Air as Air # you should let the user pass this as input
air = Air()
#compute_speed_of_sound = air.compute_speed_of_sound
#compute_gamma = air.comput_gamma

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp
#-------------------compressiblity drag----------------------------------------------------------

#def cdp_fuselage(l_fus, d_fus,l_nose,l_tail,Sref,Mc,roc,muc,Tc):
def parasite_drag_fuselage(fuselage,segment,Sref):
    """ SUAVE.Methods.parasite_drag_fuselage(Wing,segment)
        computes the parasite drag associated with a fuselage 
        
        Inputs:
            fuselage- A fuselage object is passed in
            segment - the segment object contains information regarding the mission segment
            Cl - wing Cl
            Sref - reference area for non dimensionalizion
        Outpus:
            cd_p_fus - returns the parasite drag assoicated with the fuselage
            
            >> try to minimize outputs
            >> pack up outputs into Data() if needed
        
        Assumptions:
            if needed
        
    """

    # unpack inputs
    
    l_fus=fuselage.length_cabin
    d_fus=fuselage.width
    l_nose=fuselage.length_nose
    l_tail=fuselage.length_tail
    
   
    
    Mc=segment.M
    roc=segment.rho
    muc=segment.mew
    Tc=segment.T    
    pc=segment.p
    R=287 


#---fuselage----------------------------------------

    gamma=air.compute_gamma(Tc,pc)
    V=Mc*air.compute_speed_of_sound(Tc,pc) #*(gamma*R*Tc)**0.5
    Re_fus=roc*V*l_fus/muc
    cf_inc_fus=0.455/((np.log10(Re_fus))**2.58)
    #for turbulent part
    #--effect of mach number on cf for turbulent flow
    Tw=Tc*(1+0.178*Mc**2)
    Td=Tc*(1 + 0.035*Mc**2 + 0.45*(Tw/Tc -1))
    Rd_fus=Re_fus*(Td/Tc)**1.5 *(Td+216/Tc+216)
    cf_fus=(Tc/Td)*(Re_fus/Rd_fus)**0.2*cf_inc_fus        

    
    #--------------for cylindrical bodies
    d_d=float(d_fus)/float(l_fus)
    D = np.sqrt(1-(1-Mc**2)*d_d**2)
    
    C_fus=2.3
    a=2*(1-Mc**2)*(d_d**2)/(D**3)*(np.arctanh(D)-D)
    du_max_u = a/(2-a)/(1-Mc**2)**0.5
    k_fus=(1+C_fus*du_max_u)**2    


    S_wetted_nose=0.75*np.pi*d_fus*l_nose
    S_wetted_tail=0.72*np.pi*d_fus*l_tail
    S_fus=np.pi*d_fus*l_fus
    S_fusetot=S_wetted_nose+S_wetted_tail+S_fus       
    cd_p_fus =k_fus*cf_fus*S_fusetot /Sref  


    return cd_p_fus

