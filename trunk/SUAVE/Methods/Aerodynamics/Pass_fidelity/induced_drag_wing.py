# induced_drag_wing.py
# 
# Created:  Your Name, Dec 2013
# Modified:         

""" induced_drag_wing(wing,segment)

    Computes the induced drag based on a set of fits
    """

# ----------------------------------------------------------------------
#  Imports
#

# suave imports


# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp

#def cdi(Cl, AR, cdi_inv,cdp,fd_ws):
def induced_drag_wing(aircraft,segment,cl_w,cd_w,parasite_drag_total):
    """ SUAVE.Methods.induced_drag_wing(Wing,segment)
        computes the induced drag associated with a wing 
        
        Inputs:
            Wing- A wing object is passed in
            segment - the segment object contains information regarding the mission segment
            Cl - wing Cl
            cdi_inv -  inviscid drag component computed from the vortex lattice
            cdp - parasite drag
            fd_ws - 
        
        Outpus:
            cd_i  - returns the induced drag assoicated with the wing
            
            >> try to minimize outputs
            >> pack up outputs into Data() if needed
        
        Assumptions:
            if needed
        
    """

    # unpack inputs
    

    e=0.79
    
    #AR=wing.arw_w

    #mach=segment.M


    ## process

    ##s=0.92         #put in fit
    
    ##--induced drag computation based on AA241 methods
        
    #s = -1.7861*fd_ws**2 - 0.0377*fd_ws + 1.0007
    
    #cdi_inv = cdi_inv/s   #--inviscid component
    #K=0.38
    #cdi_viscous = K*cdp*Cl**2  #--viscous component
    
    #cd_i = cdi_inv + cdi_viscous
    
    cd_i=np.empty(len(aircraft.Wings))
    
    induced_drag_total=0.0
    for k in range(len(aircraft.Wings)):
        cd_i[k]=cl_w[k]**2/(np.pi*aircraft.Wings[k].ar*e)
        induced_drag_total=induced_drag_total+cd_i[k]
    
    
    

    return cd_i,induced_drag_total  