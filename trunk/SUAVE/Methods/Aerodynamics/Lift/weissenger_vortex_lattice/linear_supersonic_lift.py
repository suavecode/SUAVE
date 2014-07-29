###################################################


# Obselete - do not use














# linear_supersonic_lift.py
# 
# Created:  Tim MacDonald 7/1/14
# Modified: Tim MacDonald 7/1/14

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
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

def linear_supersonic_lift(conditions,configuration,wing):
    """ Computes lift using linear supersonic theory

        Inputs:
            wing - geometry dictionary with fields:
            Sref - reference area

        Outputs:

        Assumptions:
        
    """

    #unpack
    
    
    
    span       = wing.span
    root_chord = wing.chord_root
    tip_chord  = wing.chord_tip
    sweep      = wing.sweep
    taper      = wing.taper
    twist_rc   = wing.twist_rc
    twist_tc   = wing.twist_tc
    sym_para   = wing.symmetric
    AR         = wing.ar
    Sref       = wing.sref
    orientation = wing.vertical

    # conditions
    aoa = conditions.aerodynamics.angle_of_attack
    aoa = aoa*np.pi/180
    
    n  = configuration.number_panels_spanwise
    
    # chord difference
    dchord=(root_chord-tip_chord)
    if sym_para is True :
        span=span/2
    deltax=span/n


    if orientation == False :

        section_length= np.empty(n)
        area_section=np.empty(n)
        twist_distri=np.empty(n)
    
    
    
    
        #--discretizing the wing sections into panels--------------------------------
        for i in range(0,n):
    
            section_length[i]= dchord/span*(span-(i+1)*deltax+deltax/2) + tip_chord
            area_section[i]=section_length[i]*deltax
            twist_distri[i]=twist_rc + i/float(n)*(twist_tc-twist_rc)
    
        area_tot = 0.0        
        cl_tot_base = 0.0
        
        for j in range(0,n):
            # Check angles here
            #cl[j] = 4*(aoa-twist_distri[j])/np.sqrt(Mc**2-1.0)*area_section[j] lift correction to go later
            cl[j] = 4*(aoa-twist_distri[j])*area_section[j]
            area_tot = area_tot+area_section[j]
            cl_tot_base = cl_tot_base + cl[j]
    
        Cl=cl_tot_base/area_tot*2.0 # Lift for both wings
        print("Linear Supersonic Theory")
    
    else:
        
        Cl= 0.0       


    return Cl