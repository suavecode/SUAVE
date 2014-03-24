# compressibility_drag_wing.py
# 
# Created:  Your Name, Dec 2013
# Modified:         

""" compressibility_drag_wing(wing,segment)

    Computes the compressibility drag based on a set of fits
    """

# ----------------------------------------------------------------------
#  Imports
#

# suave imports
#from SUAVE.Attributes.Gases.Air import compute_speed_of_sound

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp
#-------------------compressiblity drag----------------------------------------------------------

#def cdc(mach,sweep_w,t_c_w,Cl):
#def cdc(self,Minf):
def compressibility_drag_wing(aircraft,segment,cl_w):
    """ SUAVE.Methods.compressibility_drag_wing(Wing,segment)
        computes the compressibility drag associated with a wing 
        
        Inputs:
            Wing- A wing object is passed in
            segment - the segment object contains information regarding the mission segment
            Cl - wing Cl
        Outpus:
            cd_c  - returns the compressibility drag assoicated with the wing
            
            >> try to minimize outputs
            >> pack up outputs into Data() if needed
        
        Assumptions:
            if needed
        
    """


    cd_c=np.empty(len(aircraft.Wings))

    compressibility_drag_total= 0.0

    for k in range(len(aircraft.Wings)):


        # unpack inputs
        
        t_c_w=aircraft.Wings[k].t_c
        sweep_w=aircraft.Wings[k].sweep
       
        
        mach=segment.M
    
    
        # process
    
        #--get effective Cl and sweep
        tc=t_c_w/np.cos(sweep_w)
        cl=cl_w[k]/(np.cos(sweep_w))**2
    
        #--computing the compressibility drag based on regressed fits from AA241
        mcc_cos_ws=0.922321524499352 -1.153885166170620*tc -0.304541067183461*cl  + 0.332881324404729*tc**2 +  0.467317361111105*tc*cl+   0.087490431201549*cl**2;
            
        mcc = mcc_cos_ws/np.cos(sweep_w)
    
        MDiv = mcc *(1.02 +.08 *( 1 - np.cos(sweep_w)))
    
        mo_mc=mach/mcc
        
        dcdc_cos3g = 413.56*(mo_mc)**6 - 2207.8*(mo_mc)**5 + 4900.1*(mo_mc)**4 - 5786.9*(mo_mc)**3 + 3835.3*(mo_mc)**2 - 1352.5*(mo_mc) + 198.25
        
        cd_c[k]=dcdc_cos3g*(np.cos(sweep_w))**3
        
        compressibility_drag_total=compressibility_drag_total+ cd_c[k]


    return cd_c,compressibility_drag_total 