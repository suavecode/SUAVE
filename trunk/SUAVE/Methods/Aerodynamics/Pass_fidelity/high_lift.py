# high_lift.py
# 
# Created:  Your Name, Dec 2013
# Modified:         

""" high_lift(Wing,state,curr_itr)
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

def high_lift(Wing,state,Sref):
#def high_lift(l_fus, d_fus,l_nose,l_tail,mac_w,t_c_w,sweep_w, S_exposed_w, mac_h,t_c_h,sweep_h, S_exposed_h,mac_v,t_c_v,sweep_v, S_exposed_v,d_engexit,Sref,Mc,roc,muc ,Tc,Cl, AR, e,S_affected_w,S_affected_h,S_affected_v ):
##def drag(self,fus_l, d_fus,l_nose,l_tail,mac_w,t_c_w,sweep_w, S_exposed_w, mac_h,t_c_h,sweep_h, S_exposed_h,mac_v,t_c_v,sweep_v, S_exposed_v,d_engexit,Sref,Mc,rhoc,muc ,Tc,Cl, AR, e ):
    """ SUAVE.Methods.Aerodynamics.high_lift(Wing,state,curr_itr)
        computes the lift associated with an aircraft high lift system
        
        Inputs:
            Wing - Wing object # state what all the fields being used in wing are
            state - the segment object contains information regarding the mission segment
            alpha- angle of attack
            curr_itr - 
        
        Outpus:
             Cl_max_ls - 
             Cd_ind - 
            
    
        Assumptions:
            if needed
        
    """   
    #unpack

    t_c=Wing.t_c
    sweep=Wing.sweep
    taper=Wing.taper
    flap_chord=Wing.flaps_chord
    flap_angle=Wing.flaps_angle
    slat_angle=Wing.slats_angle
    Swf=Wing.S_affected
    
    #Sref=Wing.sref
    
    V=state.V
    Mcr=state.M
    roc=state.rho
    nu=state.mew
    #print taper
    
    tc=t_c*100
    #--cl max based on airfoil t_c
    Cl_max_ref= -0.0009*tc**3 + 0.0217*tc**2 - 0.0442*tc + 0.7005

    #-reynolds number effect
    Re=V*roc*Wing.chord_mac/nu
    Re_ref= 9*10**6
    op_Clmax= Cl_max_ref*(Re/Re_ref)**0.1
    
    #wing cl_max to outer panel Cl_max

    w_Clmax= op_Clmax* (0.919729714285715  -0.044504761904771*taper   -0.001835900000000*sweep +  0.247071428571446*taper**2 +  0.003191500000000*taper*sweep  -0.000056632142857*sweep**2   -0.279166666666676*taper**3 +  0.002300000000000*taper**2*sweep + 0.000049982142857*taper*sweep**2  -0.000000280000000* sweep**3)
   
    #---FAR stall speed effect---------------
    
    Cl_max_FAA= 1.1*w_Clmax
    
    #-----------wing mounted engine ----
    
    Cl_max_w_eng= Cl_max_FAA-0.2
    
    #----Cl_max_slat increment-------------
    
    dcl_slat=slat_lift(slat_angle, sweep)
    
    #-----Cl_max_flap_increment-------------
    
    dcl_flap=flap_lift(t_c,flap_chord,flap_angle,sweep,Sref,Swf)
    
    #--------effect of Mach number--------------
    
    Cl_max_ls= Cl_max_w_eng + dcl_slat + dcl_flap
    M_d=Mcr*numpy.cos(sweep)/numpy.cos(24.5*numpy.pi/180)
    
    
    
    Cd_ind = 0.01
    
    return Cl_max_ls, Cd_ind


