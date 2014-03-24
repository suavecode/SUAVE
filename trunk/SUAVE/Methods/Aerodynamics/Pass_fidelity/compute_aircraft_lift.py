# compute_aircraft_lift.py
# 
# Created:  Your Name, Dec 2013
# Modified:         

""" compute_aircraft_lift(wing,segment)
    """
import numpy
import SUAVE
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

def compute_aircraft_lift(aircraft,segment,alpha):
    """ SUAVE.Methods.Aerodynamics.compute_aircraft_lift(aircraft,segment,alpha)
        computes the lift associated with an aircraft 
        
        Inputs:
            aircraft- An aircraft object is passed in
            segment - the segment object contains information regarding the mission segment
            alpha- angle of attack
        
        Outpus:
            aircraft_lift_total - the total aircraft lift
            cl_w - array of cl aircraft
            cd_w - array of aircraft drag
            
    
        Assumptions:
            if needed
        
    """    
   
    
    #compute effeciency factor for the wings and the drag
    no_of_wings=len(aircraft.Wings)
    cl_w=numpy.empty(no_of_wings)
    cd_w=numpy.empty(no_of_wings)
    n=5
    nn=1

    aircraft_lift_total_uncorrected=0.0
    
    #---look at vertical tail
    
    for k in range(len(aircraft.Wings)):
        
        if aircraft.Wings[k].hl == 1: 
            [cl_w[k],cd_w[k]]=SUAVE.Methods.Aerodynamics.Pass_fidelity.high_lift(aircraft.Wings[k],segment,aircraft.Sref)
        else :   
            [cl_w[k],cd_w[k]]=SUAVE.Methods.Aerodynamics.Pass_fidelity.Weissinger_vortex_lattice(aircraft.Wings[k],segment,aircraft.Sref,alpha,n,nn)
        
        aircraft_lift_total_uncorrected=aircraft_lift_total_uncorrected+cl_w[k]

    aircraft_lift_total = aircraft_lift_total_uncorrected*1.14 #correction effect-document

    return aircraft_lift_total,cl_w,cd_w

