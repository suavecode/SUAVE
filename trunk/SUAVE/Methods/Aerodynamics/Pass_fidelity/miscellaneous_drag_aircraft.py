# miscellaneous_drag_aircraft.py
# 
# Created:  Your Name, Dec 2013
# Modified:         

""" miscellaneous_drag_aircraft(wing,segment)

    Computes the miscellaneous drag based on a set of fits
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

#def cdp_misc(sweep_w, sweep_h, sweep_v, d_engexit,Sref,Mc,roc,muc ,Tc,S_affected_w,S_affected_h,S_affected_v):
def miscellaneous_drag_aircraft(aircraft,segment):
    """ SUAVE.Methods.miscellaneous_drag_aircraft(Wing,segment)
        computes the miscellaneous drag associated with an aircraft
        
        Inputs:
            aircraft- An aircraft object is passed in
            segment - the segment object contains information regarding the mission segment
            Cl - wing Cl
        Outpus:
            cd_misc  - returns the miscellaneous drag assoicated with the wing
            
            >> try to minimize outputs
            >> pack up outputs into Data() if needed
        
        Assumptions:
            if needed
        
    """

    # unpack inputs
    
#-------------control surface gap drag-----------------------
    #f_gaps_w=0.0002*(numpy.cos(sweep_w))**2*S_affected_w
    #f_gaps_h=0.0002*(numpy.cos(sweep_h))**2*S_affected_h
    #f_gaps_v=0.0002*(numpy.cos(sweep_v))**2*S_affected_v

    #f_gapst=f_gaps_w+f_gaps_h+f_gaps_v




    #--compute this correctly         
    cd_gaps=0.0001

    #------------Nacelle base drag--------------------------------
    no_of_propulsors=len(aircraft.Propulsors)
    cd_nacelle_base=np.empty(no_of_propulsors)
    cd_nacelle_base_tot = 0.0
    for k in range(len(aircraft.Propulsors)):
        
        cd_nacelle_base[k]=0.5/12*np.pi*aircraft.Propulsors[k].nacelle_dia*0.2/aircraft.Sref
        cd_nacelle_base_tot =  cd_nacelle_base_tot+cd_nacelle_base[k]

    #-------fuselage upsweep drag----------------------------------
    cd_upsweep = 0.006/aircraft.Sref

    #-------------miscellaneous drag -------------
    #increment by 1.5# of parasite drag
    #---------------------induced  drag-----------------------------

    #cd_trim = 0.015*Cd_airplane
    #cd_trim=0.015*0.015;   #1-2% of airplane drag
    

    cd_misc = cd_gaps+cd_nacelle_base_tot+cd_upsweep  #+cd_trim
   
    return cd_misc
    