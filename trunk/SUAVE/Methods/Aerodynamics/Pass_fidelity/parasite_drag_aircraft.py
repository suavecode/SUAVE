# parasite_drag_wing.py
# 
# Created:  Your Name, Dec 2013
# Modified:         

""" parasite_drag_aircraft(aircraft,segment,Cl,cdi_inv,cdp,fd_ws)

    Computes the parasite drag of the entire aircraft based on a set of fits
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
import numpy #as np
import scipy as sp
import SUAVE

def parasite_drag_aircraft(aircraft,segment):   
#def parasite_drag_aircraft(aircraft,segment,Cl,cdi_inv,cdp,fd_ws):
    """ SUAVE.Methods.parasite_drag_aircraft(aircraft,segment,Cl,cdi_inv,cdp,fd_ws)
        computes the parasite_drag_aircraftassociated with an aircraft 
        
        Inputs:
            aircraft- An aircraft object is passed in
            segment - the segment object contains information regarding the mission segment
            Cl - wing Cl
            cdi_inv -  inviscid drag component computed from the vortex lattice
            cdp - parasite drag
            fd_ws - 
        
        Outpus:
            cd_p  - returns the parasite drag assoicated with the wing
            
            >> try to minimize outputs
            >> pack up outputs into Data() if needed
        
        Assumptions:
            if needed
        
    """

    # unpack inputs
    
    no_of_wings=len(aircraft.Wings)
    wings=numpy.empty(no_of_wings)
    
    no_of_fuselages=len(aircraft.Fuselages)
    fuselages=numpy.empty(no_of_fuselages)    
    
    parasite_drag_wing_values=numpy.empty(no_of_wings)
    parasite_drag_fuselage_values=numpy.empty(no_of_fuselages)
    miscellaneous_drag_aircraft_value=0.0
    
    parasite_drag_total=0.0
    
    for k in range(len(aircraft.Wings)):
        
        #wings[k]=aircraft.Wings[k]
        parasite_drag_wing_values[k]= SUAVE.Methods.Aerodynamics.Pass_fidelity.parasite_drag_wing(aircraft.Wings[k], segment,aircraft.Sref)      
        parasite_drag_total=parasite_drag_total + parasite_drag_wing_values[k]
        
        #Wing1=state.config.Wings[0]
        #Wing2=state.config.Wings[1]
        #Wing3=state.config.Wings[2]                                        
        
    for k in range(len(aircraft.Fuselages)):
        
        #fuselages[k]=aircraft.Fuselages[k]  
        parasite_drag_fuselage_values[k]=  SUAVE.Methods.Aerodynamics.Pass_fidelity.parasite_drag_fuselage(aircraft.Fuselages[k], segment, aircraft.Sref)      
        parasite_drag_total=parasite_drag_total + parasite_drag_fuselage_values[k]
        
        
        #l_fus=state.config.Fuselages[0].length_cabin
        #d_fus=state.config.Fuselages[0].width
        #l_nose=state.config.Fuselages[0].length_nose
        #l_tail=state.config.Fuselages[0].length_tail
        
        
        
        #miscellaneous drag computation
        miscellaneous_drag_aircraft_value=SUAVE.Methods.Aerodynamics.Pass_fidelity.miscellaneous_drag_aircraft(aircraft,segment)
        
        parasite_drag_total=parasite_drag_total + miscellaneous_drag_aircraft_value
        
        
        return parasite_drag_total