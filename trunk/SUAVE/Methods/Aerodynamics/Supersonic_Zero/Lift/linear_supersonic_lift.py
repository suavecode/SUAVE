# linear_supersonic_lift.py
# 
# Created:  Jun 2014, T. Macdonald
# Modified: Jan 2016, E. Botero


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

def linear_supersonic_lift(conditions,configuration,wing):
    """ Computes lift using linear supersonic theory
        Adapted from vortex lattice code to strip theory

        Inputs:
            wing - geometry dictionary with fields:
            Sref - reference area

        Outputs:
            Cl - lift coefficient

        Assumptions:
            - Reference area of the passed wing is the desired 
            reference area for Cl
            - Compressibility effects are not handled
        
    """

    # Unpack
    span        = wing.spans.projected
    root_chord  = wing.chords.root
    tip_chord   = wing.chords.tip
    sweep       = wing.sweeps.quarter_chord
    taper       = wing.taper
    twist_rc    = wing.twists.root
    twist_tc    = wing.twists.tip
    sym_para    = wing.symmetric
    AR          = wing.aspect_ratio
    Sref        = wing.areas.reference
    orientation = wing.vertical

    aoa = conditions.aerodynamics.angle_of_attack
    
    n   = configuration.number_panels_spanwise
    
    # chord difference
    dchord = (root_chord-tip_chord)
    
    # Check if the wing is symmetric
    # If so, reduce the span by half for calculations
    if sym_para is True :
        span=span/2
        
    # Width of strips
    deltax = span/n

    if orientation == False : # No lift for vertical surfaces

        # Intialize arrays with number of strips
        section_length = np.empty(n)
        area_section   = np.empty(n)
        twist_distri   = np.empty(n)
        
        # Discretize the wing sections into strips
        for i in range(0,n):
    
            section_length[i] = dchord/span*(span-(i+1)*deltax+deltax/2) + tip_chord
            area_section[i]   = section_length[i]*deltax
            twist_distri[i]   = twist_rc + i/float(n)*(twist_tc-twist_rc)
        
        # Initialize variables
        area_tot = 0.0        
        cl_tot_base = 0.0
        cl = np.array([0.0]*n)
        
        for j in range(0,n):
            # Note that compressibility effects are not included here
            cl[j] = 4*(aoa-twist_distri[j])*area_section[j]
            area_tot = area_tot+area_section[j]
            cl_tot_base = cl_tot_base + cl[j]
    
        Cl=cl_tot_base/area_tot # Lift 
    
    else:
        
        Cl= 0.0       

    return Cl

# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__': 
    
    span       = 10.0
    root_chord = 10.0
    tip_chord  = 10.0
    sweep      = 0.0
    taper      = 1.0
    twist_rc   = 0.0
    twist_tc   = 0.1
    sym_para   = True
    AR         = 1.0
    Sref       = 100.0
    orientation = False
    
    aoa = 0.1
    
    n  = 160
    
    # chord difference
    dchord=(root_chord-tip_chord)
    
    # Check if the wing is symmetric
    # If so, reduce the span by half for calculations
    if sym_para is True :
        span=span/2
        
    # Width of strips
    deltax=span/n

    if orientation == False : # No lift for vertical surfaces

        # Intialize arrays with number of strips
        section_length= np.empty(n)
        area_section=np.empty(n)
        twist_distri=np.empty(n)
        
        # Discretize the wing sections into strips
        for i in range(0,n):
    
            section_length[i]= dchord/span*(span-(i+1)*deltax+deltax/2) + tip_chord
            area_section[i]=section_length[i]*deltax
            twist_distri[i]=twist_rc + i/float(n)*(twist_tc-twist_rc)
        
        # Initialize variables
        area_tot = 0.0        
        cl_tot_base = 0.0
        cl = np.array([0.0]*n)
        
        for j in range(0,n):
            # Note that compressibility effects are not included here
            cl[j] = 4*(aoa-twist_distri[j])*area_section[j]
            area_tot = area_tot+area_section[j]
            cl_tot_base = cl_tot_base + cl[j]
    
        Cl=cl_tot_base/area_tot # Lift 
    
    else:
        
        Cl= 0.0       

    print Cl