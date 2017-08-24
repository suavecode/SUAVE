## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
# wing_compressibility_correction.py
# 
# Created:  Aug 2017, E. Botero
# Modified: 
#           

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  The Function
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
def lifting_line(state,settings,geometry):
    """

    Assumptions:
    subsonic and unswept

    Source:
    Traub, L. W., Botero, E., Waghela, R., Callahan, R., & Watson, A. (2015). Effect of Taper Ratio at Low Reynolds Number. Journal of Aircraft.
    
    Inputs:
    N/A

    Outputs:
    N/A

    Properties Used:
    N/A
    """  
    
    # Unpack first round:
    orientation = wing.vertical    
    
    # Don't bother doing the calculation if it is a vertical tail
    if orientation == True:
        Cl = 0.0
        Cd = 0.0
        return Cl, Cd
    else:
        continue
        

    
    # Potentially useful unpacks
    sweep       = wing.sweeps.quarter_chord
    sym_para    = wing.symmetric
    AR          = wing.aspect_ratio

    
    
    # unpack
    b     = wing.spans.projected # Wingspan
    S     = wing.areas.reference # Reference area
    r     = 20   # Need to set somewhere, this will be a setting
    rho   = None # Freestream density
    V     = None # Freestream velocity
    mu    = None # Freestream viscosity
    cla   = None # 2-D lift curve slope
    azl   = None # 2-D 
    alpha = None

    
    N      = r-1                  # number of spanwise divisions
    n      = np.linspace(0,N,N)   # vectorize
    thetan = n*np.pi/r            # angular stations
    yn     = -b*np.cos(thetan)/2. # y locations based on the angular spacing
    etan   = np.abs(2.*yn/b)       # normalized coordinates
    
    # Project the spanwise y locations into the chords
    # If spanwise stations are setup
    if len(wing.Segments.keys())>0:
 
        c    = None
        ageo = None
    
    # Spanwise stations are not setup
    else:
        # Use the taper ratio to determine the chord distribution
        # Use the geometric twist applied to the ends to    
        
        # unpack
        taper       = wing.taper
        tip_twist   = wing.twists.root
        root_twist  = wing.twists.tip 
        root_chord  = wing.chords.root
        tip_chord   = wing.chords.tip  
        
        c    = root_chord+root_chord*(taper-1.)*yn
        ageo = (tip_twist-root_twist)*yn+root_twist
        

    # Setup for loops to be removed later, both can be vectorized
    A = np.zeros((N,N))
    for ii in xrange(0,N):
        for jj in xrange(0,N):
            RHS[ii,jj] = np.sin(ii*thetan[jj])*(np.sin(thetan)+ii*c[jj]*cla[jj]/(4.*b))
            
    LHS = np.zeros(N)
    for ii in xrange(0,N):
        LHS[ii] = (c[ii]*cla[ii]*np.sin(thetan[ii])*(alpha+ageo[ii]-azl[ii]))/(4.*b)
        
    A = np.linalg.solve(RHS,LHS)
    
    #clcb = np.zeros(N)
    #for ii in xrange(0,N):
        #clcb = 4. * A[ii] * np.sin(ii*thetan(ii))
        
    CD = 0.0
    CL = 0.0
    
    
   
    return CL, CD