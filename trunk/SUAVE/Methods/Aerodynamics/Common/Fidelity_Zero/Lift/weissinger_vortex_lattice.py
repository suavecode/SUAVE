## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# weissinger_vortex_lattice.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Apr 2017, T. MacDonald
#           Oct 2017, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np 

# ----------------------------------------------------------------------
#  Weissinger Vortex Lattice
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def weissinger_vortex_lattice(conditions,configuration,wing):
    """Uses the vortex lattice method to compute the lift coefficient and induced drag component

    Assumptions:
    None

    Source:
    An Introduction to Theoretical and Computational Aerodynamics by Jack Moran

    Inputs:
    wing.
      spans.projected                       [m]
      chords.root                           [m]
      chords.tip                            [m]
      sweeps.quarter_chord                  [radians]
      taper                                 [Unitless]
      twists.root                           [radians]
      twists.tip                            [radians]
      symmetric                             [Boolean]
      aspect_ratio                          [Unitless]
      areas.reference                       [m^2]
      vertical                              [Boolean]
    configuration.number_panels_spanwise    [Unitless]
    configuration.number_panels_chordwise   [Unitless]
    conditions.aerodynamics.angle_of_attack [radians]

    Outputs:
    Cl                                      [Unitless]
    Cd                                      [Unitless]

    Properties Used:
    N/A
    """ 

    #unpack
    span        = wing.spans.projected
    root_chord  = wing.chords.root
    tip_chord   = wing.chords.tip
    sweep       = wing.sweeps.quarter_chord
    taper       = wing.taper
    twist_rc    = wing.twists.root
    twist_tc    = wing.twists.tip
    sym_para    = wing.symmetric
    Sref        = wing.areas.reference
    orientation = wing.vertical

    n  = configuration.number_panels_spanwise

    # conditions
    aoa = conditions.aerodynamics.angle_of_attack
    
    # chord difference
    dchord = (root_chord-tip_chord)
    if sym_para is True :
        span = span/2
        
    deltax = span/n
    
    sin_aoa = np.sin(aoa)
    cos_aoa = np.cos(aoa)

    if orientation == False :

        # discretizing the wing sections into panels            
        i              = np.arange(0,n)
        section_length = dchord/span*(span-(i+1)*deltax+deltax/2) + tip_chord
        twist_distri   = twist_rc + i/float(n)*(twist_tc-twist_rc)
        
        ya = np.atleast_2d((i)*deltax)
        yb = np.atleast_2d((i+1)*deltax)
        xa = np.atleast_2d(((i+1)*deltax-deltax/2)*np.tan(sweep) + 0.25*section_length)
        x  = np.atleast_2d(((i+1)*deltax-deltax/2)*np.tan(sweep) + 0.75*section_length)
        y  = np.atleast_2d(((i+1)*deltax-deltax/2))      
                
        RHS  = np.atleast_2d(np.sin(twist_distri+aoa))
        
        A = (whav(x,y,xa.T,ya.T)-whav(x,y,xa.T,yb.T)\
            -whav(x,y,xa.T,-ya.T)+whav(x,y,xa.T,-yb.T))*0.25/np.pi
    
        # Vortex strength computation by matrix inversion
        T = np.linalg.solve(A.T,RHS.T).T
        
        # Calculating the effective velocty         
        A_v = A*0.25/np.pi*T
        v   = np.sum(A_v,axis=1)
        
        Lfi = -T * (sin_aoa-v)
        Lfk =  T * cos_aoa 
        Lft = -Lfi * sin_aoa + Lfk * cos_aoa
        Dg  =  Lfi * cos_aoa + Lfk * sin_aoa
            
        L  = deltax * Lft
        D  = deltax * Dg
        
        # Total lift
        LT = np.sum(L)
        DT = np.sum(D)
    
        Cl = 2*LT/(0.5*Sref)
        Cd = 2*DT/(0.5*Sref)     
    
    else:
        
        Cl = 0.0
        Cd = 0.0         

    return Cl, Cd

# ----------------------------------------------------------------------
#   Helper Functions
# ----------------------------------------------------------------------
def whav(x1,y1,x2,y2):
    """ Helper function of vortex lattice method      
        Inputs:
            x1,x2 -x coordinates of bound vortex
            y1,y2 -y coordinates of bound vortex

        Outpus:
            Cl_comp - lift coefficient
            Cd_comp - drag  coefficient       

        Assumptions:
            if needed

    """    

    use_base    = 1 - np.isclose(x1,x2)*1.
    no_use_base = np.isclose(x1,x2)*1.
    
    whv = 1/(y1-y2)*(1+ (np.sqrt((x1-x2)**2+(y1-y2)**2)/(x1-x2)))*use_base + (1/(y1 -y2))*no_use_base
    
    return whv