## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
# weissinger_vortex_lattice.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Apr 2017, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import autograd.numpy as np 

# ----------------------------------------------------------------------
#  Weissinger Vortex Lattice
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
def weissinger_vortex_lattice(conditions,configuration,wing):
    """Uses the vortex lattice method to compute the lift coefficient and induced drag component

    Assumptions:
    None

    Source:
    Unknown

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
    AR          = wing.aspect_ratio
    Sref        = wing.areas.reference
    orientation = wing.vertical

    n  = configuration.number_panels_spanwise
    nn = configuration.number_panels_chordwise

    # conditions
    aoa = conditions.aerodynamics.angle_of_attack
    
    # chord difference
    dchord=(root_chord-tip_chord)
    if sym_para is True :
        span=span/2
    deltax=span/n

    if orientation == False :

        # discretizing the wing sections into panels            
        i              = np.arange(0,n)
        section_length = dchord/span*(span-(i+1)*deltax+deltax/2) + tip_chord
        area_section   = section_length[i]*deltax
        sl             = section_length[i]
        twist_distri   = twist_rc + i/float(n)*(twist_tc-twist_rc)
        xpos           = (i)*deltax
        
        ya = np.atleast_2d((i)*deltax)
        yb = np.atleast_2d((i+1)*deltax)
        xa = np.atleast_2d(((i+1)*deltax-deltax/2)*np.tan(sweep) + 0.25*sl)
        x  = np.atleast_2d(((i+1)*deltax-deltax/2)*np.tan(sweep) + 0.75*sl)
        y  = np.atleast_2d(((i+1)*deltax-deltax/2))
        
        xloc_leading  = ((i+1)*deltax)*np.tan(sweep)
        xloc_trailing = ((i+1)*deltax)*np.tan(sweep) + sl        
                
        RHS  = np.atleast_2d(np.sin(twist_distri+aoa)).T
        A = (whav(x,y,xa.T,ya.T)-whav(x,y,xa.T,yb.T)\
            -whav(x,y,xa.T,-ya.T)+whav(x,y,xa.T,-yb.T))*0.25/np.pi
    
        # Vortex strength computation by matrix inversion
        T = np.linalg.solve(A.T,RHS)
        
        # Calculating the effective velocty         
        A_v = A*0.25/np.pi*T
        v   = np.sum(A_v,axis=0)
        
        Lfi = -T.T * (np.sin(twist_tc)-v)
        Lfk =  T.T * np.cos(twist_tc)   
        Lft = -Lfi*np.sin(twist_tc)+Lfk*np.cos(twist_tc)
        Dg  = Lfi*np.cos(twist_tc)+Lfk*np.sin(twist_tc)
            
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