## @defgroup Methods-Propulsion-Rotor_Wake-Fidelity_Zero
# fidelity_zero_wake_convergence.py
#
# Created:  Feb 2022, R. Erhard
# Modified: 

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.BET_calculations import compute_airfoil_aerodynamics,compute_inflow_and_tip_loss
import numpy as np
import scipy as sp
import copy

## @defgroup Methods-Propulsion-Rotor_Wake-Fidelity_Zero
def fidelity_zero_wake_convergence(wake,rotor,wake_inputs):
    """
    Wake evaluation is performed using a simplified vortex wake method for Fidelity Zero, 
    following Helmholtz vortex theory.
    
    Assumptions:
    None

    Source:
    Drela, M. "Qprop Formulation", MIT AeroAstro, June 2006
    http://web.mit.edu/drela/Public/web/qprop/qprop_theory.pdf

    Inputs:
       self         - rotor wake
       rotor        - SUAVE rotor
       wake_inputs.
          Ua        - Axial velocity
          Ut        - Tangential velocity
          r         - radius distribution
       
       
    Outputs:
       va  - axially-induced velocity from rotor wake
       vt  - tangentially-induced velocity from rotor wake
    
    Properties Used:
    None
    
    """        
    
    # Unpack some wake inputs
    ctrl_pts        = wake_inputs.ctrl_pts
    Nr              = wake_inputs.Nr
    Na              = wake_inputs.Na    

    if wake_inputs.use_2d_analysis:
        PSI    = np.ones((ctrl_pts,Nr,Na))
    else:
        PSI     = np.ones((ctrl_pts,Nr))

    PSI_final,infodict,ier,msg = sp.optimize.fsolve(iteration,PSI,args=(wake_inputs,rotor),full_output = 1)
    
    if ier!=1:
        print("Rotor BEVW did not converge to a solution (Stall)")
    
    # Calculate the velocities given PSI
    va, vt = va_vt(PSI_final, wake_inputs, rotor)

    
    return va, vt


## @defgroup Methods-Propulsion-Rotor_Wake-Fidelity_Zero
def iteration(PSI, wake_inputs, rotor):
    """
    Computes the BEVW iteration.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
       B                          number of rotor blades                          [-]
       beta                       blade twist distribution                        [-]
       r                          radius distribution                             [m]
       R                          tip radius                                      [m]
       Wt                         tangential velocity                             [m/s]
       Wa                         axial velocity                                  [m/s]
       U                          total velocity                                  [m/s]
       Ut                         tangential velocity                             [m/s]
       Ua                         axial velocity                                  [m/s]
       cos_psi                    cosine of the inflow angle PSI                  [-]
       sin_psi                    sine of the inflow angle PSI                    [-]
       piece                      output of a step in tip loss calculation        [-]

    Outputs:
       dR_dpsi                    derivative of residual wrt inflow angle         [-]

    """    
    
    # Unpack inputs to rotor wake fidelity zero
    U               = wake_inputs.velocity_total
    Ua              = wake_inputs.velocity_axial
    Ut              = wake_inputs.velocity_tangential
    use_2d_analysis = wake_inputs.use_2d_analysis        
    beta            = wake_inputs.twist_distribution
    c               = wake_inputs.chord_distribution
    r               = wake_inputs.radius_distribution
    a               = wake_inputs.speed_of_sounds
    nu              = wake_inputs.dynamic_viscosities
    ctrl_pts        = wake_inputs.ctrl_pts
    Nr              = wake_inputs.Nr
    Na              = wake_inputs.Na

    # Unpack rotor data        
    R        = rotor.tip_radius
    B        = rotor.number_of_blades    
    tc       = rotor.thickness_to_chord
    a_geo    = rotor.airfoil_geometry
    a_loc    = rotor.airfoil_polar_stations
    cl_sur   = rotor.airfoil_cl_surrogates
    cd_sur   = rotor.airfoil_cd_surrogates    
    
    # Reshape PSI because the solver gives it flat
    if wake_inputs.use_2d_analysis:
        PSI    = np.reshape(PSI,(ctrl_pts,Nr,Na))
    else:
        PSI    = np.reshape(PSI,(ctrl_pts,Nr))
    
    # compute velocities
    sin_psi      = np.sin(PSI)
    cos_psi      = np.cos(PSI)
    Wa           = 0.5*Ua + 0.5*U*sin_psi
    Wt           = 0.5*Ut + 0.5*U*cos_psi
    vt           = Ut - Wt

    # compute blade airfoil forces and properties
    Cl, Cdval, alpha, Ma, W = compute_airfoil_aerodynamics(beta,c,r,R,B,Wa,Wt,a,nu,a_loc,a_geo,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc,use_2d_analysis)

    # compute inflow velocity and tip loss factor
    lamdaw, F, piece = compute_inflow_and_tip_loss(r,R,Wa,Wt,B)

    # compute Newton residual on circulation
    Gamma       = vt*(4.*np.pi*r/B)*F*(1.+(4.*lamdaw*R/(np.pi*B*r))*(4.*lamdaw*R/(np.pi*B*r)))**0.5
    Rsquiggly   = Gamma - 0.5*W*c*Cl
    
    return Rsquiggly.flatten()#, va, vt


## @defgroup Methods-Propulsion-Rotor_Wake-Fidelity_Zero
def va_vt(PSI, wake_inputs, rotor):
    """
    Computes the inflow velocities

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
       B                          number of rotor blades                          [-]
       beta                       blade twist distribution                        [-]
       r                          radius distribution                             [m]
       R                          tip radius                                      [m]
       Wt                         tangential velocity                             [m/s]
       Wa                         axial velocity                                  [m/s]
       U                          total velocity                                  [m/s]
       Ut                         tangential velocity                             [m/s]
       Ua                         axial velocity                                  [m/s]
       cos_psi                    cosine of the inflow angle PSI                  [-]
       sin_psi                    sine of the inflow angle PSI                    [-]
       piece                      output of a step in tip loss calculation        [-]

    Outputs:
       dR_dpsi                    derivative of residual wrt inflow angle         [-]

    """    
    
    # Unpack inputs to rotor wake fidelity zero
    U               = wake_inputs.velocity_total
    Ua              = wake_inputs.velocity_axial
    Ut              = wake_inputs.velocity_tangential
    ctrl_pts        = wake_inputs.ctrl_pts
    Nr              = wake_inputs.Nr
    Na              = wake_inputs.Na
    
    # Reshape PSI because the solver gives it flat
    if wake_inputs.use_2d_analysis:
        PSI    = np.reshape(PSI,(ctrl_pts,Nr,Na))
    else:
        PSI    = np.reshape(PSI,(ctrl_pts,Nr))
    
    # compute velocities
    sin_psi      = np.sin(PSI)
    cos_psi      = np.cos(PSI)
    Wa           = 0.5*Ua + 0.5*U*sin_psi
    Wt           = 0.5*Ut + 0.5*U*cos_psi
    va           = Wa - Ua
    vt           = Ut - Wt

    return va, vt

## @defgroup Methods-Propulsion-Rotor_Wake-Fidelity_Zero
def compute_dR_dpsi(B,beta,r,R,Wt,Wa,U,Ut,Ua,cos_psi,sin_psi,piece):
    """
    Computes the analytical derivative for the BEVW iteration.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
       B                          number of rotor blades                          [-]
       beta                       blade twist distribution                        [-]
       r                          radius distribution                             [m]
       R                          tip radius                                      [m]
       Wt                         tangential velocity                             [m/s]
       Wa                         axial velocity                                  [m/s]
       U                          total velocity                                  [m/s]
       Ut                         tangential velocity                             [m/s]
       Ua                         axial velocity                                  [m/s]
       cos_psi                    cosine of the inflow angle PSI                  [-]
       sin_psi                    sine of the inflow angle PSI                    [-]
       piece                      output of a step in tip loss calculation        [-]

    Outputs:
       dR_dpsi                    derivative of residual wrt inflow angle         [-]

    """
    # An analytical derivative for dR_dpsi used in the Newton iteration for the BEVW
    # This was solved symbolically in Matlab and exported
    pi          = np.pi
    pi2         = np.pi**2
    BB          = B*B
    BBB         = BB*B
    f_wt_2      = 4*Wt*Wt
    f_wa_2      = 4*Wa*Wa
    arccos_piece = np.arccos(piece)
    Ucospsi     = U*cos_psi
    Usinpsi     = U*sin_psi
    Utcospsi    = Ut*cos_psi
    Uasinpsi    = Ua*sin_psi
    UapUsinpsi  = (Ua + Usinpsi)
    utpUcospsi  = (Ut + Ucospsi)
    utpUcospsi2 = utpUcospsi*utpUcospsi
    UapUsinpsi2 = UapUsinpsi*UapUsinpsi
    dR_dpsi     = ((4.*U*r*arccos_piece*sin_psi*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5))/B -
                   (pi*U*(Ua*cos_psi - Ut*sin_psi)*(beta - np.arctan((Wa+Wa)/(Wt+Wt))))/(2.*(f_wt_2 + f_wa_2)**(0.5))
                   + (pi*U*(f_wt_2 +f_wa_2)**(0.5)*(U + Utcospsi  +  Uasinpsi))/(2.*(f_wa_2/(f_wt_2) + 1.)*utpUcospsi2)
                   - (4.*U*piece*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5)*(R - r)*(Ut/2. -
                    (Ucospsi)/2.)*(U + Utcospsi + Uasinpsi ))/(f_wa_2*(1. - np.exp(-(B*(Wt+Wt)*(R -
                    r))/(r*(Wa+Wa))))**(0.5)) + (128.*U*r*arccos_piece*(Wa+Wa)*(Ut/2. - (Ucospsi)/2.)*(U +
                    Utcospsi  + Uasinpsi ))/(BBB*pi2*utpUcospsi*utpUcospsi2*((16.*f_wa_2)/(BB*pi2*f_wt_2) + 1.)**(0.5)))

    dR_dpsi[np.isnan(dR_dpsi)] = 0.1
    return dR_dpsi