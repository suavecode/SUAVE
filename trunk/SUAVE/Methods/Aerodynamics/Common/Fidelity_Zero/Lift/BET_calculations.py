## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# BET_calculations.py
# 
# Created:  Jan 2022, R. Erhard
# Modified:       

import numpy as np
import jax.numpy as jnp

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def compute_airfoil_aerodynamics(beta,c,r,R,B,Wa,Wt,a,nu,a_loc,a_geo,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc):
    """
    Cl, Cdval = compute_airfoil_aerodynamics( beta,c,r,R,B,
                                              Wa,Wt,a,nu,
                                              a_loc,a_geo,cl_sur,cd_sur,
                                              ctrl_pts,Nr,Na,tc )

    Computes the aerodynamic forces at sectional blade locations. If airfoil
    geometry and locations are specified, the forces are computed using the
    airfoil polar lift and drag surrogates, accounting for the local Reynolds
    number and local angle of attack.

    If the airfoils are not specified, an approximation is used.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
       beta                       blade twist distribution                        [-]
       c                          chord distribution                              [-]
       r                          radius distribution                             [-]
       R                          tip radius                                      [-]
       B                          number of rotor blades                          [-]

       Wa                         axial velocity                                  [-]
       Wt                         tangential velocity                             [-]
       a                          speed of sound                                  [-]
       nu                         viscosity                                       [-]

       a_loc                      Locations of specified airfoils                 [-]
       a_geo                      Geometry of specified airfoil                   [-]
       cl_sur                     Lift Coefficient Surrogates                     [-]
       cd_sur                     Drag Coefficient Surrogates                     [-]
       ctrl_pts                   Number of control points                        [-]
       Nr                         Number of radial blade sections                 [-]
       Na                         Number of azimuthal blade stations              [-]
       tc                         Thickness to chord                              [-]

    Outputs:
       Cl                       Lift Coefficients                         [-]
       Cdval                    Drag Coefficients  (before scaling)       [-]
       alpha                    section local angle of attack             [rad]

    """

    alpha    = beta - jnp.arctan2(Wa,Wt)
    W        = (Wa*Wa + Wt*Wt)**0.5
    Ma       = W/a
    Re       = (W*c)/nu
    

    # If propeller airfoils are defined, use airfoil surrogate
    if a_loc != None:
        aloc  = jnp.atleast_3d(jnp.array(a_loc))
        aloc  = jnp.broadcast_to(aloc,jnp.shape(Wa))
        # Compute blade Cl and Cd distribution from the airfoil data
        # return the 2D Cl and CDval of shape (ctrl_pts, Nr, Na)
        Cl      = jnp.zeros(jnp.shape(Wa))
        Cdval   = jnp.zeros(jnp.shape(Wa))
        for jj, (cl, cd) in enumerate(zip(cl_sur,cd_sur)):
            Cl    = jnp.where(aloc==jj,cl((Re,alpha)),Cl)
            Cdval = jnp.where(aloc==jj,cd((Re,alpha)),Cdval)

    else:
        # Estimate Cl max
        Cl_max_ref = jnp.atleast_2d(-0.0009*tc**3 + 0.0217*tc**2 - 0.0442*tc + 0.7005).T
        Re_ref     = 9.*10**6
        Cl1maxp    = Cl_max_ref * ( Re / Re_ref ) **0.1

        # If not airfoil polar provided, use 2*pi as lift curve slope
        Cl = 2.*np.pi*alpha
    
        # By 90 deg, it's totally stalled.
        Cl = jnp.minimum(Cl, Cl1maxp)
        Cl = jnp.where(alpha>=jnp.pi/2,0,Cl)
    
        # Scale for Mach, this is Karmen_Tsien
        Cl = jnp.where(Ma[:,:]<1.,Cl/((1-Ma*Ma)**0.5+((Ma*Ma)/(1+(1-Ma*Ma)**0.5))*Cl/2),Cl)
        
        #This is an atrocious fit of DAE51 data at RE=50k for Cd
        Cdval = (0.108*(Cl*Cl*Cl*Cl)-0.2612*(Cl*Cl*Cl)+0.181*(Cl*Cl)-0.0139*Cl+0.0278)*((50000./Re)**0.2)
        Cdval = jnp.where(alpha>=np.pi/2,2,Cdval)    
    
        # prevent zero Cl to keep Cd/Cl from breaking in BET
        Cl = jnp.where(Cl==0,1e-6,Cl)

    return Cl, Cdval, alpha, Ma, W



def compute_inflow_and_tip_loss(r,R,Wa,Wt,B):
    """
    Computes the inflow, lamdaw, and the tip loss factor, F.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
       r          radius distribution                                              [m]
       R          tip radius                                                       [m]
       Wa         axial velocity                                                   [m/s]
       Wt         tangential velocity                                              [m/s]
       B          number of rotor blades                                           [-]
                 
    Outputs:               
       lamdaw     inflow ratio                                                     [-]
       F          tip loss factor                                                  [-]
       piece      output of a step in tip loss calculation (needed for residual)   [-]
    """
    print('Starting inflow and tip loss')
    
    lamdaw            = jnp.array(r*Wa/(R*Wt))
    lamdaw            = jnp.where(lamdaw<0.,0,lamdaw)
    f                 = (B/2.)*(1.-r/R)/lamdaw
    f                 = jnp.where(f<0.,0,f)
    
    piece             = jnp.exp(-f)
    F                 = 2.*jnp.arccos(piece)/jnp.pi

    Rtip = R
    et1, et2, et3, maxat = 1,1,1,-jnp.inf
    tipfactor = jnp.array( B/2.0*(  (Rtip/r)**et1 - 1  )**et2/lamdaw**et3)
    tipfactor = jnp.where(tipfactor<0,0,tipfactor)
    Ftip = 2.*jnp.arccos(jnp.exp(-tipfactor))/jnp.pi
    
    F = Ftip
    
    print('Finished inflow and tip loss')
    

    return lamdaw, F, piece