## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# BET_calculations.py
# 
# Created:  Jan 2022, R. Erhard
# Modified: May 2022, E. Botero

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import jax.numpy as jnp
from jax import jit


# ----------------------------------------------------------------------
#   Compute Airfoil Aerodynamics
# ----------------------------------------------------------------------

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
            sub_cl = interp2d(Re, alpha, cl.RE_data, cl.aoa_data, cl.CL_data)
            sub_cd = interp2d(Re, alpha, cd.RE_data, cd.aoa_data, cd.CD_data)
            Cl    = jnp.where(aloc==jj,sub_cl,Cl)
            Cdval = jnp.where(aloc==jj,sub_cd,Cdval)

    else:
        # Estimate Cl max
        Cl_max_ref = jnp.atleast_2d(-0.0009*tc**3 + 0.0217*tc**2 - 0.0442*tc + 0.7005).T
        Re_ref     = 9.*10**6
        Cl1maxp    = Cl_max_ref * ( Re / Re_ref ) **0.1

        # If not airfoil polar provided, use 2*pi as lift curve slope
        Cl = 2.*jnp.pi*alpha

        # By 90 deg, it's totally stalled.
        Cl = jnp.minimum(Cl, Cl1maxp)
        Cl = jnp.where(alpha>=jnp.pi/2,0,Cl)

        # Scale for Mach, this is Karmen_Tsien
        Cl = jnp.where(Ma[:,:]<1.,Cl/((1-Ma*Ma)**0.5+((Ma*Ma)/(1+(1-Ma*Ma)**0.5))*Cl/2),Cl)

        #This is an atrocious fit of DAE51 data at RE=50k for Cd
        Cdval = (0.108*(Cl*Cl*Cl*Cl)-0.2612*(Cl*Cl*Cl)+0.181*(Cl*Cl)-0.0139*Cl+0.0278)*((50000./Re)**0.2)
        Cdval = jnp.where(alpha>=jnp.pi/2,2,Cdval)    

        # prevent zero Cl to keep Cd/Cl from breaking in BET
        Cl = jnp.where(Cl==0,1e-6,Cl)
    return Cl, Cdval, alpha, Ma, W


@jit
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
    
    lamdaw = jnp.array(r*Wa/(R*Wt))
    lamdaw = jnp.where(lamdaw<=0.,0,lamdaw)
    f      = (B/2.)*(R/r - 1.)/lamdaw
    f      = jnp.where(f<=0.,0,f)
    piece  = jnp.exp(-f)

    lamdaw            = jnp.array(r*Wa/(R*Wt))
    lamdaw            = jnp.where(lamdaw<=0.,1e-12,lamdaw) # Zero causes Nans when differentiated

    Rtip = R
    et1, et2, et3, maxat = 1,1,1,-jnp.inf
    tipfactor = jnp.array( B/2.0*(  (Rtip/r)**et1 - 1  )**et2/lamdaw**et3)
    tipfactor = jnp.where(tipfactor<=0,0,tipfactor) # This is also needed
    Ftip      = jnp.where(tipfactor<=0,0,2.*jnp.arccos(jnp.exp(-tipfactor))/jnp.pi) # this extra where is for grad to keep from nan-ing    
    
    F = Ftip

    return lamdaw, F, piece

@jit
def interp2d(x,y,xp,yp,zp,fill_value= None):
    """
    Bilinear interpolation on a grid. ``CartesianGrid`` is much faster if the data
    lies on a regular grid.
    Args:
        x, y: 1D arrays of point at which to interpolate. Any out-of-bounds
            coordinates will be clamped to lie in-bounds.
        xp, yp: 1D arrays of points specifying grid points where function values
            are provided.
        zp: 2D array of function values. For a function `f(x, y)` this must
            satisfy `zp[i, j] = f(xp[i], yp[j])`
    Returns:
        1D array `z` satisfying `z[i] = f(x[i], y[i])`.
    """
    #if xp.ndim != 1 or yp.ndim != 1:
        #raise ValueError("xp and yp must be 1D arrays")
    #if zp.shape != (xp.shape + yp.shape):
        #raise ValueError("zp must be a 2D array with shape xp.shape + yp.shape")

    ix = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
    iy = jnp.clip(jnp.searchsorted(yp, y, side="right"), 1, len(yp) - 1)

    # Using Wikipedia's notation (https://en.wikipedia.org/wiki/Bilinear_interpolation)
    z_11 = zp[ix - 1, iy - 1]
    z_21 = zp[ix, iy - 1]
    z_12 = zp[ix - 1, iy]
    z_22 = zp[ix, iy]

    z_xy1 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_11 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_21
    z_xy2 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_12 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_22

    z = (yp[iy] - y) / (yp[iy] - yp[iy - 1]) * z_xy1 + (y - yp[iy - 1]) / (
        yp[iy] - yp[iy - 1]
    ) * z_xy2

    if fill_value is not None:
        oob = jnp.logical_or(
            x < xp[0], jnp.logical_or(x > xp[-1], jnp.logical_or(y < yp[0], y > yp[-1]))
        )
        z = jnp.where(oob, fill_value, z)

    return z