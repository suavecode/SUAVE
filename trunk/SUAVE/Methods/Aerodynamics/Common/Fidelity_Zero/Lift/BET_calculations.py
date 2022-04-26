## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# BET_calculations.py
# 
# Created:  Jan 2022, R. Erhard
# Modified:       

import numpy as np
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

    alpha    = beta - np.arctan2(Wa,Wt)
    W        = (Wa*Wa + Wt*Wt)**0.5
    Ma       = W/a
    Re       = (W*c)/nu

    # If propeller airfoils are defined, use airfoil surrogate
    if a_loc != None:
        # Compute blade Cl and Cd distribution from the airfoil data
        dim_sur = len(cl_sur)
        # return the 2D Cl and CDval of shape (ctrl_pts, Nr, Na)
        Cl      = np.zeros((ctrl_pts,Nr,Na))
        Cdval   = np.zeros((ctrl_pts,Nr,Na))
        for jj in range(dim_sur):
            Cl_af           = cl_sur[a_geo[jj]]((Re,alpha))
            Cdval_af        = cd_sur[a_geo[jj]]((Re,alpha))
            locs            = np.where(np.array(a_loc) == jj )
            Cl[:,locs,:]    = Cl_af[:,locs,:]
            Cdval[:,locs,:] = Cdval_af[:,locs,:]

    else:
        # Estimate Cl max
        Cl_max_ref = np.atleast_2d(-0.0009*tc**3 + 0.0217*tc**2 - 0.0442*tc + 0.7005).T
        Re_ref     = 9.*10**6
        Cl1maxp    = Cl_max_ref * ( Re / Re_ref ) **0.1

        # If not airfoil polar provided, use 2*pi as lift curve slope
        Cl = 2.*np.pi*alpha

        # By 90 deg, it's totally stalled.
        Cl[Cl>Cl1maxp]  = Cl1maxp[Cl>Cl1maxp] # This line of code is what changed the regression testing
        Cl[alpha>=np.pi/2] = 0.

        # Scale for Mach, this is Karmen_Tsien
        Cl[Ma[:,:]<1.] = Cl[Ma[:,:]<1.]/((1-Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])**0.5+((Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])/(1+(1-Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])**0.5))*Cl[Ma<1.]/2)

        # If the blade segments are supersonic, don't scale
        Cl[Ma[:,:]>=1.] = Cl[Ma[:,:]>=1.]

        #This is an atrocious fit of DAE51 data at RE=50k for Cd
        Cdval = (0.108*(Cl*Cl*Cl*Cl)-0.2612*(Cl*Cl*Cl)+0.181*(Cl*Cl)-0.0139*Cl+0.0278)*((50000./Re)**0.2)
        Cdval[alpha>=np.pi/2] = 2.


    # prevent zero Cl to keep Cd/Cl from breaking in BET
    Cl[Cl==0] = 1e-6

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
    lamdaw            = r*Wa/(R*Wt)
    lamdaw[lamdaw<0.] = 0.
    f                 = (B/2.)*(1.-r/R)/lamdaw
    f[f<0.]           = 0.
    
    piece             = np.exp(-f)
    F                 = 2.*np.arccos(piece)/np.pi

    Rtip = R
    et1, et2, et3, maxat = 1,1,1,-np.inf
    tipfactor = B/2.0*(  (Rtip/r)**et1 - 1  )**et2/lamdaw**et3
    tipfactor[tipfactor<0.]   = 0.
    Ftip = 2.*np.arccos(np.exp(-tipfactor))/np.pi
    
    F = Ftip
    

    return lamdaw, F, piece