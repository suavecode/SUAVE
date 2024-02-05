## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# BET_calculations.py
# 
# Created:  Jan 2022, R. Erhard
# Modified:       
from SUAVE.Core.Utilities import interp2d 
import numpy as np
## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def compute_airfoil_aerodynamics(beta,c,r,R,B,Wa,Wt,a,nu,airfoils,a_loc,ctrl_pts,Nr,Na,tc,use_2d_analysis):
    """
    Cl, Cdval = compute_airfoil_aerodynamics( beta,c,r,R,B,
                                              Wa,Wt,a,nu,
                                              airfoils,a_loc
                                              ctrl_pts,Nr,Na,tc,use_2d_analysis )

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
       airfoil_data               Data structure of airfoil polar information     [-]
       ctrl_pts                   Number of control points                        [-]
       Nr                         Number of radial blade sections                 [-]
       Na                         Number of azimuthal blade stations              [-]
       tc                         Thickness to chord                              [-]
       use_2d_analysis            Specifies 2d disc vs. 1d single angle analysis  [Boolean]

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
        if use_2d_analysis:
            # return the 2D Cl and CDval of shape (ctrl_pts, Nr, Na)
            Cl      = np.zeros((ctrl_pts,Nr,Na))
            Cdval   = np.zeros((ctrl_pts,Nr,Na))
            for jj,airfoil in enumerate(airfoils):
                pd              = airfoil.polars
                Cl_af           = interp2d(Re,alpha,pd.reynolds_numbers, pd.angle_of_attacks, pd.lift_coefficients) 
                Cdval_af        = interp2d(Re,alpha,pd.reynolds_numbers, pd.angle_of_attacks, pd.drag_coefficients)
                locs            = np.where(np.array(a_loc) == jj )
                Cl[:,locs,:]    = Cl_af[:,locs,:]
                Cdval[:,locs,:] = Cdval_af[:,locs,:]
        else:
            # return the 1D Cl and CDval of shape (ctrl_pts, Nr)
            Cl      = np.zeros((ctrl_pts,Nr))
            Cdval   = np.zeros((ctrl_pts,Nr))

            for jj,airfoil in enumerate(airfoils):
                pd            = airfoil.polars
                Cl_af         = interp2d(Re,alpha,pd.reynolds_numbers, pd.angle_of_attacks, pd.lift_coefficients)
                Cdval_af      = interp2d(Re,alpha,pd.reynolds_numbers, pd.angle_of_attacks, pd.drag_coefficients)
                locs          = np.where(np.array(a_loc) == jj )
                Cl[:,locs]    = Cl_af[:,locs]
                Cdval[:,locs] = Cdval_af[:,locs]
    else:
        # Estimate Cl max
        tc_1 = tc*100
        Cl_max_ref = -0.0009*tc_1**3 + 0.0217*tc_1**2 - 0.0442*tc_1 + 0.7005
        Cl_max_ref[Cl_max_ref<0.7] = 0.7
        Re_ref     = 9.*10**6
        Cl1maxp    = Cl_max_ref * ( Re / Re_ref ) **0.1

        # If not airfoil polar provided, use 2*pi as lift curve slope
        Cl = 2.*np.pi*alpha

        # By 90 deg, it's totally stalled.
        Cl[Cl>Cl1maxp]  = Cl1maxp[Cl>Cl1maxp] # This line of code is what changed the regression testing
        Cl[alpha>=np.pi/2] = 0.

        # Scale for Mach, this is Karmen_Tsien
        KT_cond = np.logical_and((Ma[:,:]<1.),(Cl>0))
        Cl[KT_cond] = Cl[KT_cond]/((1-Ma[KT_cond]*Ma[KT_cond])**0.5+((Ma[KT_cond]*Ma[KT_cond])/(1+(1-Ma[KT_cond]*Ma[KT_cond])**0.5))*Cl[KT_cond]/2)

        # If the blade segments are supersonic, don't scale
        Cl[Ma[:,:]>=1.] = Cl[Ma[:,:]>=1.]

        #This is an atrocious fit of DAE51 data at RE=50k for Cd
        Cdval = (0.108*(Cl*Cl*Cl*Cl)-0.2612*(Cl*Cl*Cl)+0.181*(Cl*Cl)-0.0139*Cl+0.0278)*((50000./Re)**0.2)
        Cdval[alpha>=np.pi/2] = 2.


    # prevent zero Cl to keep Cd/Cl from breaking in BET
    Cl[Cl==0] = 1e-6

    return Cl, Cdval, alpha, Ma, W



def compute_inflow_and_tip_loss(r,R,Rh,Wa,Wt,B,et1=1,et2=1,et3=1):
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
       et1        tuning parameter for tip loss function 
       et2        tuning parameter for tip loss function 
       et3        tuning parameter for tip loss function 
       
    Outputs:               
       lamdaw     inflow ratio                                                     [-]
       F          tip loss factor                                                  [-]
       piece      output of a step in tip loss calculation (needed for residual)   [-]
    """

    lamdaw             = r*Wa/(R*Wt)
    lamdaw[lamdaw<=0.] = 1e-12

    tipfactor = B/2.0*(  (R/r)**et1 - 1  )**et2/lamdaw**et3 
    hubfactor = B/2.0*(  (r/Rh)**et1 - 1  )**et2/lamdaw**et3 

    tippiece = np.exp(-tipfactor)
    hubpiece = np.exp(-hubfactor)
    Ftip = 2.*np.arccos(tippiece)/np.pi  
    Fhub = 2.*np.arccos(hubpiece)/np.pi  
    
    piece = tippiece
    piece[tippiece<1e-3] = hubpiece[tippiece<1e-3]
    
    F = Ftip * Fhub
    F[F<1e-6] = 1e-6
    return lamdaw, F, piece