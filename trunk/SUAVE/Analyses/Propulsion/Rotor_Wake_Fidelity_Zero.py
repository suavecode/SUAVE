## @ingroup Analyses-Propulsion
# Rotor_Wake_Fidelity_Zero.py
#
# Created:  Jan 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.BET_calculations \
     import compute_airfoil_aerodynamics,compute_inflow_and_tip_loss

# package imports
import numpy as np
import copy

# ----------------------------------------------------------------------
#  Generalized Rotor Class
# ----------------------------------------------------------------------
## @ingroup Analyses-Propulsion
class Rotor_Wake_Fidelity_Zero(Energy_Component):
    """This is a general rotor wake component. 

    Assumptions:
    None

    Source:
    None
    """
    def __defaults__(self):
        """This sets the default values for the component to function.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """

        self.tag            = 'rotor_wake'
        self.wake_method    = 'VW'

    
    def evaluate(self,rotor,U,Ua,Ut,PSI,omega,beta,c,r,R,B,a,nu,a_loc,a_geo,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc,use_2d_analysis,conditions):
        """
        Wake evaluation is performed using a simplified vortex wake (VW) method for Fidelity Zero.
        
           
        Outputs of this function include the inflow velocities induced by rotor wake:
           va  - axially-induced velocity from rotor wake
           vt  - tangentially-induced velocity from rotor wake
        
        """
        
        # Simplified vortex formulation
        # Setup a Newton iteration
        diff   = 1.
        tol    = 1e-6  # Convergence tolerance
        ii     = 0
        PSIold = copy.deepcopy(PSI)*0
        # BEVW Iteration
        while (diff>tol):
            # compute velocities
            sin_psi      = np.sin(PSI)
            cos_psi      = np.cos(PSI)
            Wa           = 0.5*Ua + 0.5*U*sin_psi
            Wt           = 0.5*Ut + 0.5*U*cos_psi
            va           = Wa - Ua
            vt           = Ut - Wt

            # compute blade airfoil forces and properties
            Cl, Cdval, alpha, Ma, W = compute_airfoil_aerodynamics(beta,c,r,R,B,Wa,Wt,a,nu,a_loc,a_geo,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc,use_2d_analysis)

            # compute inflow velocity and tip loss factor
            lamdaw, F, piece = compute_inflow_and_tip_loss(r,R,Wa,Wt,B)

            # compute Newton residual on circulation
            Gamma       = vt*(4.*np.pi*r/B)*F*(1.+(4.*lamdaw*R/(np.pi*B*r))*(4.*lamdaw*R/(np.pi*B*r)))**0.5
            Rsquiggly   = Gamma - 0.5*W*c*Cl

            # use analytical derivative to get dR_dpsi
            dR_dpsi = compute_dR_dpsi(B,beta,r,R,Wt,Wa,U,Ut,Ua,cos_psi,sin_psi,piece)

            # update inflow angle
            dpsi        = -Rsquiggly/dR_dpsi
            PSI         = PSI + dpsi
            diff        = np.max(abs(PSIold-PSI))
            PSIold      = PSI

            # If omega = 0, do not run BEVW convergence loop
            if all(omega[:,0]) == 0. :
                break

            # If its really not going to converge
            if np.any(PSI>np.pi/2) and np.any(dpsi>0.0):
                print("Rotor BEVW did not converge to a solution (Stall)")
                break

            ii+=1
            if ii>10000:
                print("Rotor BEVW did not converge to a solution (Iteration Limit)")
                break    
                
            
        return va, vt
    



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
