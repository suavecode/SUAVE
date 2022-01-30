## @ingroup Analyses-Propulsion
# Rotor_Wake_Fidelity_Zero.py
#
# Created:  Jun 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Components.Energy.Converters.Rotor \
     import compute_airfoil_aerodynamics,compute_inflow_and_tip_loss,compute_dR_dpsi

# package imports
import numpy as np
import copy

# ----------------------------------------------------------------------
#  Generalized Rotor Class
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
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

        self.tag                          = 'rotor_wake'


    
    def evaluate(self,U,Ua,Ut,PSI,omega,beta,c,r,R,B,a,nu,a_loc,a_geo,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc,use_2d_analysis):
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
        # BEMT Iteration
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

            # If omega = 0, do not run BEMT convergence loop
            if all(omega[:,0]) == 0. :
                break

            # If its really not going to converge
            if np.any(PSI>np.pi/2) and np.any(dpsi>0.0):
                print("Rotor BEMT did not converge to a solution (Stall)")
                break

            ii+=1
            if ii>10000:
                print("Rotor BEMT did not converge to a solution (Iteration Limit)")
                break    
                
            
        return va, vt




