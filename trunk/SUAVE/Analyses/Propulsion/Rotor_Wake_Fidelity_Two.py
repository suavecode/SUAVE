## @ingroup Analyses-Propulsion
# Rotor_Wake_Fidelity_Two.py
#
# Created:  Jan 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Generalized Rotor Class
# ----------------------------------------------------------------------
## @ingroup Analyses-Propulsion
class Rotor_Wake_Fidelity_Two(Energy_Component):
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
        self.wake_method    = 'Fid2'

    
    def evaluate(self,rotor,U,Ua,Ut,PSI,omega,beta,c,r,R,B,a,nu,a_loc,a_geo,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc,use_2d_analysis,conditions):
        """
        Wake evaluation is performed using an externally-applied inflow field at the rotor.
        This requires an external solver to generate the inflow to the rotor, which must have been appended to the 
        rotor wake as wake.external_flow.va and wake.external_flow.vt. This is then used within the BET.
           
        Outputs of this function include the inflow velocities induced by rotor wake:
           va  - axially-induced velocity from rotor wake
           vt  - tangentially-induced velocity from rotor wake
        
        """
        
        try:
                va = self.external_inflow.va[None,:,:] 
                vt = self.external_inflow.vt[None,:,:]    
            except:
                va = 0
                vt = 0
                print("No external inflow specified! No inflow velocity used.")
            
        return va, vt