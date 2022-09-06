## @ingroup Analyses-Propulsion
# Rotor_Wake_Fidelity_Two.py
#
# Created:  Jan 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component

import numpy as np
from scipy.interpolate import interp1d
# ----------------------------------------------------------------------
#  Generalized Rotor Class
# ----------------------------------------------------------------------
## @ingroup Analyses-Propulsion
class Rotor_Wake_External(Energy_Component):
    """This is a general rotor wake component requiring external inflow specifications. 

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

        self.tag             = 'rotor_wake'
        self.wake_method     = 'Fid2'

        self.external_inflow = Data()
        self.external_inflow.disc_axial_induced_velocity = None
        self.external_inflow.disc_tangential_induced_velocity = None

    
    def evaluate(self,rotor,U,Ua,Ut,PSI,omega,beta,c,r,R,B,a,nu,a_loc,a_geo,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc,use_2d_analysis,conditions):
        """
        Wake evaluation is performed using an externally-applied inflow field at the rotor.
        This requires an external solver to generate the inflow to the rotor, which must have been appended to the 
        rotor wake as wake.external_flow.va and wake.external_flow.vt. This is then used within the BET.
           
           
        Assumptions:
           Assumes external inflow is provided with shape (Nr_external,Na_external), where Nr_external is the number
           of radial stations at which the inflow is provided and Na_external is the number of azimuthal stations.
           
           Assumes external radial stations matches the rotor radial stations
           Assumes azimuth stations are evenly spaced 
           
        Outputs of this function include the inflow velocities induced by rotor wake:
           va  - axially-induced velocity from rotor wake
           vt  - tangentially-induced velocity from rotor wake
        
        """
        va_external = self.external_inflow.disc_axial_induced_velocity
        vt_external = self.external_inflow.disc_tangential_induced_velocity
        if (va_external is not None) and (vt_external is not None):
            va = np.atleast_3d(va_external.T).T
            vt = np.atleast_3d(vt_external.T).T
            
            # interpolate the inflow to obtain values at each rotor station
            inflow_shape     = np.shape(va)
            rotor_disc_shape = (1,len(rotor.radius_distribution),rotor.number_azimuthal_stations)
            
            Na          = rotor_disc_shape[2]
            Na_external = inflow_shape[2]
            
            if Na_external != Na:
                # interpolate along azimuth direction:
                external_psi = np.linspace(0, 2*np.pi, Na_external+1)[:-1]
                psi          = np.linspace(0, 2*np.pi, Na+1)[:-1]
                
                va_interp = interp1d(external_psi, va)
                vt_interp = interp1d(external_psi, vt)
                va        = va_interp(psi)
                vt        = vt_interp(psi)

        else:
            va = 0
            vt = 0  
            print("External inflow not specified for rotor wake Fidelity Two!")
      
        

        
        return va, vt