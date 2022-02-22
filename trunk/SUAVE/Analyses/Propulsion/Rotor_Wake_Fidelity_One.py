## @ingroup Analyses-Propulsion
# Rotor_Wake_Fidelity_One.py
#
# Created:  Jan 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Analyses.Propulsion.Rotor_Wake_Fidelity_Zero import Rotor_Wake_Fidelity_Zero
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_fidelity_one_inflow_velocities import compute_fidelity_one_inflow_velocities
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.generate_fidelity_one_wake_shape import generate_fidelity_one_wake_shape
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.BET_calculations import compute_inflow_and_tip_loss


# package imports
import numpy as np
import copy

# ----------------------------------------------------------------------
#  Generalized Rotor Class
# ----------------------------------------------------------------------
## @ingroup Analyses-Propulsion
class Rotor_Wake_Fidelity_One(Energy_Component):
    """ SUAVE.Analyses.Propulsion.Rotor_Wake_Fidelity_One()
    
    The Fidelity One Rotor Wake Class
    Uses a semi-prescribed vortex wake (PVW) model of the rotor wake

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

        self.tag                        = 'rotor_wake'
        self.wake_method                = 'Fidelity_One'
        self.wake_vortex_distribution   = Data()
        self.wake_method_fidelity       = 0
        self.semi_prescribed_converge   = False      # flag for convergence on semi-prescribed wake shape
        self.vtk_save_flag              = False      # flag for saving vtk outputs of wake
        self.vtk_save_loc               = None       # location to save vtk outputs of wake
        
        self.wake_settings              = Data()
        self.wake_settings.number_rotor_rotations     = 5
        self.wake_settings.number_steps_per_rotation  = 72
        self.wake_settings.initial_timestep_offset    = 0    # initial timestep
        
        # wake convergence criteria
        self.maximum_convergence_iteration            = 10
        self.axial_velocity_convergence_tolerance     = 1e-2
        
        # flags for slipstream interaction
        self.slipstream                 = False
        self.verbose                    = True
        
    def initialize(self,rotor,conditions):
        """
        Initializes the rotor by evaluating the BET once. This is required for generating the 
        circulation strengths for the vortex distribution in the prescribed vortex wake, and the 
        initial wake shape, which relies on the axial inflow induced by the wake at the rotor disc.
        
        Assumptions:
        None

        Source:
        N/A

        Inputs:
           self         - rotor wake
           rotor        - SUAVE rotor
           conditions   - conditions
           
           
        Outputs:
        None
        
        Properties Used:
        None
        
        """
        # run the BET once using fidelity zero inflow
        rotor_temp = copy.deepcopy(rotor)
        rotor_temp.Wake = Rotor_Wake_Fidelity_Zero()
        _,_,_,_,outputs,_ = rotor_temp.spin(conditions)
        
        rotor.outputs = outputs
        
        # match the azimuthal discretization betwen rotor and wake
        if self.wake_settings.number_steps_per_rotation  != rotor.number_azimuthal_stations:
            self.wake_settings.number_steps_per_rotation = rotor.number_azimuthal_stations
            
            if self.verbose:
                print("Wake azimuthal discretization does not match rotor discretization. \
                Resetting wake to match rotor of Na="+str(rotor.number_azimuthal_stations))
        
        return
    
    def evaluate(self,rotor,wake_inputs,conditions):
        """
        Wake evaluation is performed using a semi-prescribed vortex wake (PVW) method for Fidelity One.
        
        Assumptions:
        None

        Source:
        N/A

        Inputs:
           self         - rotor wake
           rotor        - SUAVE rotor
           wake_inputs.
              Ua        - Axial velocity
              Ut        - Tangential velocity
              r         - radius distribution
           conditions   - conditions
           
           
        Outputs:
           va  - axially-induced velocity from rotor wake
           vt  - tangentially-induced velocity from rotor wake
        
        Properties Used:
        None
        """   
        
        # Unpack inputs
        Ua = wake_inputs.velocity_axial
        Ut = wake_inputs.velocity_tangential
        r  = wake_inputs.radius_distribution
        
        R  = rotor.tip_radius
        B  = rotor.number_of_blades
        
        # Initialize rotor with single pass of VW 
        self.initialize(rotor,conditions)
        
        # converge on va for a semi-prescribed wake method
        va_diff, ii = 1, 0
        tol = self.axial_velocity_convergence_tolerance
        if self.semi_prescribed_converge:
            if self.verbose:
                print("\tConverging on semi-prescribed wake shape...")
            ii_max = self.maximum_convergence_iteration
        else:
            if self.verbose:
                print("\tGenerating fully-prescribed wake shape...")
            ii_max = 1
        

        while va_diff > tol:  
            # generate wake geometry for rotor
            WD  = generate_fidelity_one_wake_shape(self,rotor)
            
            # compute axial wake-induced velocity (a byproduct of the circulation distribution which is an input to the wake geometry)
            va, vt = compute_fidelity_one_inflow_velocities(self,rotor, WD)

            # compute new blade velocities
            Wa   = va + Ua
            Wt   = Ut - vt

            lamdaw, F, _ = compute_inflow_and_tip_loss(r,R,Wa,Wt,B)

            va_diff = np.max(abs(F*va - rotor.outputs.disc_axial_induced_velocity))

            # update the axial disc velocity based on new va from HFW
            rotor.outputs.disc_axial_induced_velocity = F*va 
            
            ii+=1
            if ii>=ii_max and va_diff>tol:
                if self.semi_prescribed_converge and self.verbose:
                    print("Semi-prescribed vortex wake did not converge on axial inflow used for wake shape.")
                break

            
        # save converged wake:
        WD  = generate_fidelity_one_wake_shape(self,rotor)
        self.vortex_distribution = WD
            
        return va, vt
    

    
    def shift_wake_VD(self,wVD, offset):
        for mat in wVD.keys():
            if 'X' in mat:
                wVD[mat] += offset[0]
            elif 'Y' in mat:
                wVD[mat] += offset[1]
            elif 'Z' in mat:
                wVD[mat] += offset[2]
        for mat in wVD.reshaped_wake.keys():
            if 'X' in mat:
                wVD.reshaped_wake[mat] += offset[0]
            elif 'Y' in mat:
                wVD.reshaped_wake[mat] += offset[1]
            elif 'Z' in mat:
                wVD.reshaped_wake[mat] += offset[2]        
        # update wake distribution
        self.wake_vortex_distribution = wVD
        self.vortex_distribution = wVD
        return
        
        





