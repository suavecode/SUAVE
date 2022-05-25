## @ingroup Analyses-Propulsion
# Rotor_Wake_Fidelity_One.py
#
# Created:  Jan 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data
from SUAVE.Components import Wings
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Analyses.Propulsion.Rotor_Wake_Fidelity_Zero import Rotor_Wake_Fidelity_Zero
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.fidelity_one_wake_convergence import fidelity_one_wake_convergence
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_wake_induced_velocity import compute_wake_induced_velocity

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.extract_wing_VD import extract_wing_collocation_points

# package imports
import copy
import numpy as np
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
        self.vortex_distribution        = Data()
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
        self.verbose                    = False
        
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
        
        # Initialize rotor with single pass of VW 
        self.initialize(rotor,conditions)
        
        # Converge on the Fidelity-One rotor wake shape
        WD, va, vt = fidelity_one_wake_convergence(self,rotor,wake_inputs)
        
        # Store wake shape
        self.vortex_distribution = WD
            
        return va, vt
    
    def evaluate_slipstream(self,rotor,geometry,ctrl_pts):
        """
        Evaluates the velocities induced by the rotor on a specified wing of the vehicle.
        If no wing instance is specified, uses main wing or last available wing in geometry.
        
        Assumptions:
        None

        Source:
        N/A

        Inputs:
           self         - rotor wake
           rotor        - rotor
           geometry     - vehicle geometry
           
        Outputs:
           wake_V_ind   - induced velocity from rotor wake at (VD.XC, VD.YC, VD.ZC)
        
        Properties Used:
        None
        """
        # Check if any wings are specified as interaction components
        try:
            wing_instance_tags = self.interaction_components
        except:
            wing_instance_tags=None
            
            
        # If no wing is specified, default to the main wing
        if wing_instance_tags == None:
            nmw = 0
            # check for main wing
            for i,wing in enumerate(geometry.wings):
                if not isinstance(wing,Wings.Main_Wing): continue
                nmw +=1                
                wing_instance = wing
                wing_instance_idx = i
            if nmw == 1:
                if self.verbose:
                    print("No wing specified for slipstream analysis. Using '"+ wing_instance.tag +"'.")
                pass
            elif nmw>1:
                if self.verbose:
                    print("No wing specified for slipstream analysis. Multiple main wings in vehicle, using '"+ wing_instance.tag +"'.")             
            else:
                if self.verbose:
                    print("No wing specified for slipstream analysis. No main wing defined, using '"+ wing.tag +"'.")
                wing_instance = wing 
                
                
            # Isolate the VD components corresponding to this wing instance
            wing_CPs, slipstream_vd_ids = extract_wing_collocation_points(geometry, wing_instance_idx)
            
            # Evaluate rotor slipstream effect on specified wing instance
            rot_V_wake_ind = self.evaluate_wake_velocities(rotor, wing_CPs, ctrl_pts)
            
            # Expand
            wake_V_ind = np.zeros((ctrl_pts,geometry.vortex_distribution.n_cp,3))
            wake_V_ind[:,slipstream_vd_ids,:] = rot_V_wake_ind  
            
        else:
            #Check for VLM_wings to include control surfaces
            if "VLM_wings" in geometry.vortex_distribution.keys():
                wings = geometry.vortex_distribution.VLM_wings
            else:
                wings = geometry.wings
    
            # Initialize influences            
            rot_V_wake_ind = []
            wake_V_ind = np.zeros((ctrl_pts,geometry.vortex_distribution.n_cp,3))
            for i,w in enumerate(wings):
                
                if w.tag in wing_instance_tags:
                    wing_instance_idx = i     
                    
                    # Isolate the VD components corresponding to this wing instance
                    wing_CPs, slipstream_vd_ids = extract_wing_collocation_points(geometry, wing_instance_idx)
                   
                    if self.verbose:
                        print("Computing effect of '"+rotor.tag+"' on '"+w.tag+"'.")
                    # Evaluate rotor slipstream effect on specified wing instance
                    rot_V_wake_ind = self.evaluate_wake_velocities(rotor, wing_CPs, ctrl_pts)
                    
                    # Add effects
                    wake_V_ind[:,slipstream_vd_ids,:] += rot_V_wake_ind                    
            
        return wake_V_ind 
    
    def evaluate_wake_velocities(self,rotor,VD,num_ctrl_pts):
        """
        Links the rotor wake to compute the wake-induced velocities at the vortex distribution
        control points.
        
        Assumptions:
        None

        Source:
        N/A

        Inputs:
           self         - rotor wake
           rotor        - rotor
           VD           - vortex distribution
           num_ctrl_pts - number of analysis control points
           
        Outputs:
           prop_V_wake_ind  - induced velocity from rotor wake at (VD.XC, VD.YC, VD.ZC)
        
        Properties Used:
        None
        """           
        #extract wake shape previously generated
        wake_vortex_distribution = rotor.Wake.vortex_distribution
    
        # compute the induced velocity from the rotor wake on the lifting surfaces
        VD.Wake         = wake_vortex_distribution
        start_angle_idx = rotor.start_angle_idx
        rot_V_wake_ind  = compute_wake_induced_velocity(wake_vortex_distribution,VD,num_ctrl_pts,azi_start_idx=start_angle_idx)        
        
        return rot_V_wake_ind
    
    def shift_wake_VD(self,wVD, offset):
        """
        This shifts the wake by the (x,y,z) coordinates of the offset. 
        This is useful for rotors with identical wakes that can be reused and shifted without regeneration.
        
        Assumptions
        None
        
        Source:
        N/A
        
        Inputs:
        wVD    - wake vortex distribution
        offset - (x,y,z) offset distances
        
        Outputs
        None
        
        Properties Used
        None
        
        """
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
        
        





