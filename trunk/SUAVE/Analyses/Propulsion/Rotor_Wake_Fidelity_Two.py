## @ingroup Analyses-Propulsion
# Rotor_Wake_Fidelity_Two.py
#
# Created:  Jun 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data
from SUAVE.Analyses.Propulsion.Rotor_Wake_Fidelity_One import Rotor_Wake_Fidelity_One
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.fidelity_one_wake_convergence import fidelity_one_wake_convergence
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_Two.update_wake_position import update_wake_position
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_Two.update_wake_positions2 import update_wake_position2

from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_fidelity_one_inflow_velocities import compute_fidelity_one_inflow_velocities
from DCode.Common.Visualization_Tools.box_contour_field_vtk import box_contour_field_vtk
#from DCode.Common.generalFunctions import save_single_prop_vehicle_vtk

# package imports
import copy
import numpy as np
import pickle
# ----------------------------------------------------------------------
#  Generalized Rotor Class
# ----------------------------------------------------------------------
## @ingroup Analyses-Propulsion
class Rotor_Wake_Fidelity_Two(Rotor_Wake_Fidelity_One):
    """ SUAVE.Analyses.Propulsion.Rotor_Wake_Fidelity_Two()
    
    The Fidelity Two Rotor Wake Class
    Uses a free vortex wake (FVW) method of modeling the rotor wake

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
        self.wake_method                = 'Fidelity_Two'
        self.vortex_distribution        = Data()
        self.wake_method_fidelity       = 2
        self.vtk_save_flag              = False      # flag for saving vtk outputs of wake
        self.vtk_save_loc               = None       # location to save vtk outputs of wake
        self.restart_file               = None       # file of initial wake instance to use if specified
        
        self.wake_settings              = Data()
        self.wake_settings.number_rotor_rotations     = 5
        self.wake_settings.number_steps_per_rotation  = 72
        self.wake_settings.initial_timestep_offset    = 0    # initial timestep
        
        # wake convergence criteria
        self.maximum_convergence_iteration            = 10
        self.axial_velocity_convergence_tolerance     = 1e-2
        self.influencing_vortex_distribution          = None  # any additional vortex distribution elements influencing wake development (ie. vehicle.vortex_distribution)
        
        # flags for slipstream interaction
        self.slipstream                 = False
        self.verbose                    = True
    
    def initialize(self, rotor, conditions):
        """
        Initializes the rotor wake with a Fidelity One wake.
        
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
        # Initialize with fidelity one wake
        rotor_temp = copy.deepcopy(rotor)
        rotor_temp.Wake = Rotor_Wake_Fidelity_One()
        _,_,_,_,outputs,_ = rotor_temp.spin(conditions)
        
        rotor.outputs = outputs
        rotor.Wake.vortex_distribution = rotor_temp.Wake.vortex_distribution
        
        return
        
        
    def evaluate(self,rotor,wake_inputs,conditions,VD=None):
        """
        Wake evaluation is performed using a free vortex wake (FVW) method for Fidelity Two.
        
        Assumptions:
        Periodic boundary conditions with rotational frequency of the rotor.

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
        
        # Initialize rotor with PVW unless restart file specified
        if self.restart_file is not None:
            # load restart file
            with open(self.restart_file, 'rb') as file:
                data = pickle.load(file)   
            
            # update rotor and wake initialization
            rotor.outputs = data.prop.outputs
            rotor.Wake.vortex_distribution = data.prop.Wake.vortex_distribution
            self.vortex_distribution = rotor.Wake.vortex_distribution
        else:
            self.initialize(rotor,conditions)
        
        # evolve the wake with itself over time, generating a force-free wake
        self, rotor, interpolatedBoxData = self.evolve_wake_vortex_distribution(rotor,conditions,VD=self.influencing_vortex_distribution)
        
        # compute wake-induced velocities
        va, vt = compute_fidelity_one_inflow_velocities(self,rotor)   
            
        return va, vt, rotor
    
    def evolve_wake_vortex_distribution(self,rotor,conditions,VD=None):
        """
        Time-evolves the wake under its own wake distribution (self.vortex_distribution) and any external
        vortex distribution (VD).
        
        """
        diff = 10
        tol = 1e-3       # converged when delta in wake geoemtry is less than 1mm 
        iteration = 0
        max_iter = 1#10    # max number of iterations for wake convergence

        #print("\n\nQUASI FREE VORTEX WAKE SINGLE PASS:\n")  
        print("\n\nQUASI FREE VORTEX WAKE CONVERGENCE:\n")        
        while diff >= tol and iteration <= max_iter:

            # ---- DEBUG -----------------------------------------------------------------------
            # ----------------------------------------------------------------------------------
            # save vortex vtk for this iteration
            #save_single_prop_vehicle_vtk(rotor, iteration=iteration, save_loc="/Users/rerha/Desktop/test_relaxed_wake/convergenceLoop/")   
            # ----------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------        
            
            prior_VD = copy.deepcopy(self.vortex_distribution)
            
            # Update the position of each vortex filament due to component interactions
            self, rotor, interpolatedBoxData = update_wake_position(self,rotor,conditions,VD)
            #self, rotor, interpolatedBoxData = update_wake_position2(self,rotor,conditions,VD)
            
            # Compute residual between wake shapes
            keys = ['XA1', 'XA2', 'YA1', 'YA2', 'ZA1', 'ZA2', 'XB1', 'XB2', 'YB1', 'YB2', 'ZB1', 'ZB2']
            max_diff = 0.
            for key in keys:
                max_diff =  max(max_diff, np.max(abs(prior_VD.reshaped_wake[key] - self.vortex_distribution.reshaped_wake[key])))
    
    
            # ---- DEBUG -----------------------------------------------------------------------
            # ----------------------------------------------------------------------------------
            #save the contour box velocity field for new wake
            ##stateData = Data()
            ##stateData.vFreestream = conditions.freestream.velocity
            ##stateData.alphaDeg = rotor.orientation_euler_angles[1] / Units.deg
            ##box_contour_field_vtk(interpolatedBoxData, stateData, iteration=iteration, filename="/Users/rerha/Desktop/test_relaxed_wake/convergenceLoop/ContourBox.vtk")
            # ----------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------   
                
            iteration += 1
            diff = max_diff
            print(diff)               
                           
                
            # Update the vortex strengths of each vortex ring accordingly
        
        return self, rotor, interpolatedBoxData
    
    def get_fluid_domain_boundaries(self, prop, factor=1.5):
        """
        Generates the boundary points for this propeller.
        factorX - Number of rotor diameters over which to stretch the computational domain in X
        factorY - Number of rotor diameters over which to stretch the computational domain in Y
        factorZ - Number of rotor diameters over which to stretch the computational domain in Z
        
        """
        if prop.Wake.vortex_distribution !=None:
            WD = prop.Wake.vortex_distribution
            Xmin = np.min([WD.XA1, WD.XA2, WD.XB1, WD.XB2])
            Xmax = np.max([WD.XA1, WD.XA2, WD.XB1, WD.XB2])
            Ymin = np.min([WD.YA1, WD.YA2, WD.YB1, WD.YB2])
            Ymax = np.max([WD.YA1, WD.YA2, WD.YB1, WD.YB2])
            Zmin = np.min([WD.ZA1, WD.ZA2, WD.ZB1, WD.ZB2])
            Zmax = np.max([WD.ZA1, WD.ZA2, WD.ZB1, WD.ZB2])
            
        else:
            print("Wake not initialized! Generating temporary computational domain grid.\n")
            # extract properties
            R = prop.tip_radius
            O = prop.origin
            Alpha_rot = prop.orientation_euler_angles[1]
            
            # compute domain boundaries
            Xmin = O[0] - 2*R 
            Xmax = O[0] + 2*R * 4
            Ymin = O[1] - 2*R
            Ymax = O[1] + 2*R
            Zmin = O[2] - 2*R
            Zmax = O[2] + 2*R
        
            # Apply y-axis rotation
        
        return Xmin, Xmax, Ymin, Ymax, Zmin, Zmax    
  