## @ingroup Analyses-Propulsion
# Rotor_Wake_Fidelity_One.py
#
# Created:  Jan 2022, R. Erhard
# Modified: Aug 2022, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data, Units
from SUAVE.Components import Wings
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Analyses.Propulsion.Rotor_Wake_Fidelity_Zero import Rotor_Wake_Fidelity_Zero
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.fidelity_one_wake_convergence import fidelity_one_wake_convergence
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_wake_induced_velocity import compute_wake_induced_velocity

from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry 
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.extract_wing_VD import extract_wing_collocation_points

from DCode.Common.Visualization_Tools.box_contour_field_vtk import box_contour_field_vtk
from DCode.Common.generalFunctions import save_single_prop_vehicle_vtk

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
        self.wake_method_fidelity       = 1
        self.semi_prescribed_converge   = True      # flag for convergence on semi-prescribed wake shape
        self.vtk_save_flag              = False      # flag for saving vtk outputs of wake
        self.vtk_save_loc               = None       # location to save vtk outputs of wake
        
        self.wake_settings              = Data()
        self.wake_settings.number_rotor_rotations     = 4
        self.wake_settings.number_steps_per_rotation  = 72
        self.wake_settings.initial_timestep_offset    = 0    # initial timestep
        self.influencing_rotor_wake_network = None
        
        # wake convergence criteria
        self.maximum_convergence_iteration_gamma      = 1#50
        self.maximum_convergence_iteration_va         = 1#50
        self.axial_velocity_convergence_tolerance     = 1e-3
        self.circulation_convergence_tolerance        = 1e-3
        
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
        
        # store airfoil data to the rotor
        if rotor.airfoil_data == None:
            a_sec              = rotor.airfoil_geometry   
            rotor.airfoil_data = import_airfoil_geometry(a_sec,npoints=100)         
        
        # match the azimuthal discretization betwen rotor and wake
        if self.wake_settings.number_steps_per_rotation  != rotor.number_azimuthal_stations:
            self.wake_settings.number_steps_per_rotation = rotor.number_azimuthal_stations
            
            if self.verbose:
                print("Wake azimuthal discretization does not match rotor discretization. \
                Resetting wake to match rotor of Na="+str(rotor.number_azimuthal_stations))
        
        return
    
    def evaluate(self,rotor,wake_inputs,conditions,VD=None):
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
            
        return va, vt, rotor
    
    
    def evaluate_slipstream(self,rotor,geometry,ctrl_pts,wing_instance=None):
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
        # Check for wing if wing instance is unspecified
        if wing_instance == None:
            nmw = 0
            # check for main wing
            for i,wing in enumerate(geometry.wings):
                if not isinstance(wing,Wings.Main_Wing): continue
                nmw +=1                
                wing_instance = wing
                wing_instance_idx = i
            if nmw == 1:
                pass
            elif nmw>1:
                print("No wing specified for slipstream analysis. Multiple main wings in vehicle, using the last one.")
            else:
                print("No wing specified for slipstream analysis. No main wing defined, using the last wing in vehicle.")
                wing_instance = wing 
                wing_instance_idx = i
        
        # Isolate the VD components corresponding to this wing instance
        wing_CPs, slipstream_vd_ids = extract_wing_collocation_points(geometry, wing_instance_idx)
        
        # Evaluate rotor slipstream effect on specified wing instance
        rot_V_wake_ind = self.evaluate_wake_velocities(rotor, wing_CPs, ctrl_pts)
        
        # Expand
        wake_V_ind = np.zeros((ctrl_pts,geometry.vortex_distribution.n_cp,3))
        wake_V_ind[:,slipstream_vd_ids,:] = rot_V_wake_ind
        
            
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
        #VD.Wake         = wake_vortex_distribution

        Na = rotor.number_azimuthal_stations
        
        start_angle = rotor.start_angle
        angles = np.linspace(0,2*np.pi,Na+1)[:-1]
        azi_start_idx = np.where(np.isclose(abs(start_angle),angles))[0][0]
        
        rot_V_wake_ind  = compute_wake_induced_velocity(wake_vortex_distribution,VD,num_ctrl_pts,azi_start_idx)        
        
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
        self.vortex_distribution = wVD
        return
    
    def rotate_propFrame_to_globalFrame(self, rotor):
        """
        This rotates all points in the vortex wake by the rotation angle. This is primarily
        used for transforming the wake from the prop to the vehicle frame to maintain consistent
        reference frames for vehicle analysis.
        """
        wVD = self.vortex_distribution
        
        # rotate to prop frame
        rot_mat = rotor.prop_vel_to_vehicle_body()[0]
        if 'XC' in wVD.keys():
            C = np.matmul(rot_mat, np.array([wVD.XC,wVD.YC,wVD.ZC]))
            
            wVD.XC = np.reshape(C[0,:], np.shape(wVD.XC))
            wVD.YC = np.reshape(C[1,:], np.shape(wVD.XC))
            wVD.ZC = np.reshape(C[2,:], np.shape(wVD.XC))            
        
        A1 = np.matmul(rot_mat, np.array([np.ravel(wVD.XA1),np.ravel(wVD.YA1),np.ravel(wVD.ZA1)]))
        A2 = np.matmul(rot_mat, np.array([np.ravel(wVD.XA2),np.ravel(wVD.YA2),np.ravel(wVD.ZA2)]))
        B1 = np.matmul(rot_mat, np.array([np.ravel(wVD.XB1),np.ravel(wVD.YB1),np.ravel(wVD.ZB1)]))
        B2 = np.matmul(rot_mat, np.array([np.ravel(wVD.XB2),np.ravel(wVD.YB2),np.ravel(wVD.ZB2)]))

        rsA1 = np.matmul(rot_mat, np.array([np.ravel(wVD.reshaped_wake.XA1),np.ravel(wVD.reshaped_wake.YA1),np.ravel(wVD.reshaped_wake.ZA1)]))
        rsA2 = np.matmul(rot_mat, np.array([np.ravel(wVD.reshaped_wake.XA2),np.ravel(wVD.reshaped_wake.YA2),np.ravel(wVD.reshaped_wake.ZA2)]))
        rsB1 = np.matmul(rot_mat, np.array([np.ravel(wVD.reshaped_wake.XB1),np.ravel(wVD.reshaped_wake.YB1),np.ravel(wVD.reshaped_wake.ZB1)]))
        rsB2 = np.matmul(rot_mat, np.array([np.ravel(wVD.reshaped_wake.XB2),np.ravel(wVD.reshaped_wake.YB2),np.ravel(wVD.reshaped_wake.ZB2)]))        


        wVD.XA1 = np.reshape(A1[0,:], np.shape(wVD.XA1))
        wVD.YA1 = np.reshape(A1[1,:], np.shape(wVD.XA1))
        wVD.ZA1 = np.reshape(A1[2,:], np.shape(wVD.XA1))

        wVD.XA2 = np.reshape(A2[0,:], np.shape(wVD.XA1))
        wVD.YA2 = np.reshape(A2[1,:], np.shape(wVD.XA1))
        wVD.ZA2 = np.reshape(A2[2,:], np.shape(wVD.XA1))    

        wVD.XB1 = np.reshape(B1[0,:], np.shape(wVD.XA1))
        wVD.YB1 = np.reshape(B1[1,:], np.shape(wVD.XA1))
        wVD.ZB1 = np.reshape(B1[2,:], np.shape(wVD.XA1))      

        wVD.XB2 = np.reshape(B2[0,:], np.shape(wVD.XA1))
        wVD.YB2 = np.reshape(B2[1,:], np.shape(wVD.XA1))
        wVD.ZB2 = np.reshape(B2[2,:], np.shape(wVD.XA1))      
        
        
        wVD.reshaped_wake.XA1 = np.reshape(rsA1[0,:], np.shape(wVD.reshaped_wake.XA1))
        wVD.reshaped_wake.YA1 = np.reshape(rsA1[1,:], np.shape(wVD.reshaped_wake.XA1))
        wVD.reshaped_wake.ZA1 = np.reshape(rsA1[2,:], np.shape(wVD.reshaped_wake.XA1))

        wVD.reshaped_wake.XA2 = np.reshape(rsA2[0,:], np.shape(wVD.reshaped_wake.XA1))
        wVD.reshaped_wake.YA2 = np.reshape(rsA2[1,:], np.shape(wVD.reshaped_wake.XA1))
        wVD.reshaped_wake.ZA2 = np.reshape(rsA2[2,:], np.shape(wVD.reshaped_wake.XA1))    

        wVD.reshaped_wake.XB1 = np.reshape(rsB1[0,:], np.shape(wVD.reshaped_wake.XA1))
        wVD.reshaped_wake.YB1 = np.reshape(rsB1[1,:], np.shape(wVD.reshaped_wake.XA1))
        wVD.reshaped_wake.ZB1 = np.reshape(rsB1[2,:], np.shape(wVD.reshaped_wake.XA1))      

        wVD.reshaped_wake.XB2 = np.reshape(rsB2[0,:], np.shape(wVD.reshaped_wake.XA1))
        wVD.reshaped_wake.YB2 = np.reshape(rsB2[1,:], np.shape(wVD.reshaped_wake.XA1))
        wVD.reshaped_wake.ZB2 = np.reshape(rsB2[2,:], np.shape(wVD.reshaped_wake.XA1))              
        
        self.vortex_distribution = wVD        
        return
    
        
        
        





