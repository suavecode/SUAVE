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
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry 
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_Zero.compute_wake_contraction_matrix import compute_wake_contraction_matrix
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_PVW_inflow_velocities import compute_PVW_inflow_velocities
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.BET_calculations import compute_inflow_and_tip_loss
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.BET_calculations import compute_airfoil_aerodynamics

from SUAVE.Input_Output.VTK.save_vehicle_vtk import save_vehicle_vtks
from SUAVE.Input_Output.VTK.save_evaluation_points_vtk import save_evaluation_points_vtk

# package imports
import numpy as np
import copy

import SUAVE

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
        self.wake_method                = 'PVW'
        self.Wake_VD                    = Data()
        self.wake_method_fidelity       = 0
        self.semi_prescribed_converge   = False      # flag for convergence on semi-prescribed wake shape
        self.vtk_save_flag              = False      # flag for saving vtk outputs of wake
        self.vtk_save_loc               = None       # location to save vtk outputs of wake
        
        self.wake_settings              = Data()
        self.wake_settings.number_rotor_rotations     = 5
        self.wake_settings.number_steps_per_rotation  = 72
        self.wake_settings.initial_timestep_offset    = 0    # initial timestep

        # flags for slipstream interaction
        self.slipstream                 = False
        
    def initialize(self,rotor,conditions):
        """
        Initializes the rotor by evaluating the BET once. This is required for generating the 
        circulation strengths for the vortex distribution in the PVW, and the initial wake shape,
        which relies on the axial inflow induced by the wake at the rotor disc.
        
        """
        # run the BET once using fidelity zero inflow
        rotor_temp = copy.deepcopy(rotor)
        rotor_temp.Wake = Rotor_Wake_Fidelity_Zero()
        _,_,_,_,outputs,_ = rotor_temp.spin(conditions)
        
        rotor.outputs = outputs
        
        # match the azimuthal discretization betwen rotor and wake
        if self.wake_settings.number_steps_per_rotation  != rotor.number_azimuthal_stations:
            self.wake_settings.number_steps_per_rotation = rotor.number_azimuthal_stations
            print("Wake azimuthal discretization does not match rotor discretization. \
            Resetting wake to match rotor of Na="+str(rotor.number_azimuthal_stations))
        
        return
    
    def evaluate(self,rotor,U,Ua,Ut,PSI,omega,beta,c,r,R,B,a,nu,a_loc,a_geo,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc,use_2d_analysis,conditions):
        """
        Wake evaluation is performed using a semi-prescribed vortex wake (PVW) method for Fidelity One.
        
           
        Outputs of this function include the inflow velocities induced by rotor wake:
           va  - axially-induced velocity from rotor wake
           vt  - tangentially-induced velocity from rotor wake
        
        """   
        
        # Initialize rotor with single pass of VW 
        self.initialize(rotor,conditions)
        
        # converge on va for a semi-prescribed wake method
        va_diff, tol, ii = 1, 1e-2, 0
        if self.semi_prescribed_converge:
            print("\tConverging on semi-prescribed wake shape...")
            ii_max = 10
        else:
            print("\tGenerating fully-prescribed wake shape...")
            ii_max = 1
        

        while va_diff > tol:  
            # generate wake geometry for rotor
            WD  = self.generate_wake_shape(rotor)
            
            # compute axial wake-induced velocity (a byproduct of the circulation distribution which is an input to the wake geometry)
            va, vt = compute_PVW_inflow_velocities(self,rotor, WD)

            # compute new blade velocities
            Wa   = va + Ua
            Wt   = Ut - vt

            lamdaw, F, _ = compute_inflow_and_tip_loss(r,R,Wa,Wt,B)

            va_diff = np.max(abs(F*va - rotor.outputs.disc_axial_induced_velocity))

            # update the axial disc velocity based on new va from HFW
            rotor.outputs.disc_axial_induced_velocity = F*va 
            
            ii+=1
            if ii>=ii_max and va_diff>tol:
                if self.semi_prescribed_converge:
                    print("Semi-prescribed vortex wake did not converge on axial inflow used for wake shape.")
                break

            
        # save converged wake:
        WD  = self.generate_wake_shape(rotor)
        self.vortex_distribution = WD
            
        return va, vt
    
    def generate_wake_shape(self,rotor):
        """
        This generates the propeller wake control points and vortex distribution that make up the PVW. 
        All (x,y,z) coordinates are in the vehicle frame of reference (X points nose to tail).
        
        Assumptions:
           None
        
        Source: 
           None
        
        Inputs:
           rotor  -  A SUAVE rotor component for which the rotor wake is generated
        
        """
            
        # Unpack rotor
        R                = rotor.tip_radius
        r                = rotor.radius_distribution 
        c                = rotor.chord_distribution 
        beta             = rotor.twist_distribution
        B                = rotor.number_of_blades  
        
        rotor_outputs    = rotor.outputs
        Na               = rotor_outputs.number_azimuthal_stations
        Nr               = rotor_outputs.number_radial_stations
        omega            = rotor_outputs.omega                               
        va               = rotor_outputs.disc_axial_induced_velocity 
        V_inf            = rotor_outputs.velocity
        gamma            = rotor_outputs.disc_circulation   
        rot              = rotor.rotation
        
        # apply rotation direction to twist and chord distribution
        c    = -rot*c
        beta = -rot*beta
        
        # dimensions for analysis                      
        Nr   = len(r)                   # number of radial stations
        m    = len(omega)                         # number of control points

        # Compute blade angles starting from each of Na azimuthal stations, shape: (Na,B)
        azi          = np.linspace(0,2*np.pi,Na+1)[:-1]
        azi_initial  = np.atleast_2d(np.linspace(0,2*np.pi,B+1)[:-1])
        blade_angles = (azi_initial + np.atleast_2d(azi).T) 
        
        # Extract specified wake settings:
        init_timestep_offset = self.wake_settings.initial_timestep_offset
        n_rotations          = self.wake_settings.number_rotor_rotations
        tsteps_per_rot       = self.wake_settings.number_steps_per_rotation
        
        # Calculate additional wake properties
        dt    = (azi[1]-azi[0])/omega[0][0]
        nts   = tsteps_per_rot*n_rotations
        
        # Compute properties for each wake timestep
        ts                = np.linspace(0,dt*(nts-1),nts) 
        omega_ts          = np.multiply(omega,np.atleast_2d(ts))  # Angle of each azimuthal station in nts
        
        # Update start angle of rotor
        t0                = dt*init_timestep_offset
        start_angle       = omega[0]*t0 
        rotor.start_angle = start_angle[0]
        
        # extract mean inflow velocities
        axial_induced_velocity = np.mean(va,axis = 2) # radial inflow, averaged around the azimuth
        mean_induced_velocity  = np.mean( axial_induced_velocity,axis = 1)   
    
        alpha = rotor.orientation_euler_angles[1]
        rots  = np.array([[np.cos(alpha), 0, np.sin(alpha)], [0,1,0], [-np.sin(alpha), 0, np.cos(alpha)]])
        
        lambda_tot   =  np.atleast_2d((np.dot(V_inf,rots[0])  + mean_induced_velocity)).T /(omega*R)   # inflow advance ratio (page 99 Leishman)
        mu_prop      =  np.atleast_2d(np.dot(V_inf,rots[2])).T /(omega*R)                              # rotor advance ratio  (page 99 Leishman) 
        Vx           = np.repeat(V_inf[:,0,None], Nr, axis=1) # shape: (m,Nr)
        Vz           = np.repeat(V_inf[:,2,None], Nr, axis=1) # shape: (m,Nr)
        V_prop       =  np.sqrt((Vx  + axial_induced_velocity)**2 + Vz**2)

        # wake skew angle 
        wake_skew_angle = -(np.arctan(mu_prop/lambda_tot))
        wake_skew_angle = np.tile(wake_skew_angle[:,:,None],(1,Nr,nts))
        
        # reshape gamma to find the average between stations           
        gamma_new = (gamma[:,:-1,:] + gamma[:,1:,:])*0.5  # [control points, Nr-1, Na ] one less radial station because ring
        
        num       = Na//B
        time_idx  = np.arange(nts)
        Gamma     = np.zeros((Na,m,B,Nr-1,nts))
        
        # generate Gamma for each start angle
        for ito in range(Na):
            t_idx     = np.atleast_2d(time_idx).T 
            B_idx     = np.arange(B) 
            B_loc     = (ito + B_idx*num - t_idx )%Na 
            Gamma1    = gamma_new[:,:,B_loc]  
            Gamma1    = Gamma1.transpose(0,3,1,2) 
            Gamma[ito,:,:,:,:] = Gamma1
      
        # --------------------------------------------------------------------------------------------------------------
        #    ( control point , blade number , radial location on blade , time step )
        # --------------------------------------------------------------------------------------------------------------
        V_p = np.repeat(V_prop[:,:,None],len(ts),axis=2)
                        
        sx_inf0            = np.multiply(V_p*np.cos(wake_skew_angle), np.repeat(np.atleast_2d(ts)[:,None,:],Nr,axis=1))
        sx_inf             = np.tile(sx_inf0[None,:, None, :,:], (Na,1,B,1,1))
                          
        sy_inf0            = np.multiply(np.atleast_2d(V_inf[:,1]).T,np.atleast_2d(ts)) # = zero since no crosswind
        sy_inf             = -rot*np.tile(sy_inf0[None,:, None, None,:], (Na,1,B,Nr,1)) 
        
        sz_inf0            = np.multiply(V_p*np.sin(wake_skew_angle),np.repeat(np.atleast_2d(ts)[:,None,:],Nr,axis=1))
        sz_inf             = np.tile(sz_inf0[None,:, None, :,:], (Na,1,B,1,1))        
        
        # wake panel and blade angles
        start_angle_offset = np.tile(start_angle[None,:,None,None,None], (Na,1,B,Nr,nts))
        blade_angle_loc    = start_angle_offset + np.tile( blade_angles[:,None,:,None,None], (1,m,1,Nr,nts))  # negative rotation, positive blade angle location
        
        # offset angle of trailing wake panels relative to blade location
        total_angle_offset = np.tile(omega_ts[None,:,None,None,:], (Na,1,B,Nr,1))   
        
        # azimuthal position of each wake panel, (blade start index, ctrl_pts, B, Nr, nts)
        panel_azimuthal_positions = rot*(total_angle_offset - blade_angle_loc)      # axial view in rotor frame (angle 0 aligned with z-axis); 
        
        # put into velocity frame and find (y,z) components
        azi_y   = np.sin(panel_azimuthal_positions)
        azi_z   = np.cos(panel_azimuthal_positions)
        

        # extract airfoil trailing edge coordinates for initial location of vortex wake
        a_sec        = rotor.airfoil_geometry   
        a_secl       = rotor.airfoil_polar_stations
        airfoil_data = import_airfoil_geometry(a_sec,npoints=100)  
       
        # trailing edge points in airfoil coordinates
        xupper         = np.take(airfoil_data.x_upper_surface,a_secl,axis=0)
        yupper         = np.take(airfoil_data.y_upper_surface,a_secl,axis=0)   
        
        # Align the quarter chords of the airfoils (zero sweep)
        airfoil_le_offset = -c/2
        xte_airfoils      = xupper[:,-1]*c + airfoil_le_offset
        yte_airfoils      = yupper[:,-1]*c 
        
        xle_airfoils = xupper[:,0]*c + airfoil_le_offset
        yle_airfoils = yupper[:,0]*c 
        
        
        x_c_4_airfoils = (xle_airfoils - xte_airfoils)/4 - airfoil_le_offset
        y_c_4_airfoils = (yle_airfoils - yte_airfoils)/4
        
        # apply blade twist rotation along rotor radius
        xte_twisted = np.cos(beta)*xte_airfoils - np.sin(beta)*yte_airfoils        
        yte_twisted = np.sin(beta)*xte_airfoils + np.cos(beta)*yte_airfoils    
        
        x_c_4_twisted = np.cos(beta)*x_c_4_airfoils - np.sin(beta)*y_c_4_airfoils 
        y_c_4_twisted = np.sin(beta)*x_c_4_airfoils + np.cos(beta)*y_c_4_airfoils  
        
        # transform coordinates from airfoil frame to rotor frame
        xte = np.tile(np.atleast_2d(yte_twisted), (B,1))
        xte_rotor = np.tile(xte[None,:,:,None], (m,1,1,nts))  
        yte_rotor = -np.tile(xte_twisted[None,None,:,None],(m,B,1,1))*np.cos(panel_azimuthal_positions)
        zte_rotor = np.tile(xte_twisted[None,None,:,None],(m,B,1,1))*np.sin(panel_azimuthal_positions)
        
        r_4d = np.tile(r[None,None,:,None], (m,B,1,nts))
        
        x0 = 0
        y0 = r_4d*azi_y
        z0 = r_4d*azi_z
        
        x_pts0 = x0 + xte_rotor
        y_pts0 = y0 + yte_rotor
        z_pts0 = z0 + zte_rotor
        
        x_c_4_rotor = x0 - np.tile(y_c_4_twisted[None,None,:,None], (m,B,1,nts))
        y_c_4_rotor = y0 + np.tile(x_c_4_twisted[None,None,:,None], (m,B,1,nts))*np.cos(panel_azimuthal_positions)
        z_c_4_rotor = z0 - np.tile(x_c_4_twisted[None,None,:,None], (m,B,1,nts))*np.sin(panel_azimuthal_positions)   
        
        # compute wake contraction, apply to y-z plane
        X_pts0           = x_pts0 + sx_inf
        wake_contraction = compute_wake_contraction_matrix(rotor,Nr,m,nts,X_pts0,rotor_outputs) 
        Y_pts0           = y_pts0*wake_contraction + sy_inf
        Z_pts0           = z_pts0*wake_contraction + sz_inf
        
        # append propeller wake to each of its repeated origins  
        X_pts   = rotor.origin[0][0] + X_pts0  
        Y_pts   = rotor.origin[0][1] + Y_pts0
        Z_pts   = rotor.origin[0][2] + Z_pts0

        #------------------------------------------------------     
        # Account for lifting line panels
        #------------------------------------------------------
        x_c_4 = np.repeat(x_c_4_rotor[None,:,:,:,:], Na, axis=0) + rotor.origin[0][0]
        y_c_4 = (y_c_4_rotor) + rotor.origin[0][1]
        z_c_4 = (z_c_4_rotor) + rotor.origin[0][2]
        
        # prepend points at quarter chord to account for rotor lifting line
        X_pts = np.append(x_c_4[:,:,:,:,0][:,:,:,:,None], X_pts, axis=4) 
        Y_pts = np.append(y_c_4[:,:,:,:,0][:,:,:,:,None], Y_pts, axis=4)
        Z_pts = np.append(z_c_4[:,:,:,:,0][:,:,:,:,None], Z_pts, axis=4)

        #------------------------------------------------------
        # Store points  
        #------------------------------------------------------
        # Initialize vortex distribution and arrays with required matrix sizes
        VD = Data()
        rotor.vortex_distribution = VD        
        VD, WD = initialize_distributions(Nr, Na, B, nts, m,VD)
        
        # ( azimuthal start index, control point  , blade number , location on blade, time step )
        if rot==-1:
            # panels ordered root to tip, A for inner-most panel edge
            VD.Wake.XA1[:,:,0:B,:,:] = X_pts[:, : , :, :-1 , :-1 ]
            VD.Wake.YA1[:,:,0:B,:,:] = Y_pts[:, : , :, :-1 , :-1 ]
            VD.Wake.ZA1[:,:,0:B,:,:] = Z_pts[:, : , :, :-1 , :-1 ]
            VD.Wake.XA2[:,:,0:B,:,:] = X_pts[:, : , :, :-1 ,  1: ]
            VD.Wake.YA2[:,:,0:B,:,:] = Y_pts[:, : , :, :-1 ,  1: ]
            VD.Wake.ZA2[:,:,0:B,:,:] = Z_pts[:, : , :, :-1 ,  1: ]
            VD.Wake.XB1[:,:,0:B,:,:] = X_pts[:, : , :, 1:  , :-1 ]
            VD.Wake.YB1[:,:,0:B,:,:] = Y_pts[:, : , :, 1:  , :-1 ]
            VD.Wake.ZB1[:,:,0:B,:,:] = Z_pts[:, : , :, 1:  , :-1 ]
            VD.Wake.XB2[:,:,0:B,:,:] = X_pts[:, : , :, 1:  ,  1: ]
            VD.Wake.YB2[:,:,0:B,:,:] = Y_pts[:, : , :, 1:  ,  1: ]
            VD.Wake.ZB2[:,:,0:B,:,:] = Z_pts[:, : , :, 1:  ,  1: ] 
        else:            
            # positive rotation reverses the A,B nomenclature of the panel
            VD.Wake.XA1[:,:,0:B,:,:] = X_pts[:, : , :, 1: , :-1 ]
            VD.Wake.YA1[:,:,0:B,:,:] = Y_pts[:, : , :, 1: , :-1 ]
            VD.Wake.ZA1[:,:,0:B,:,:] = Z_pts[:, : , :, 1: , :-1 ]
            VD.Wake.XA2[:,:,0:B,:,:] = X_pts[:, : , :, 1: ,  1: ]
            VD.Wake.YA2[:,:,0:B,:,:] = Y_pts[:, : , :, 1: ,  1: ]
            VD.Wake.ZA2[:,:,0:B,:,:] = Z_pts[:, : , :, 1: ,  1: ]
            VD.Wake.XB1[:,:,0:B,:,:] = X_pts[:, : , :, :-1  , :-1 ]
            VD.Wake.YB1[:,:,0:B,:,:] = Y_pts[:, : , :, :-1  , :-1 ]
            VD.Wake.ZB1[:,:,0:B,:,:] = Z_pts[:, : , :, :-1  , :-1 ]
            VD.Wake.XB2[:,:,0:B,:,:] = X_pts[:, : , :, :-1  ,  1: ]
            VD.Wake.YB2[:,:,0:B,:,:] = Y_pts[:, : , :, :-1  ,  1: ]
            VD.Wake.ZB2[:,:,0:B,:,:] = Z_pts[:, : , :, :-1  ,  1: ] 
            

        VD.Wake.GAMMA[:,:,0:B,:,:] = Gamma 
        
        # Append wake geometry and vortex strengths to each individual propeller
        self.Wake_VD   = VD.Wake
        
        # append trailing edge locations
        self.Wake_VD.Xblades_te = X_pts[:,0,:,:,0]
        self.Wake_VD.Yblades_te = Y_pts[:,0,:,:,0]
        self.Wake_VD.Zblades_te = Z_pts[:,0,:,:,0]

        # append quarter chord lifting line point locations        
        self.Wake_VD.Xblades_c_4 = x_c_4_rotor 
        self.Wake_VD.Yblades_c_4 = y_c_4_rotor
        self.Wake_VD.Zblades_c_4 = z_c_4_rotor
        
        # append three-quarter chord evaluation point locations        
        self.Wake_VD.Xblades_cp = x_c_4 
        self.Wake_VD.Yblades_cp = y_c_4 
        self.Wake_VD.Zblades_cp = z_c_4 
    
        # Compress Data into 1D Arrays  
        mat6_size = (Na,m,nts*B*(Nr-1)) 
    
        WD.XA1    =  np.reshape(VD.Wake.XA1,mat6_size)
        WD.YA1    =  np.reshape(VD.Wake.YA1,mat6_size)
        WD.ZA1    =  np.reshape(VD.Wake.ZA1,mat6_size)
        WD.XA2    =  np.reshape(VD.Wake.XA2,mat6_size)
        WD.YA2    =  np.reshape(VD.Wake.YA2,mat6_size)
        WD.ZA2    =  np.reshape(VD.Wake.ZA2,mat6_size)
        WD.XB1    =  np.reshape(VD.Wake.XB1,mat6_size)
        WD.YB1    =  np.reshape(VD.Wake.YB1,mat6_size)
        WD.ZB1    =  np.reshape(VD.Wake.ZB1,mat6_size)
        WD.XB2    =  np.reshape(VD.Wake.XB2,mat6_size)
        WD.YB2    =  np.reshape(VD.Wake.YB2,mat6_size)
        WD.ZB2    =  np.reshape(VD.Wake.ZB2,mat6_size)
        WD.GAMMA  =  np.reshape(VD.Wake.GAMMA,mat6_size)
        
        rotor.wake_skew_angle = wake_skew_angle
        WD.reshaped_wake = self.Wake_VD
                
           
        return WD
    
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
        self.Wake_VD = wVD
        self.vortex_distribution = wVD
        return
        
        
    def store_vtks(self,rotor):
        generate_vtks = self.vtk_save_flag
        save_loc = self.vtk_save_loc

        # Unpack rotor
        rotor_outputs = rotor.outputs
        Na            = rotor_outputs.number_azimuthal_stations
        omega         = rotor_outputs.omega                 
        VD            = rotor.vortex_distribution
        
        # Get start angle of rotor
        azi   = np.linspace(0,2*np.pi,Na+1)[:-1]
        ito   = self.wake_settings.initial_timestep_offset
        dt    = (azi[1]-azi[0])/omega[0][0]
        t0    = dt*ito
        
        # --------------------------------------------------------------------------------------------------------------
        #    Store VTKs after wake is generated
        # --------------------------------------------------------------------------------------------------------------      
        if generate_vtks:
            if save_loc == None:
                pass
            else:
                # after converged, store vtks for final wake shape for each of Na starting positions
                for i in range(Na):
                    # increment blade angle to new azimuthal position
                    blade_angle       = (omega[0]*t0 + i*(2*np.pi/(Na))) * rotor.rotation  # Positive rotation, positive blade angle
                    rotor.start_angle = blade_angle
        
                    print("\nStoring VTKs...")
                    
                    # create dummy vehicle
                    vehicle = SUAVE.Vehicle()
                    net     = SUAVE.Components.Energy.Networks.Battery_Propeller()
                    net.number_of_engines  = 1
                    net.propellers.append(rotor)
                    vehicle.append_component(net) 
        
                    save_vehicle_vtks(vehicle, Results=Data(), time_step=i,save_loc=save_loc)  
        
                    Yb   = self.Wake_VD.Yblades_cp[i,0,0,:,0] 
                    Zb   = self.Wake_VD.Zblades_cp[i,0,0,:,0] 
                    Xb   = self.Wake_VD.Xblades_cp[i,0,0,:,0] 
        
                    VD.YC = (Yb[1:] + Yb[:-1])/2
                    VD.ZC = (Zb[1:] + Zb[:-1])/2
                    VD.XC = (Xb[1:] + Xb[:-1])/2
        
                    points = Data()
                    points.XC = VD.XC
                    points.YC = VD.YC
                    points.ZC = VD.ZC
                    points.induced_velocities = Data()
                    points.induced_velocities.va = rotor_outputs.disc_axial_induced_velocity[0,:,i]
                    points.induced_velocities.vt = rotor_outputs.disc_tangential_induced_velocity[0,:,i]
                    save_evaluation_points_vtk(points,filename=save_loc+"/eval_pts.vtk", time_step=i)
        return
    
def initialize_distributions(Nr, Na, B, n_wts, m, VD):
    """
    Initialize the matrices
    
    Inputs:
       Nr    - number of radial blade elemnts
       Na    - number of azimuthal start positions
       B     - number of rotor blades
       n_wts - total number of wake time steps in wake simulation
       m     - number of control points to evaluate
       VD    - vehicle vortex distribution
    """
    nmax = Nr - 1 # one less vortex ring than blade elements
    
    VD.Wake       = Data()
    mat1_size     = (Na,m,B,nmax,n_wts)
    VD.Wake.XA1   = np.zeros(mat1_size) 
    VD.Wake.YA1   = np.zeros(mat1_size) 
    VD.Wake.ZA1   = np.zeros(mat1_size) 
    VD.Wake.XA2   = np.zeros(mat1_size) 
    VD.Wake.YA2   = np.zeros(mat1_size) 
    VD.Wake.ZA2   = np.zeros(mat1_size)    
    VD.Wake.XB1   = np.zeros(mat1_size) 
    VD.Wake.YB1   = np.zeros(mat1_size) 
    VD.Wake.ZB1   = np.zeros(mat1_size) 
    VD.Wake.XB2   = np.zeros(mat1_size) 
    VD.Wake.YB2   = np.zeros(mat1_size) 
    VD.Wake.ZB2   = np.zeros(mat1_size) 
    VD.Wake.GAMMA  = np.zeros(mat1_size)  
      
    WD        = Data()
    mat2_size = (Na,m*n_wts*B*nmax)
    WD.XA1    = np.zeros(mat2_size)
    WD.YA1    = np.zeros(mat2_size)
    WD.ZA1    = np.zeros(mat2_size)
    WD.XA2    = np.zeros(mat2_size)
    WD.YA2    = np.zeros(mat2_size)
    WD.ZA2    = np.zeros(mat2_size)   
    WD.XB1    = np.zeros(mat2_size)
    WD.YB1    = np.zeros(mat2_size)
    WD.ZB1    = np.zeros(mat2_size)
    WD.XB2    = np.zeros(mat2_size)
    WD.YB2    = np.zeros(mat2_size)
    WD.ZB2    = np.zeros(mat2_size) 

 
    return VD, WD

