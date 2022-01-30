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


# package imports
import numpy as np
import copy

# ----------------------------------------------------------------------
#  Generalized Rotor Class
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
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
        self.Wake_VD                    = Data()
        self.wake_method_fidelity       = 0
        self.number_rotor_rotations     = 5
        self.number_steps_per_rotation  = 72
        self.wake_settings              = Data()
        self.system_vortex_distribution = None

        self.wake_settings.initial_timestep_offset   = 0    # initial timestep
        self.wake_settings.wake_development_time     = 0.05 # total simulation time required for wake development
        self.wake_settings.number_of_wake_timesteps  = 72*5   # total number of time steps in wake development
        
    def initialize(self,rotor,conditions):
        """
        Initializes the rotor by evaluating the BET once. This is required for generating the 
        circulation strengths for the vortex distribution in the PVW.
        
        """
        # run the BET once using fidelity zero inflow
        rotor_temp = copy.deepcopy(rotor)
        rotor_temp.Wake = Rotor_Wake_Fidelity_Zero()
        _,_,_,_,outputs,_ = rotor_temp.spin(conditions)
        
        rotor.outputs = outputs
        
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
        ii,ii_max = 0, 20            
        va_diff, tol = 1, 1e-2   

        while va_diff > tol:  
            # generate wake geometry for rotor
            WD, dt, ts, B, Nr  = self.generate_wake_shape(rotor)
            
            # compute axial wake-induced velocity (a byproduct of the circulation distribution which is an input to the wake geometry)
            va, vt = compute_PVW_inflow_velocities(self,rotor, WD)

            # compute new blade velocities
            Wa   = va + Ua
            Wt   = Ut - vt

            lamdaw, F, _ = compute_inflow_and_tip_loss(r,R,Wa,Wt,B)

            va_diff = np.max(abs(F*va - rotor.outputs.disc_axial_induced_velocity))
            print(va_diff)

            # update the axial disc velocity based on new va from HFW
            rotor.outputs.disc_axial_induced_velocity = F*va 
            
            ii+=1
            if ii>ii_max and va_diff>tol:
                print("Semi-prescribed vortex wake did not converge on axial inflow used for wake shape.")
                break
    

        return va, vt
    
    def generate_wake_shape(self,rotor):
        """
        This generates the propeller wake control points and vortex distribution that make up the PVW.
        
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
        B                = rotor.number_of_blades  
        
        rotor_outputs    = rotor.outputs
        Na               = rotor_outputs.number_azimuthal_stations
        Nr               = rotor_outputs.number_radial_stations
        omega            = rotor_outputs.omega                               
        va               = rotor_outputs.disc_axial_induced_velocity 
        V_inf            = rotor_outputs.velocity
        gamma            = rotor_outputs.disc_circulation   
        
        try:
            VD = rotor.vortex_distribution
        except:
            VD = Data()
            rotor.vortex_distribution = VD
      
        # dimensions for analysis                      
        Nr   = len(r)                   # number of radial stations
        m    = len(omega)                         # number of control points
        nts  = self.wake_settings.number_of_wake_timesteps      # number of wake time steps
        
        time                 = self.wake_settings.wake_development_time
        init_timestep_offset = self.wake_settings.initial_timestep_offset
        
        # Initialize empty arrays with required sizes
        VD, WD, Wmid = initialize_distributions(Nr, B, nts, m,VD)
        
        # Compute wake geometry properties
        blade_angles     = np.linspace(0,2*np.pi,B+1)[:-1]   
        dt               = time/nts
        ts               = np.linspace(0,time,nts) 
        
        t0                = dt*init_timestep_offset
        start_angle       = omega[0]*t0 
        rotor.start_angle = start_angle[0]

        # compute lambda and mu 
        mean_radial_induced_velocity  = np.mean(va,axis = 2)
        mean_induced_velocity  = np.mean( mean_radial_induced_velocity,axis = 1)   
    
        alpha = rotor.orientation_euler_angles[1]
        rots  = np.array([[np.cos(alpha), 0, np.sin(alpha)], [0,1,0], [-np.sin(alpha), 0, np.cos(alpha)]])
        
        lambda_tot   =  np.atleast_2d((np.dot(V_inf,rots[0])  + mean_induced_velocity)).T /(omega*R)   # inflow advance ratio (page 99 Leishman)
        mu_prop      =  np.atleast_2d(np.dot(V_inf,rots[2])).T /(omega*R)                              # rotor advance ratio  (page 99 Leishman) 
        V_prop       =  np.atleast_2d(np.sqrt((V_inf[:,0]  + mean_radial_induced_velocity)**2 + (V_inf[:,2])**2))

        # wake skew angle 
        wake_skew_angle = -(np.arctan(mu_prop/lambda_tot))
        
        # reshape gamma to find the average between stations 
        gamma_new = np.zeros((m,(Nr-1),Na))                  # [control points, Nr-1, Na ] one less radial station because ring
        gamma_new = (gamma[:,:-1,:] + gamma[:,1:,:])*0.5
        
        num       = Na//B
        time_idx  = np.arange(nts)
        time_idx  = np.roll(time_idx,rotor.rotation*int(init_timestep_offset)) # (nts//Na)*
        t_idx     = np.atleast_2d(time_idx).T 
        B_idx     = np.arange(B) 
        B_loc     = (B_idx*num + t_idx )%Na 
        Gamma     = gamma_new[:,:,B_loc]  
        Gamma     = Gamma.transpose(0,3,1,2)   
        
        # --------------------------------------------------------------------------------------------------------------
        #    ( control point , blade number , radial location on blade , time step )
        # --------------------------------------------------------------------------------------------------------------
        V_p = np.repeat(V_prop[:,:,None],len(ts),axis=2)
                        
        sx_inf0            = np.multiply(V_p*np.cos(wake_skew_angle), np.repeat(np.atleast_2d(ts)[:,None,:],Nr,axis=1))
        sx_inf             = np.repeat(sx_inf0[:, None, :,:], B, axis = 1)
                          
        sy_inf0            = np.multiply(np.atleast_2d(V_inf[:,1]).T,np.atleast_2d(ts)) # = zero since no crosswind
        sy_inf             = np.repeat(np.repeat(sy_inf0[:, None, :], B, axis = 1)[:, :, None, :], Nr, axis = 2)   
                          
        sz_inf0            = np.multiply(V_p*np.sin(wake_skew_angle),np.repeat(np.atleast_2d(ts)[:,None,:],Nr,axis=1))
        sz_inf             = np.repeat(sz_inf0[:, None, :,:], B, axis = 1)        
    
        angle_offset       = np.repeat(np.repeat(np.multiply(omega,np.atleast_2d(ts))[:, None, :],B, axis = 1)[:, :, None, :],Nr, axis = 2) 
        blade_angle_loc    = np.repeat(np.repeat(np.tile(np.atleast_2d(blade_angles),(m,1))[:, :, None ], Nr, axis = 2)[:, :, :, None],nts, axis = 3) 
        start_angle_offset = np.repeat(np.repeat(np.atleast_2d(start_angle)[:, None, :],B, axis = 1)[:, :, None, :],Nr, axis = 2) 
        
        total_angle_offset = angle_offset - start_angle_offset
        
        if (rotor.rotation != None) and (rotor.rotation == 1):        
            total_angle_offset = -total_angle_offset
            Gamma = np.flip(Gamma,axis=1)

        azi_y   = np.sin(blade_angle_loc + total_angle_offset)  
        azi_z   = np.cos(blade_angle_loc + total_angle_offset)
        

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
        beta = rotor.twist_distribution
        xte_twisted = np.cos(beta)*xte_airfoils - np.sin(beta)*yte_airfoils        
        yte_twisted = np.sin(beta)*xte_airfoils + np.cos(beta)*yte_airfoils    
        
        x_c_4_twisted = np.cos(beta)*x_c_4_airfoils - np.sin(beta)*y_c_4_airfoils 
        y_c_4_twisted = np.sin(beta)*x_c_4_airfoils + np.cos(beta)*y_c_4_airfoils  
        
        # transform coordinates from airfoil frame to rotor frame
        xte = np.tile(np.atleast_2d(yte_twisted), (B,1))
        xte_rotor = np.tile(xte[None,:,:,None], (m,1,1,nts))
        yte_rotor = -np.tile(xte_twisted[None,None,:,None],(m,B,1,1))*np.cos(blade_angle_loc+total_angle_offset) 
        zte_rotor = np.tile(xte_twisted[None,None,:,None],(m,B,1,1))*np.sin(blade_angle_loc+total_angle_offset)
        
        r_4d = np.tile(r[None,None,:,None], (m,B,1,nts))
        
        x0 = 0
        y0 = r_4d*azi_y
        z0 = r_4d*azi_z
        
        x_pts0 = x0 + xte_rotor
        y_pts0 = y0 + yte_rotor
        z_pts0 = z0 + zte_rotor
        
        x_c_4_rotor = x0 - np.tile(y_c_4_twisted[None,None,:,None], (m,B,1,nts))
        y_c_4_rotor = y0 + np.tile(x_c_4_twisted[None,None,:,None], (m,B,1,nts))*np.cos(blade_angle_loc+total_angle_offset)
        z_c_4_rotor = z0 - np.tile(x_c_4_twisted[None,None,:,None], (m,B,1,nts))*np.sin(blade_angle_loc+total_angle_offset)   
        
        # compute wake contraction, apply to y-z plane
        X_pts0           = x_pts0 + sx_inf
        wake_contraction = compute_wake_contraction_matrix(rotor,Nr,m,nts,X_pts0,rotor_outputs) 
        Y_pts0           = y_pts0*wake_contraction + sy_inf
        Z_pts0           = z_pts0*wake_contraction + sz_inf
 
        # Rotate wake by thrust angle
        rot_to_body = rotor.prop_vel_to_body()  # rotate points into the body frame: [Z,Y,X]' = R*[Z,Y,X]
        
        # append propeller wake to each of its repeated origins  
        X_pts   = rotor.origin[0][0] + X_pts0*rot_to_body[2,2] + Z_pts0*rot_to_body[2,0]   
        Y_pts   = rotor.origin[0][1] + Y_pts0*rot_to_body[1,1]                       
        Z_pts   = rotor.origin[0][2] + Z_pts0*rot_to_body[0,0] + X_pts0*rot_to_body[0,2] 
        
        #------------------------------------------------------     
        # Account for lifting line panels
        #------------------------------------------------------
        rots  = np.array([[np.cos(alpha), 0, np.sin(alpha)], [0,1,0], [-np.sin(alpha), 0, np.cos(alpha)]])
                    
        # rotate rotor points to incidence angle
        x_c_4 = (x_c_4_rotor)*rots[0,0] + (y_c_4_rotor)*rots[0,1] + (z_c_4_rotor)*rots[0,2] + rotor.origin[0][0]
        y_c_4 = (x_c_4_rotor)*rots[1,0] + (y_c_4_rotor)*rots[1,1] + (z_c_4_rotor)*rots[1,2] + rotor.origin[0][1]
        z_c_4 = (x_c_4_rotor)*rots[2,0] + (y_c_4_rotor)*rots[2,1] + (z_c_4_rotor)*rots[2,2] + rotor.origin[0][2]
        
        # prepend points at quarter chord to account for rotor lifting line
        X_pts = np.append(x_c_4[:,:,:,0][:,:,:,None], X_pts, axis=3) 
        Y_pts = np.append(y_c_4[:,:,:,0][:,:,:,None], Y_pts, axis=3) 
        Z_pts = np.append(z_c_4[:,:,:,0][:,:,:,None], Z_pts, axis=3)
            

        #------------------------------------------------------
        # Store points  
        #------------------------------------------------------
        # ( control point,  prop  , blade number , location on blade, time step )
        if (rotor.rotation != None) and (rotor.rotation == -1):  
            Wmid.WD_XA1[:,0:B,:,:] = X_pts[: , :, :-1 , :-1 ]
            Wmid.WD_YA1[:,0:B,:,:] = Y_pts[: , :, :-1 , :-1 ]
            Wmid.WD_ZA1[:,0:B,:,:] = Z_pts[: , :, :-1 , :-1 ]
            Wmid.WD_XA2[:,0:B,:,:] = X_pts[: , :, :-1 ,  1: ]
            Wmid.WD_YA2[:,0:B,:,:] = Y_pts[: , :, :-1 ,  1: ]
            Wmid.WD_ZA2[:,0:B,:,:] = Z_pts[: , :, :-1 ,  1: ]
            Wmid.WD_XB1[:,0:B,:,:] = X_pts[: , :, 1:  , :-1 ]
            Wmid.WD_YB1[:,0:B,:,:] = Y_pts[: , :, 1:  , :-1 ]
            Wmid.WD_ZB1[:,0:B,:,:] = Z_pts[: , :, 1:  , :-1 ]
            Wmid.WD_XB2[:,0:B,:,:] = X_pts[: , :, 1:  ,  1: ]
            Wmid.WD_YB2[:,0:B,:,:] = Y_pts[: , :, 1:  ,  1: ]
            Wmid.WD_ZB2[:,0:B,:,:] = Z_pts[: , :, 1:  ,  1: ] 
        else: 
            Wmid.WD_XA1[:,0:B,:,:] = X_pts[: , :, 1:  , :-1 ]
            Wmid.WD_YA1[:,0:B,:,:] = Y_pts[: , :, 1:  , :-1 ]
            Wmid.WD_ZA1[:,0:B,:,:] = Z_pts[: , :, 1:  , :-1 ]
            Wmid.WD_XA2[:,0:B,:,:] = X_pts[: , :, 1:  ,  1: ]
            Wmid.WD_YA2[:,0:B,:,:] = Y_pts[: , :, 1:  ,  1: ]
            Wmid.WD_ZA2[:,0:B,:,:] = Z_pts[: , :, 1:  ,  1: ] 
            Wmid.WD_XB1[:,0:B,:,:] = X_pts[: , :, :-1 , :-1 ]
            Wmid.WD_YB1[:,0:B,:,:] = Y_pts[: , :, :-1 , :-1 ]
            Wmid.WD_ZB1[:,0:B,:,:] = Z_pts[: , :, :-1 , :-1 ]
            Wmid.WD_XB2[:,0:B,:,:] = X_pts[: , :, :-1 ,  1: ]
            Wmid.WD_YB2[:,0:B,:,:] = Y_pts[: , :, :-1 ,  1: ]
            Wmid.WD_ZB2[:,0:B,:,:] = Z_pts[: , :, :-1 ,  1: ]

        Wmid.WD_GAMMA[:,0:B,:,:] = Gamma 

        # store points for plotting 
        VD.Wake.XA1[:,0:B,:,:] =  X_pts[: , :, :-1 , :-1 ]
        VD.Wake.YA1[:,0:B,:,:] =  Y_pts[: , :, :-1 , :-1 ]
        VD.Wake.ZA1[:,0:B,:,:] =  Z_pts[: , :, :-1 , :-1 ]
        VD.Wake.XA2[:,0:B,:,:] =  X_pts[: , :, :-1 ,  1: ]
        VD.Wake.YA2[:,0:B,:,:] =  Y_pts[: , :, :-1 ,  1: ]
        VD.Wake.ZA2[:,0:B,:,:] =  Z_pts[: , :, :-1 ,  1: ]
        VD.Wake.XB1[:,0:B,:,:] =  X_pts[: , :, 1:  , :-1 ]
        VD.Wake.YB1[:,0:B,:,:] =  Y_pts[: , :, 1:  , :-1 ]
        VD.Wake.ZB1[:,0:B,:,:] =  Z_pts[: , :, 1:  , :-1 ]
        VD.Wake.XB2[:,0:B,:,:] =  X_pts[: , :, 1:  ,  1: ]
        VD.Wake.YB2[:,0:B,:,:] =  Y_pts[: , :, 1:  ,  1: ]
        VD.Wake.ZB2[:,0:B,:,:] =  Z_pts[: , :, 1:  ,  1: ]  
        
        
        
        # Append wake geometry and vortex strengths to each individual propeller
        self.Wake_VD.XA1   = VD.Wake.XA1[:,0:B,:,:]
        self.Wake_VD.YA1   = VD.Wake.YA1[:,0:B,:,:]
        self.Wake_VD.ZA1   = VD.Wake.ZA1[:,0:B,:,:]
        self.Wake_VD.XA2   = VD.Wake.XA2[:,0:B,:,:]
        self.Wake_VD.YA2   = VD.Wake.YA2[:,0:B,:,:]
        self.Wake_VD.ZA2   = VD.Wake.ZA2[:,0:B,:,:]
        self.Wake_VD.XB1   = VD.Wake.XB1[:,0:B,:,:]
        self.Wake_VD.YB1   = VD.Wake.YB1[:,0:B,:,:]
        self.Wake_VD.ZB1   = VD.Wake.ZB1[:,0:B,:,:]
        self.Wake_VD.XB2   = VD.Wake.XB2[:,0:B,:,:]
        self.Wake_VD.YB2   = VD.Wake.YB2[:,0:B,:,:]
        self.Wake_VD.ZB2   = VD.Wake.ZB2[:,0:B,:,:]
        self.Wake_VD.GAMMA = Wmid.WD_GAMMA[:,0:B,:,:]
        
        # append trailing edge locations
        self.Wake_VD.Xblades_te = X_pts[0,:,:,0]
        self.Wake_VD.Yblades_te = Y_pts[0,:,:,0]
        self.Wake_VD.Zblades_te = Z_pts[0,:,:,0]

        # append quarter chord lifting line point locations        
        self.Wake_VD.Xblades_c_4 = x_c_4_rotor 
        self.Wake_VD.Yblades_c_4 = y_c_4_rotor
        self.Wake_VD.Zblades_c_4 = z_c_4_rotor
        
        # append three-quarter chord evaluation point locations        
        self.Wake_VD.Xblades_cp = x_c_4 
        self.Wake_VD.Yblades_cp = y_c_4 
        self.Wake_VD.Zblades_cp = z_c_4 
        
        self.Wake_VD.Xblades_cp2 =  X_pts[0,:,:,0] + (X_pts[0,:,:,0]-X_pts[0,:,:,1])/2
        self.Wake_VD.Yblades_cp2 =  Y_pts[0,:,:,0] + (Y_pts[0,:,:,0]-Y_pts[0,:,:,1])/2
        self.Wake_VD.Zblades_cp2 =  Z_pts[0,:,:,0] + (Z_pts[0,:,:,0]-Z_pts[0,:,:,1])/2
    
        # Compress Data into 1D Arrays  
        mat6_size = (m,nts*B*(Nr-1)) 
    
        WD.XA1    =  np.reshape(Wmid.WD_XA1,mat6_size)
        WD.YA1    =  np.reshape(Wmid.WD_YA1,mat6_size)
        WD.ZA1    =  np.reshape(Wmid.WD_ZA1,mat6_size)
        WD.XA2    =  np.reshape(Wmid.WD_XA2,mat6_size)
        WD.YA2    =  np.reshape(Wmid.WD_YA2,mat6_size)
        WD.ZA2    =  np.reshape(Wmid.WD_ZA2,mat6_size)
        WD.XB1    =  np.reshape(Wmid.WD_XB1,mat6_size)
        WD.YB1    =  np.reshape(Wmid.WD_YB1,mat6_size)
        WD.ZB1    =  np.reshape(Wmid.WD_ZB1,mat6_size)
        WD.XB2    =  np.reshape(Wmid.WD_XB2,mat6_size)
        WD.YB2    =  np.reshape(Wmid.WD_YB2,mat6_size)
        WD.ZB2    =  np.reshape(Wmid.WD_ZB2,mat6_size)
        WD.GAMMA  =  np.reshape(Wmid.WD_GAMMA,mat6_size)
        WD.wake_skew_angle = wake_skew_angle
        
        return WD, dt, ts, B, Nr 
    
    
def initialize_distributions(Nr, B, n_wts, m,VD):
    nmax = Nr - 1 # one less vortex ring than blade elements
    
    Wmid        = Data()
    mat1_size = (m,B,nmax, n_wts)
    Wmid.WD_XA1    = np.zeros(mat1_size)  
    Wmid.WD_YA1    = np.zeros(mat1_size)  
    Wmid.WD_ZA1    = np.zeros(mat1_size)  
    Wmid.WD_XA2    = np.zeros(mat1_size)  
    Wmid.WD_YA2    = np.zeros(mat1_size)  
    Wmid.WD_ZA2    = np.zeros(mat1_size)      
    Wmid.WD_XB1    = np.zeros(mat1_size)  
    Wmid.WD_YB1    = np.zeros(mat1_size)  
    Wmid.WD_ZB1    = np.zeros(mat1_size)  
    Wmid.WD_XB2    = np.zeros(mat1_size)   
    Wmid.WD_YB2    = np.zeros(mat1_size)   
    Wmid.WD_ZB2    = np.zeros(mat1_size)     
    Wmid.WD_GAMMA  = np.zeros(mat1_size)     

    WD        = Data()
    mat2_size = (m*n_wts*B*nmax)
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

    VD.Wake       = Data()
    mat3_size     = (m,B,nmax,n_wts)
    VD.Wake.XA1   = np.zeros(mat3_size) 
    VD.Wake.YA1   = np.zeros(mat3_size) 
    VD.Wake.ZA1   = np.zeros(mat3_size) 
    VD.Wake.XA2   = np.zeros(mat3_size) 
    VD.Wake.YA2   = np.zeros(mat3_size) 
    VD.Wake.ZA2   = np.zeros(mat3_size)    
    VD.Wake.XB1   = np.zeros(mat3_size) 
    VD.Wake.YB1   = np.zeros(mat3_size) 
    VD.Wake.ZB1   = np.zeros(mat3_size) 
    VD.Wake.XB2   = np.zeros(mat3_size) 
    VD.Wake.YB2   = np.zeros(mat3_size) 
    VD.Wake.ZB2   = np.zeros(mat3_size) 
    
    return VD, WD, Wmid


