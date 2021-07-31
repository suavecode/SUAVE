## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
#  generate_propeller_wake_distribution.py
# 
# Created:  Sep 2020, M. Clarke 
# Modified: May 2021, R. Erhard
#           Jul 2021, E. Botero
#           Jul 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np
from SUAVE.Core import Data 
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wake_contraction_matrix import compute_wake_contraction_matrix

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift   
def generate_propeller_wake_distribution(props,identical,m,VD,init_timestep_offset, time, number_of_wake_timesteps,conditions ): 
    """ This generates the propeller wake control points used to compute the 
    influence of the wake

    Assumptions: 
    None

    Source:   
    None

    Inputs:  
    identical            - if all props are identical        [Bool]
    m                    - control point                     [Unitless] 
    VD                   - vortex distribution               
    prop                 - propeller/rotor data structure         
    init_timestep_offset - intial time step                  [Unitless] 
    time                 - time                              [s]

    Properties Used:
    N/A
    """    
    num_prop = len(props) 
    
    if identical:
        # All props are identical in geometry, so only the first one is unpacked
        prop_keys    = list(props.keys())
        prop_key     = prop_keys[0]
        prop         = props[prop_key]
        prop_outputs = conditions.noise.sources.propellers[prop_key]
        
        Bmax = int(prop.number_of_blades)
        nmax = int(prop_outputs.number_radial_stations - 1)
        
    else:
        # Props are unique, must find required matrix sizes to fit all vortex distributions
        prop_keys   = list(props.keys())
        B_list      = np.ones(len(prop_keys))
        Nr_list     = np.ones(len(prop_keys))
    
        for i in range(len(prop_keys)):
            p_key      = list(props.keys())[i]
            p          = props[p_key]
            p_out      = conditions.noise.sources.propellers[p_key]
            
            B_list[i]  = p.number_of_blades
            Nr_list[i] = p_out.number_radial_stations
            
        # Identify max indices for pre-allocating vortex wake distribution matrices
        Bmax = int(max(B_list))
        nmax = int(max(Nr_list)-1)
        
    # Initialize empty arrays with required sizes
    VD, WD, Wmid = initialize_distributions(nmax, Bmax, number_of_wake_timesteps, num_prop, m,VD)
    
    # for each propeller, unpack and compute 
    for i, propi in enumerate(props):
        propi_key        = list(props.keys())[i]
        if identical:
            propi_outputs = prop_outputs
        else:
            propi_outputs     = conditions.noise.sources.propellers[propi_key]
        
        # Unpack
        R                = propi.tip_radius
        r                = propi.radius_distribution 
        c                = propi.chord_distribution 
        MCA              = propi.mid_chord_alignment 
        B                = propi.number_of_blades    
        
        Na               = propi_outputs.number_azimuthal_stations
        Nr               = propi_outputs.number_radial_stations
        omega            = propi_outputs.omega                               
        va               = propi_outputs.disc_axial_induced_velocity 
        V_inf            = propi_outputs.velocity
        gamma            = propi_outputs.disc_circulation
        
        blade_angles     = np.linspace(0,2*np.pi,B+1)[:-1]   
        dt               = time/number_of_wake_timesteps
        ts               = np.linspace(0,time,number_of_wake_timesteps) 
        
        t0                = dt*init_timestep_offset
        start_angle       = omega[0]*t0 
        propi.start_angle = start_angle

        # define points ( control point, time step , blade number , location on blade )
        # compute lambda and mu 
        mean_induced_velocity  = np.mean( np.mean(va,axis = 1),axis = 1)   
    
        lambda_tot   =  np.atleast_2d((V_inf[:,0]  + mean_induced_velocity)).T /(omega*R)   # inflow advance ratio (page 30 Leishman)
        mu_prop      =  np.atleast_2d(V_inf[:,2]).T /(omega*R)                              # rotor advance ratio  (page 30 Leishman) 
        V_prop       =  np.atleast_2d(np.sqrt((V_inf[:,0]  + mean_induced_velocity)**2 + (V_inf[:,2])**2)).T
    
        # wake skew angle 
        wake_skew_angle = np.arctan(mu_prop/lambda_tot)
    
        # reshape gamma to find the average between stations 
        gamma_new = np.zeros((m,Na,(Nr-1))) #[control points, azimuth  , radial stations -1 ] one less because ring
        gamma_new = (gamma[:,:,:-1] + gamma[:,:,1:])*0.5 #  not sure if correct 


        num       = int(Na/B)  
        time_idx  = np.arange(number_of_wake_timesteps-1) 
        t_idx     = np.atleast_2d(time_idx).T 
        B_idx     = np.arange(B) 
        B_loc     = (B_idx*num + t_idx)%Na  
        Gamma     = gamma_new[:,B_loc,:]  
    
        #( control point, time step , blade number , location on blade )
        sx_inf0            = np.multiply(V_prop*np.cos(wake_skew_angle), np.atleast_2d(ts))
        sx_inf             = np.repeat(np.repeat(sx_inf0[:, :,  np.newaxis], Nr, axis = 2)[:,  : ,np.newaxis,  :], B, axis = 2)  
                          
        sy_inf0            = np.multiply(np.atleast_2d(V_inf[:,1]).T,np.atleast_2d(ts)) # = zero since no crosswind
        sy_inf             = np.repeat(np.repeat(sy_inf0[:, :,  np.newaxis], Nr, axis = 2)[:,  : ,np.newaxis,  :], B, axis = 2)   
                          
        sz_inf0            = np.multiply(V_prop*np.sin(wake_skew_angle),np.atleast_2d(ts))
        sz_inf             = np.repeat(np.repeat(sz_inf0[:, :,  np.newaxis], Nr, axis = 2)[:,  : ,np.newaxis,  :], B, axis = 2)           
    
        angle_offset       = np.repeat(np.repeat(np.multiply(omega,np.atleast_2d(ts))[:, :,  np.newaxis],B, axis = 2)[:, :,:, np.newaxis],Nr, axis = 3) 
        blade_angle_loc    = np.repeat(np.repeat(np.tile(np.atleast_2d(blade_angles),(m,1))[:,  np.newaxis, :],number_of_wake_timesteps, axis = 1) [:, :,:, np.newaxis],Nr, axis = 3) 
        start_angle_offset = np.repeat(np.repeat(np.atleast_2d(start_angle)[:, :, np.newaxis],B, axis = 2)[:, :,:, np.newaxis],Nr, axis = 3) 
        
        total_angle_offset = angle_offset - start_angle_offset
        
        if (propi.rotation != None) and (propi.rotation == 1):        
            total_angle_offset = -total_angle_offset

        azi_y   = np.sin(blade_angle_loc + total_angle_offset)  
        azi_z   = np.cos(blade_angle_loc + total_angle_offset)

        x0_pts = np.tile(np.atleast_2d(MCA+c/4),(B,1))  
        x_pts  = np.repeat(np.repeat(x0_pts[np.newaxis,:,  :], number_of_wake_timesteps, axis=0)[ np.newaxis, : ,:, :,], m, axis=0) 
        X_pts0 = x_pts + sx_inf   

        # compute wake contraction  
        wake_contraction = compute_wake_contraction_matrix(i,propi,Nr,m,number_of_wake_timesteps,X_pts0,propi_outputs)          

        y0_pts = np.tile(np.atleast_2d(r),(B,1))
        y_pts  = np.repeat(np.repeat(y0_pts[np.newaxis,:,  :], number_of_wake_timesteps, axis=0)[ np.newaxis, : ,:, :,], m, axis=0) 
        Y_pts0 = (y_pts*wake_contraction)*azi_y  + sy_inf    

        z0_pts = np.tile(np.atleast_2d(r),(B,1))
        z_pts  = np.repeat(np.repeat(z0_pts[np.newaxis,:,  :], number_of_wake_timesteps, axis=0)[ np.newaxis, : ,:, :,], m, axis=0)
        Z_pts0 = (z_pts*wake_contraction)*azi_z + sz_inf     
 
        # Rotate wake by thrust angle
        rot_mat = propi.prop_vel_to_body()

        # append propeller wake to each of its repeated origins        
        #for rep in range(len(propi_origins)):
        X_pts   = propi.origin[0][0] + X_pts0*rot_mat[0,0] - Z_pts0*rot_mat[0,2]
        Y_pts   = propi.origin[0][1] + Y_pts0*rot_mat[1,1]
        Z_pts   = propi.origin[0][2] + X_pts0*rot_mat[2,0] + Z_pts0*rot_mat[2,2]

        # Store points  
        # ( control point,  prop ,  time step , blade number , location on blade )
        if (propi.rotation != None) and (propi.rotation == -1):  
            Wmid.WD_XA1[:,i,:,0:B,:] = X_pts[: , :-1 , : , :-1 ]
            Wmid.WD_YA1[:,i,:,0:B,:] = Y_pts[: , :-1 , : , :-1 ]
            Wmid.WD_ZA1[:,i,:,0:B,:] = Z_pts[: , :-1 , : , :-1 ]
            Wmid.WD_XA2[:,i,:,0:B,:] = X_pts[: ,  1: , : , :-1 ]
            Wmid.WD_YA2[:,i,:,0:B,:] = Y_pts[: ,  1: , : , :-1 ]
            Wmid.WD_ZA2[:,i,:,0:B,:] = Z_pts[: ,  1: , : , :-1 ]
            Wmid.WD_XB1[:,i,:,0:B,:] = X_pts[: , :-1 , : , 1:  ]
            Wmid.WD_YB1[:,i,:,0:B,:] = Y_pts[: , :-1 , : , 1:  ]
            Wmid.WD_ZB1[:,i,:,0:B,:] = Z_pts[: , :-1 , : , 1:  ]
            Wmid.WD_XB2[:,i,:,0:B,:] = X_pts[: ,  1: , : , 1:  ]
            Wmid.WD_YB2[:,i,:,0:B,:] = Y_pts[: ,  1: , : , 1:  ]
            Wmid.WD_ZB2[:,i,:,0:B,:] = Z_pts[: ,  1: , : , 1:  ] 
        else: 
            Wmid.WD_XA1[:,i,:,0:B,:] = X_pts[: , :-1 , : , 1:  ]
            Wmid.WD_YA1[:,i,:,0:B,:] = Y_pts[: , :-1 , : , 1:  ]
            Wmid.WD_ZA1[:,i,:,0:B,:] = Z_pts[: , :-1 , : , 1:  ]
            Wmid.WD_XA2[:,i,:,0:B,:] = X_pts[: ,  1: , : , 1:  ]
            Wmid.WD_YA2[:,i,:,0:B,:] = Y_pts[: ,  1: , : , 1:  ]
            Wmid.WD_ZA2[:,i,:,0:B,:] = Z_pts[: ,  1: , : , 1:  ] 
            Wmid.WD_XB1[:,i,:,0:B,:] = X_pts[: , :-1 , : , :-1 ]
            Wmid.WD_YB1[:,i,:,0:B,:] = Y_pts[: , :-1 , : , :-1 ]
            Wmid.WD_ZB1[:,i,:,0:B,:] = Z_pts[: , :-1 , : , :-1 ]
            Wmid.WD_XB2[:,i,:,0:B,:] = X_pts[: ,  1: , : , :-1 ]
            Wmid.WD_YB2[:,i,:,0:B,:] = Y_pts[: ,  1: , : , :-1 ]
            Wmid.WD_ZB2[:,i,:,0:B,:] = Z_pts[: ,  1: , : , :-1 ]

        Wmid.WD_GAMMA[:,i,:,0:B,:] = Gamma 

        # store points for plotting 
        VD.Wake.XA1[i,:,0:B,:] =  X_pts[0 , :-1 , : , :-1 ]
        VD.Wake.YA1[i,:,0:B,:] =  Y_pts[0 , :-1 , : , :-1 ]
        VD.Wake.ZA1[i,:,0:B,:] =  Z_pts[0 , :-1 , : , :-1 ]
        VD.Wake.XA2[i,:,0:B,:] =  X_pts[0 ,  1: , : , :-1 ]
        VD.Wake.YA2[i,:,0:B,:] =  Y_pts[0 ,  1: , : , :-1 ]
        VD.Wake.ZA2[i,:,0:B,:] =  Z_pts[0 ,  1: , : , :-1 ]
        VD.Wake.XB1[i,:,0:B,:] =  X_pts[0 , :-1 , : , 1:  ]
        VD.Wake.YB1[i,:,0:B,:] =  Y_pts[0 , :-1 , : , 1:  ]
        VD.Wake.ZB1[i,:,0:B,:] =  Z_pts[0 , :-1 , : , 1:  ]
        VD.Wake.XB2[i,:,0:B,:] =  X_pts[0 ,  1: , : , 1:  ]
        VD.Wake.YB2[i,:,0:B,:] =  Y_pts[0 ,  1: , : , 1:  ]
        VD.Wake.ZB2[i,:,0:B,:] =  Z_pts[0 ,  1: , : , 1:  ]  
        
    # Compress Data into 1D Arrays  
    mat4_size = (m,num_prop,(number_of_wake_timesteps-1),Bmax*nmax)
    mat5_size = (m,num_prop,(number_of_wake_timesteps-1)*Bmax*nmax)
    mat6_size = (m,num_prop*(number_of_wake_timesteps-1)*Bmax*nmax) 

    WD.XA1    =  np.reshape(np.reshape(np.reshape(Wmid.WD_XA1,mat4_size),mat5_size),mat6_size)
    WD.YA1    =  np.reshape(np.reshape(np.reshape(Wmid.WD_YA1,mat4_size),mat5_size),mat6_size)
    WD.ZA1    =  np.reshape(np.reshape(np.reshape(Wmid.WD_ZA1,mat4_size),mat5_size),mat6_size)
    WD.XA2    =  np.reshape(np.reshape(np.reshape(Wmid.WD_XA2,mat4_size),mat5_size),mat6_size)
    WD.YA2    =  np.reshape(np.reshape(np.reshape(Wmid.WD_YA2,mat4_size),mat5_size),mat6_size)
    WD.ZA2    =  np.reshape(np.reshape(np.reshape(Wmid.WD_ZA2,mat4_size),mat5_size),mat6_size)
    WD.XB1    =  np.reshape(np.reshape(np.reshape(Wmid.WD_XB1,mat4_size),mat5_size),mat6_size)
    WD.YB1    =  np.reshape(np.reshape(np.reshape(Wmid.WD_YB1,mat4_size),mat5_size),mat6_size)
    WD.ZB1    =  np.reshape(np.reshape(np.reshape(Wmid.WD_ZB1,mat4_size),mat5_size),mat6_size)
    WD.XB2    =  np.reshape(np.reshape(np.reshape(Wmid.WD_XB2,mat4_size),mat5_size),mat6_size)
    WD.YB2    =  np.reshape(np.reshape(np.reshape(Wmid.WD_YB2,mat4_size),mat5_size),mat6_size)
    WD.ZB2    =  np.reshape(np.reshape(np.reshape(Wmid.WD_ZB2,mat4_size),mat5_size),mat6_size)
    WD.GAMMA  =  np.reshape(np.reshape(np.reshape(Wmid.WD_GAMMA,mat4_size),mat5_size),mat6_size)

    return WD, dt, ts, B, Nr 


def initialize_distributions(nmax, Bmax, n_wts, n_props, m,VD):
    
    Wmid        = Data()
    mat1_size = (m,n_props,n_wts-1,Bmax,nmax)
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
    mat2_size = (m,n_props*(n_wts-1)*Bmax*nmax)
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
    mat3_size     = (n_props,(n_wts-1),Bmax,nmax)
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