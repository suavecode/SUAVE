## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
#  generate_propeller_wake_distribution.py
# 
# Created:  Sep 2020, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np
from SUAVE.Core import Data 
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wake_contraction_matrix import compute_wake_contraction_matrix
from SUAVE.Methods.Geometry.Three_Dimensional import  orientation_product, orientation_transpose

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift   
def generate_propeller_wake_distribution(prop,thrust_angle,m,VD,init_timestep_offset, time): 
    """ This generates the propeller wake control points used to compute the 
    influence of the wake

    Assumptions: 
    None

    Source:   
    None

    Inputs:  
    m                    - control point                     [Unitless] 
    VD                   - vortex distribution               
    prop                 - propeller/rotor data structure         
    init_timestep_offset - intial time step                  [Unitless] 
    time                 - time                              [s]

    Properties Used:
    N/A
    """    

    # Unpack unknowns  
    R            = prop.tip_radius
    r            = prop.radius_distribution 
    c            = prop.chord_distribution 
    Na           = prop.outputs.number_azimuthal_stations
    Nr           = prop.outputs.number_radial_stations
    omega        = prop.outputs.omega                               
    va           = prop.outputs.disc_axial_induced_velocity 
    V_inf        = prop.outputs.velocity
    MCA          = prop.mid_chord_aligment 
    B            = prop.number_blades
    gamma        = prop.outputs.disc_circulation
    blade_angles = np.linspace(0,2*np.pi,B+1)[:-1]        
    dt           = (2*np.pi/Na)/omega[0]
    nts          = int(time/dt)
    ts           = np.linspace(0,time,nts)
    num_prop     = len(prop.origin) 

    t0           = dt*init_timestep_offset
    start_angle  = omega[0]*t0 

    # define points ( control point, time step , blade number , location on blade )
    # compute lambda and mu 
    mean_induced_velocity  = np.mean( np.mean(va,axis = 1),axis = 1)   

    lambda_tot   =  np.atleast_2d((V_inf[:,0]  + mean_induced_velocity)).T /(omega*R)   # inflow advance ratio (page 30 Leishman)
    mu_prop      =  np.atleast_2d(V_inf[:,2]).T /(omega*R)                              # rotor advance ratio  (page 30 Leishman) 
    V_prop       =  np.atleast_2d(np.sqrt((V_inf[:,0]  + mean_induced_velocity)**2 + (V_inf[:,2])**2)).T

    # wake skew angle 
    wake_skew_angle = np.arctan(mu_prop/lambda_tot)

    # reshape gamma to find the average between stations 
    gamma_new = np.zeros((m,Na,(Nr-1)))
    gamma_new = (gamma[:,:,:-1] + gamma[:,:,1:])*0.5

    # define empty arrays 
    WD        = Data()
    n         = Nr-1
    mat1_size = (m,num_prop,nts-1,B,n)
    WD_XA1    = np.zeros(mat1_size)  
    WD_YA1    = np.zeros(mat1_size)  
    WD_ZA1    = np.zeros(mat1_size)  
    WD_XA2    = np.zeros(mat1_size)  
    WD_YA2    = np.zeros(mat1_size)  
    WD_ZA2    = np.zeros(mat1_size)      
    WD_XB1    = np.zeros(mat1_size)  
    WD_YB1    = np.zeros(mat1_size)  
    WD_ZB1    = np.zeros(mat1_size)  
    WD_XB2    = np.zeros(mat1_size)   
    WD_YB2    = np.zeros(mat1_size)   
    WD_ZB2    = np.zeros(mat1_size)     
    WD_GAMMA  = np.zeros(mat1_size)     

    mat2_size = (m,num_prop*(nts-1)*B*n)
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
    mat3_size     = (num_prop,(nts-1),B,n)
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
    
    Gamma     = np.zeros((m,nts-1,B,n))   
    num       = int(Na/B)  
    time_idx  = np.arange(nts-1) 
    t_idx     = np.atleast_2d(time_idx).T 
    B_idx     = np.arange(B) 
    B_loc     = (B_idx*num + t_idx)%Na  
    Gamma     = gamma_new[:,B_loc,:]  

    #( control point, time step , blade number , location on blade )
    sx_inf0      = np.multiply(V_prop*np.cos(wake_skew_angle), np.atleast_2d(ts))
    sx_inf       = np.repeat(np.repeat(sx_inf0[:, :,  np.newaxis], Nr, axis = 2)[:,  : ,np.newaxis,  :], B, axis = 2)  

    sy_inf0      = np.multiply(np.atleast_2d(V_inf[:,1]).T,np.atleast_2d(ts)) # = zero since no crosswind
    sy_inf       = np.repeat(np.repeat(sy_inf0[:, :,  np.newaxis], Nr, axis = 2)[:,  : ,np.newaxis,  :], B, axis = 2)   

    sz_inf0      = np.multiply(V_prop*np.sin(wake_skew_angle),np.atleast_2d(ts))
    sz_inf       = np.repeat(np.repeat(sz_inf0[:, :,  np.newaxis], Nr, axis = 2)[:,  : ,np.newaxis,  :], B, axis = 2)           

    angle_offset       = np.repeat(np.repeat(np.multiply(omega,np.atleast_2d(ts))[:, :,  np.newaxis],B, axis = 2)[:, :,:, np.newaxis],Nr, axis = 3) 
    blade_angle_loc    = np.repeat(np.repeat(np.tile(np.atleast_2d(blade_angles),(m,1))[:,  np.newaxis, :],nts, axis = 1) [:, :,:, np.newaxis],Nr, axis = 3) 
    start_angle_offset = np.repeat(np.repeat(np.atleast_2d(start_angle)[:, :, np.newaxis],B, axis = 2)[:, :,:, np.newaxis],Nr, axis = 3) 
    
    # adjust for clockwise/counter clockwise rotation
    total_angle_offset = angle_offset - start_angle_offset

    for i in range(num_prop): 

        if (prop.rotation != None) and (prop.rotation[i] == 1):        
            total_angle_offset = -total_angle_offset    

        azi_y   = np.sin(blade_angle_loc + total_angle_offset)  
        azi_z   = np.cos(blade_angle_loc + total_angle_offset)

        x0_pts = np.tile(np.atleast_2d(MCA+c/4),(B,1))  
        x_pts  = np.repeat(np.repeat(x0_pts[np.newaxis,:,  :], nts, axis=0)[ np.newaxis, : ,:, :,], m, axis=0) 
        X_pts0 = x_pts + sx_inf   

        # compute wake contraction  
        wake_contraction = compute_wake_contraction_matrix(i,prop,Nr,m,nts,X_pts0)          

        y0_pts = np.tile(np.atleast_2d(r),(B,1))
        y_pts  = np.repeat(np.repeat(y0_pts[np.newaxis,:,  :], nts, axis=0)[ np.newaxis, : ,:, :,], m, axis=0) 
        Y_pts0 = (y_pts*wake_contraction)*azi_y  + sy_inf    

        z0_pts = np.tile(np.atleast_2d(r),(B,1))
        z_pts  = np.repeat(np.repeat(z0_pts[np.newaxis,:,  :], nts, axis=0)[ np.newaxis, : ,:, :,], m, axis=0)
        Z_pts0 = (z_pts*wake_contraction)*azi_z + sz_inf     
 
        # Rotate wake by thrust angle 
        X_pts  = prop.origin[i][0] + X_pts0*np.cos(-thrust_angle) - Z_pts0*np.sin(-thrust_angle)
        Y_pts  = prop.origin[i][1] + Y_pts0
        Z_pts  = prop.origin[i][2] + X_pts0*np.sin(-thrust_angle) + Z_pts0*np.cos(-thrust_angle) 

        # Store points  
        # ( control point,  prop ,  time step , blade number , location on blade )
        if (prop.rotation != None) and (prop.rotation[i] == -1):  
            WD_XA1[:,i,:,:,:] = X_pts[: , :-1 , : , :-1 ]
            WD_YA1[:,i,:,:,:] = Y_pts[: , :-1 , : , :-1 ]
            WD_ZA1[:,i,:,:,:] = Z_pts[: , :-1 , : , :-1 ]
            WD_XA2[:,i,:,:,:] = X_pts[: ,  1: , : , :-1 ]
            WD_YA2[:,i,:,:,:] = Y_pts[: ,  1: , : , :-1 ]
            WD_ZA2[:,i,:,:,:] = Z_pts[: ,  1: , : , :-1 ]
            WD_XB1[:,i,:,:,:] = X_pts[: , :-1 , : , 1:  ]
            WD_YB1[:,i,:,:,:] = Y_pts[: , :-1 , : , 1:  ]
            WD_ZB1[:,i,:,:,:] = Z_pts[: , :-1 , : , 1:  ]
            WD_XB2[:,i,:,:,:] = X_pts[: ,  1: , : , 1:  ]
            WD_YB2[:,i,:,:,:] = Y_pts[: ,  1: , : , 1:  ]
            WD_ZB2[:,i,:,:,:] = Z_pts[: ,  1: , : , 1:  ] 
        else: 
            WD_XA1[:,i,:,:,:] = X_pts[: , :-1 , : , 1:  ]
            WD_YA1[:,i,:,:,:] = Y_pts[: , :-1 , : , 1:  ]
            WD_ZA1[:,i,:,:,:] = Z_pts[: , :-1 , : , 1:  ]
            WD_XA2[:,i,:,:,:] = X_pts[: ,  1: , : , 1:  ]
            WD_YA2[:,i,:,:,:] = Y_pts[: ,  1: , : , 1:  ]
            WD_ZA2[:,i,:,:,:] = Z_pts[: ,  1: , : , 1:  ] 
            WD_XB1[:,i,:,:,:] = X_pts[: , :-1 , : , :-1 ]
            WD_YB1[:,i,:,:,:] = Y_pts[: , :-1 , : , :-1 ]
            WD_ZB1[:,i,:,:,:] = Z_pts[: , :-1 , : , :-1 ]
            WD_XB2[:,i,:,:,:] = X_pts[: ,  1: , : , :-1 ]
            WD_YB2[:,i,:,:,:] = Y_pts[: ,  1: , : , :-1 ]
            WD_ZB2[:,i,:,:,:] = Z_pts[: ,  1: , : , :-1 ]

        WD_GAMMA[:,i,:,:,:] = Gamma 

        # store points for plotting 
        VD.Wake.XA1[i,:,:,:] =  X_pts[0 , :-1 , : , :-1 ]
        VD.Wake.YA1[i,:,:,:] =  Y_pts[0 , :-1 , : , :-1 ]
        VD.Wake.ZA1[i,:,:,:] =  Z_pts[0 , :-1 , : , :-1 ]
        VD.Wake.XA2[i,:,:,:] =  X_pts[0 ,  1: , : , :-1 ]
        VD.Wake.YA2[i,:,:,:] =  Y_pts[0 ,  1: , : , :-1 ]
        VD.Wake.ZA2[i,:,:,:] =  Z_pts[0 ,  1: , : , :-1 ]
        VD.Wake.XB1[i,:,:,:] =  X_pts[0 , :-1 , : , 1:  ]
        VD.Wake.YB1[i,:,:,:] =  Y_pts[0 , :-1 , : , 1:  ]
        VD.Wake.ZB1[i,:,:,:] =  Z_pts[0 , :-1 , : , 1:  ]
        VD.Wake.XB2[i,:,:,:] =  X_pts[0 ,  1: , : , 1:  ]
        VD.Wake.YB2[i,:,:,:] =  Y_pts[0 ,  1: , : , 1:  ]
        VD.Wake.ZB2[i,:,:,:] =  Z_pts[0 ,  1: , : , 1:  ]  

    # Compress Data into 1D Arrays  
    mat4_size = (m,num_prop,(nts-1),B*n)
    mat5_size = (m,num_prop,(nts-1)*B*n)
    mat6_size = (m,num_prop*(nts-1)*B*n) 

    WD.XA1    =  np.reshape(np.reshape(np.reshape(WD_XA1,mat4_size),mat5_size),mat6_size)
    WD.YA1    =  np.reshape(np.reshape(np.reshape(WD_YA1,mat4_size),mat5_size),mat6_size)
    WD.ZA1    =  np.reshape(np.reshape(np.reshape(WD_ZA1,mat4_size),mat5_size),mat6_size)
    WD.XA2    =  np.reshape(np.reshape(np.reshape(WD_XA2,mat4_size),mat5_size),mat6_size)
    WD.YA2    =  np.reshape(np.reshape(np.reshape(WD_YA2,mat4_size),mat5_size),mat6_size)
    WD.ZA2    =  np.reshape(np.reshape(np.reshape(WD_ZA2,mat4_size),mat5_size),mat6_size)
    WD.XB1    =  np.reshape(np.reshape(np.reshape(WD_XB1,mat4_size),mat5_size),mat6_size)
    WD.YB1    =  np.reshape(np.reshape(np.reshape(WD_YB1,mat4_size),mat5_size),mat6_size)
    WD.ZB1    =  np.reshape(np.reshape(np.reshape(WD_ZB1,mat4_size),mat5_size),mat6_size)
    WD.XB2    =  np.reshape(np.reshape(np.reshape(WD_XB2,mat4_size),mat5_size),mat6_size)
    WD.YB2    =  np.reshape(np.reshape(np.reshape(WD_YB2,mat4_size),mat5_size),mat6_size)
    WD.ZB2    =  np.reshape(np.reshape(np.reshape(WD_ZB2,mat4_size),mat5_size),mat6_size)
    WD.GAMMA  =  np.reshape(np.reshape(np.reshape(WD_GAMMA,mat4_size),mat5_size),mat6_size)

    return WD, dt, ts, B, Nr 