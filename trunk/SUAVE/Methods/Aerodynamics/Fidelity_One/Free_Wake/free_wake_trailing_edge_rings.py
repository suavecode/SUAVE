## @ingroup Methods-Aerodynamics-Common-Fidelity_One-Free_Wake
#  free_wake_trailing_edge_rings.py
# 
# Created:  Oct 2021, R. ERhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np
from SUAVE.Core import Data 
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wake_contraction_matrix import compute_wake_contraction_matrix
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry   

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift   
def free_wake_trailing_edge_rings(prop,m,VD,init_timestep_offset, dt, number_of_wake_timesteps): 
    """ This generates the propeller wake control points used to compute the 
    influence of the wake

    Assumptions: 
    None

    Source:   
    None

    Inputs:  
    m                        - control point                     [Unitless] 
    VD                       - vortex distribution               
    prop                     - propeller/rotor data structure         
    init_timestep_offset     - intial time step                  [Unitless] 
    time                     - time                              [s]
    number_of_wake_timesteps - number of wake timesteps          [Unitless]

    Properties Used:
    N/A
    """    

    # Unpack bemt outputs for propeller
    prop_outputs = prop.outputs
    Bmax         = int(prop.number_of_blades)
    nmax         = int(len(prop.radius_distribution) - 1)
    
    num_prop = 1
    nts      = number_of_wake_timesteps
        
    # Initialize empty arrays with required sizes
    WD, Wmid = initialize_distributions(nmax, Bmax, nts, num_prop, m)

    # Unpack
    R                = prop.tip_radius
    r                = prop.radius_distribution 
    c                = prop.chord_distribution 
    B                = prop.number_of_blades    
    
    Na               = prop_outputs.number_azimuthal_stations
    Nr               = prop_outputs.number_radial_stations
    omega            = prop_outputs.omega                               
    va               = prop_outputs.disc_axial_induced_velocity 
    V_inf            = prop_outputs.velocity
    gamma            = prop_outputs.disc_circulation
    
    blade_angles     = np.linspace(0,2*np.pi,B+1)[:-1]   
    #dt               = time/number_of_wake_timesteps
    ts               = np.linspace(0,dt,number_of_wake_timesteps) 
    
    t0                = dt*init_timestep_offset
    start_angle       = omega[0]*t0 
    prop.start_angle = start_angle[0]

    # compute lambda and mu 
    mean_induced_velocity  = np.mean( np.mean(va,axis = 1),axis = 1)   

    alpha = prop.orientation_euler_angles[1]
    rots  = np.array([[np.cos(alpha), 0, np.sin(alpha)], [0,1,0], [-np.sin(alpha), 0, np.cos(alpha)]])
    
    lambda_tot   =  np.atleast_2d((np.dot(V_inf,rots[0])  + mean_induced_velocity)).T /(omega*R)   # inflow advance ratio (page 99 Leishman)
    mu_prop      =  np.atleast_2d(np.dot(V_inf,rots[2])).T /(omega*R)                              # rotor advance ratio  (page 99 Leishman) 
    V_prop       =  np.atleast_2d(np.sqrt((V_inf[:,0]  + mean_induced_velocity)**2 + (V_inf[:,2])**2)).T

    # wake skew angle 
    wake_skew_angle = -np.arctan(mu_prop/lambda_tot)
    
    # reshape gamma to find the average between stations 
    gamma_new = np.zeros((m,(Nr-1),Na))    # [control points, Nr-1, Na ] one less radial station because ring
    gamma_new = (gamma[:,:-1,:] + gamma[:,1:,:])*0.5

    
    num       = int(Na/B)  
    time_idx  = np.arange(nts)
    t_idx     = np.atleast_2d(time_idx).T 
    B_idx     = np.arange(B) 
    B_loc     = (B_idx*num + t_idx)%Na  
    Gamma     = gamma_new[:,:,B_loc]  
    Gamma     = Gamma.transpose(0,3,1,2)
    
    
    
    # --------------------------------------------------------------------------------------------------------------
    #    ( control point , blade number , radial location on blade , time step )
    # --------------------------------------------------------------------------------------------------------------
    sx_inf0            = np.multiply(V_prop*np.cos(wake_skew_angle), np.atleast_2d(ts))
    sx_inf             = np.repeat(np.repeat(sx_inf0[:, None, :], B, axis = 1)[:, :, None, :], Nr, axis = 2) 
                      
    sy_inf0            = np.multiply(np.atleast_2d(V_inf[:,1]).T,np.atleast_2d(ts)) # = zero since no crosswind
    sy_inf             = np.repeat(np.repeat(sy_inf0[:, None, :], B, axis = 1)[:, :, None, :], Nr, axis = 2)   
                      
    sz_inf0            = np.multiply(V_prop*np.sin(wake_skew_angle),np.atleast_2d(ts))
    sz_inf             = np.repeat(np.repeat(sz_inf0[:, None, :], B, axis = 1)[:, :, None, :], Nr, axis = 2)           

    angle_offset       = np.repeat(np.repeat(np.multiply(omega,np.atleast_2d(ts))[:, None, :],B, axis = 1)[:, :, None, :],Nr, axis = 2) 
    blade_angle_loc    = np.repeat(np.repeat(np.tile(np.atleast_2d(blade_angles),(m,1))[:, :, None ], Nr, axis = 2)[:, :, :, None],number_of_wake_timesteps, axis = 3) 
    start_angle_offset = np.repeat(np.repeat(np.atleast_2d(start_angle)[:, None, :],B, axis = 1)[:, :, None, :],Nr, axis = 2) 
    
    total_angle_offset = angle_offset - start_angle_offset
    
    if (prop.rotation != None) and (prop.rotation == 1):        
        total_angle_offset = -total_angle_offset

    azi_y   = np.sin(blade_angle_loc + total_angle_offset)  
    azi_z   = np.cos(blade_angle_loc + total_angle_offset)
    

    # extract airfoil trailing edge coordinates for initial location of vortex wake
    a_sec        = prop.airfoil_geometry   
    a_secl       = prop.airfoil_polar_stations
    airfoil_data = import_airfoil_geometry(a_sec,npoints=100)  
   
    # trailing edge points in airfoil coordinates
    xupper         = np.take(airfoil_data.x_upper_surface,a_secl,axis=0)   
    yupper         = np.take(airfoil_data.y_upper_surface,a_secl,axis=0)   
    
    # Align the quarter chords of the airfoils (zero sweep)
    airfoil_le_offset = (c[0]/2 - c/2 )
    xte_airfoils      = xupper[:,-1]*c + airfoil_le_offset
    yte_airfoils      = yupper[:,-1]*c 
    
    # apply blade twist rotation along rotor radius
    beta = prop.twist_distribution
    xte_twisted = np.cos(beta)*xte_airfoils - np.sin(beta)*yte_airfoils        
    yte_twisted = np.sin(beta)*xte_airfoils + np.cos(beta)*yte_airfoils    
    
    
    # transform coordinates from airfoil frame to rotor frame
    xte = np.tile(np.atleast_2d(yte_twisted), (B,1))
    xte_rotor = np.tile(xte[None,:,:,None], (m,1,1,number_of_wake_timesteps))
    yte_rotor = -np.tile(xte_twisted[None,None,:,None],(m,B,1,1))*np.cos(blade_angle_loc+total_angle_offset) 
    zte_rotor = np.tile(xte_twisted[None,None,:,None],(m,B,1,1))*np.sin(blade_angle_loc+total_angle_offset)
         

    r_4d = np.tile(r[None,None,:,None], (m,B,1,number_of_wake_timesteps))
    
    x0 = 0
    y0 = r_4d*azi_y
    z0 = r_4d*azi_z
    
    x_pts0 = x0 + xte_rotor
    y_pts0 = y0 + yte_rotor
    z_pts0 = z0 + zte_rotor
         

    # compute wake contraction, apply to y-z plane
    X_pts0           = x_pts0 + sx_inf
    wake_contraction = compute_wake_contraction_matrix(0,prop,Nr,m,number_of_wake_timesteps,X_pts0,prop_outputs) 
    Y_pts0           = y_pts0*wake_contraction + sy_inf
    Z_pts0           = z_pts0*wake_contraction + sz_inf

    # Rotate wake by thrust angle
    rot_to_body = prop.prop_vel_to_body()  # rotate points into the body frame: [Z,Y,X]' = R*[Z,Y,X]
    
    # append propeller wake to each of its repeated origins  
    X_pts   = prop.origin[0][0] + X_pts0*rot_to_body[2,2] + Z_pts0*rot_to_body[2,0]   
    Y_pts   = prop.origin[0][1] + Y_pts0*rot_to_body[1,1]                       
    Z_pts   = prop.origin[0][2] + Z_pts0*rot_to_body[0,0] + X_pts0*rot_to_body[0,2] 
            
            
            
    
    ## ----------------------------------------------------------------
    ## A1 and A2 get trailing edge points from rotor 
    ## ----------------------------------------------------------------    
    XA1 = X_pts[:,:, :-1, 0]   # (ctrl_pt, B, Nr-1)
    XA2 = X_pts[:,:, 1: , 0]
    YA1 = Y_pts[:,:, :-1, 0]
    YA2 = Y_pts[:,:, 1: , 0]       
    ZA1 = Z_pts[:,:, :-1, 0]
    ZA2 = Z_pts[:,:, 1: , 0]     
    

    ## ----------------------------------------------------------------
    ## B1 and B2 get convected downstream
    ## ----------------------------------------------------------------   
    
    XB1 = X_pts[:,:, :-1, 1]
    XB2 = X_pts[:,:, 1: , 1]
    YB1 = Y_pts[:,:, :-1, 1]
    YB2 = Y_pts[:,:, 1: , 1]      
    ZB1 = Z_pts[:,:, :-1, 1]
    ZB2 = Z_pts[:,:, 1: , 1]     
    
    GAMMA = Gamma[:,:,:,0]

    # store points for plotting 
    if len(VD.Wake.XA1)==0:
        VD.Wake.XA1 = XA1[:,:,:,None]
        VD.Wake.YA1 = YA1[:,:,:,None]
        VD.Wake.ZA1 = ZA1[:,:,:,None]
        VD.Wake.XA2 = XA2[:,:,:,None]
        VD.Wake.YA2 = YA2[:,:,:,None]
        VD.Wake.ZA2 = ZA2[:,:,:,None]
        VD.Wake.XB1 = XB1[:,:,:,None]
        VD.Wake.YB1 = YB1[:,:,:,None]
        VD.Wake.ZB1 = ZB1[:,:,:,None]
        VD.Wake.XB2 = XB2[:,:,:,None]
        VD.Wake.YB2 = YB2[:,:,:,None]
        VD.Wake.ZB2 = ZB2[:,:,:,None]
        
        VD.Wake.GAMMA = GAMMA[:,:,:,None]
    else:   
        VD.Wake.XA1 = np.append(VD.Wake.XA1,XA1[:,:,:,None], axis=3)
        VD.Wake.YA1 = np.append(VD.Wake.YA1,YA1[:,:,:,None], axis=3)
        VD.Wake.ZA1 = np.append(VD.Wake.ZA1,ZA1[:,:,:,None], axis=3)
        VD.Wake.XA2 = np.append(VD.Wake.XA2,XA2[:,:,:,None], axis=3)
        VD.Wake.YA2 = np.append(VD.Wake.YA2,YA2[:,:,:,None], axis=3)
        VD.Wake.ZA2 = np.append(VD.Wake.ZA2,ZA2[:,:,:,None], axis=3)
        VD.Wake.XB1 = np.append(VD.Wake.XB1,XB1[:,:,:,None], axis=3)
        VD.Wake.YB1 = np.append(VD.Wake.YB1,YB1[:,:,:,None], axis=3)
        VD.Wake.ZB1 = np.append(VD.Wake.ZB1,ZB1[:,:,:,None], axis=3)
        VD.Wake.XB2 = np.append(VD.Wake.XB2,XB2[:,:,:,None], axis=3)
        VD.Wake.YB2 = np.append(VD.Wake.YB2,YB2[:,:,:,None], axis=3)
        VD.Wake.ZB2 = np.append(VD.Wake.ZB2,ZB2[:,:,:,None], axis=3)
        VD.Wake.GAMMA = np.append(VD.Wake.GAMMA,GAMMA[:,:,:,None], axis=3)
    
    
    VD.Wake.XC = (VD.Wake.XA2 + VD.Wake.XA1 + VD.Wake.XB2 + VD.Wake.XB1)/4
    VD.Wake.YC = (VD.Wake.YA2 + VD.Wake.YA1 + VD.Wake.YB2 + VD.Wake.YB1)/4
    VD.Wake.ZC = (VD.Wake.ZA2 + VD.Wake.ZA1 + VD.Wake.ZB2 + VD.Wake.ZB1)/4
    
    # Compress Data into 1D Arrays  
    mat6_size = (m,np.size(VD.Wake.XC)) 

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
    
    WD.XC    =  np.reshape(VD.Wake.XC,mat6_size)
    WD.YC    =  np.reshape(VD.Wake.YC,mat6_size)
    WD.ZC    =  np.reshape(VD.Wake.ZC,mat6_size)
    
    WD.GAMMA  =  np.reshape(VD.Wake.GAMMA,mat6_size)
    
    
    VD.Wake_collapsed = WD

    return VD.Wake, WD


def initialize_distributions(nmax, Bmax, n_wts, n_props, m):
    
    Wmid        = Data()
    mat1_size = (m,n_props,Bmax,nmax, n_wts)
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
    mat2_size = (m,n_props*n_wts*Bmax*nmax)
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
    
    return WD, Wmid