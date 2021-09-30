## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# compute_noise_evaluation_locations.py
# 
# Created: Sep 2021, M. Clarke  

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
#  Compute Point Source Coordinates 
# ---------------------------------------------------------------------

## @ingroupMethods-Noise-Fidelity_One-Noise_Tools 
def compute_noise_evaluation_locations(settings,segment):
    """This computes all the locations in the computational domain where the 
    propogated sound is computed.   
            
    Assumptions: 
        Acoustic scattering is not modeled

    Source:
        N/A  

    Inputs:  
        settings. 
            ground_microphone_phi_angles                    - azimuth measured from observer to aircraft body frame     [radians]
            ground_microphone_theta_angles                  - axial angle measured from observer to aircraft body frame [radians] 
            microphone_array_dimension                      - flag for fixing lateral microphone distance               [boolean]
            lateral_ground_distance                         - maximum ground distance in the +/- y-direction            [meters]
            urban_canyon_microphone_locations               - array of microphone locations on a building               [meters]
            static_microphone_array                         - flag for fixing microphone location in interial frame     [boolean] 
        segment.conditions.frames.inertial.position_vector  - position of aircraft                                      [boolean]                                     

    Outputs: 
    
    
    Properties Used:
        N/A       
    """       
    
    conditions  = segment.state.conditions  
    N           = settings.microphone_array_dimension
    y           = settings.lateral_ground_distance
    ucml        = settings.urban_canyon_microphone_locations
    sma         = settings.static_microphone_array 
    pos         = conditions.frames.inertial.position_vector
    alt         = -pos[:,2]     
    gm_theta    = settings.ground_microphone_theta_angles  
    Y           = np.linspace(-y,y,N)  
    ctrl_pts    = segment.state.numerics.number_control_points  
    
    dim_alt     = len(alt)
    dim_phi     = N
    dim_theta   = N
    num_gm_mic  = dim_phi*dim_theta 
    
    if sma: # static microphone locations  
        # dimension:[control point, theta, phi]
        theta    = np.repeat(np.repeat(np.atleast_2d(gm_theta).T  ,dim_phi  , axis = 1)[np.newaxis,:,:],dim_alt, axis = 0)   
        altitude = np.repeat(np.repeat(np.atleast_2d(alt).T  ,dim_theta, axis = 1)[:,:,np.newaxis],dim_phi, axis = 2) 
        x_vals   = altitude/np.tan(theta) 
        y_vals   = np.repeat(np.repeat(np.atleast_2d(Y) ,dim_theta, axis = 0)[np.newaxis,:,:],dim_alt, axis = 0) 
        gm_phi   = np.arctan(altitude/y_vals)  
        z_vals   = altitude    
    
    else: # dynamic microhpone locations  
        X        = np.linspace(pos[:,0][0],pos[:,0][-1],N)  
        altitude = np.repeat(np.repeat(np.atleast_2d(alt).T  ,dim_theta, axis = 1)[:,:,np.newaxis],dim_phi, axis = 2) 
        x_vals   = np.repeat(np.repeat(np.atleast_2d(X).T ,dim_phi  , axis = 1)[np.newaxis,:,:],dim_alt, axis = 0)   
        y_vals   = np.repeat(np.repeat(np.atleast_2d(Y) ,dim_theta, axis = 0)[np.newaxis,:,:],dim_alt, axis = 0) 
        gm_phi   = np.arctan(altitude/y_vals)  
        z_vals   = altitude       
        
    # store microphone locations  
    GM_PHI     = gm_phi.reshape(dim_alt,num_gm_mic) 
    GM_THETA   = np.repeat(np.atleast_2d(gm_theta),ctrl_pts,axis=0) 
    GML        = np.zeros((dim_alt,num_gm_mic,3))   
    GML[:,:,0] = x_vals.reshape(dim_alt,num_gm_mic) 
    GML[:,:,1] = y_vals.reshape(dim_alt,num_gm_mic) 
    GML[:,:,2] = z_vals.reshape(dim_alt,num_gm_mic) 
     
    if type(ucml) is np.ndarray: # urban canyon microphone locations
        num_b_mic      = len(ucml)
        ucml_3d        = np.repeat(ucml[np.newaxis,:,:],ctrl_pts,axis=0)
        UCML           = np.zeros_like(ucml_3d) 
        Aircraft_x     = np.repeat(np.atleast_2d(pos[:,0] ).T,num_b_mic , axis = 1)
        Aircraft_y     = np.repeat(np.atleast_2d(pos[:,1]).T,num_b_mic , axis = 1)
        Aircraft_z     = np.repeat(np.atleast_2d(alt).T,num_b_mic , axis = 1)
        UCML[:,:,0]    = Aircraft_x - ucml_3d[:,:,0] 
        UCML[:,:,1]    = Aircraft_y - ucml_3d[:,:,1]        
        UCML[:,:,2]    = Aircraft_z - ucml_3d[:,:,2]
        BM_THETA       = np.arctan(UCML[:,:,2]/UCML[:,:,0]) 
        BM_PHI         = np.arctan(UCML[:,:,2]/UCML[:,:,1])    
    else:
        UCML           = np.empty(shape=[ctrl_pts,0,3])
        BM_THETA       = np.empty(shape=[ctrl_pts,0]) 
        BM_PHI         = np.empty(shape=[ctrl_pts,0])   
        num_b_mic      = 0
        
     
    return GM_THETA,BM_THETA,GM_PHI,BM_PHI,GML,UCML,num_gm_mic,num_b_mic