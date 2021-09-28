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
    gm_phi      = settings.ground_microphone_phi_angles 
    gm_theta    = settings.ground_microphone_theta_angles  
    Y           = np.linspace(0.01,y,N)  
    ctrl_pts    = segment.state.numerics.number_control_points  
    
    dim_alt     = len(alt)
    dim_phi     = len(gm_phi)  
    dim_theta   = len(gm_theta)
    num_mic     = dim_phi*dim_theta 
    
    if sma: # static microphone locations  
        # dimension:[control point, theta, phi]
        theta    = np.repeat(np.repeat(np.atleast_2d(gm_theta).T  ,dim_phi  , axis = 1)[np.newaxis,:,:],dim_alt, axis = 0)   
        altitude = np.repeat(np.repeat(np.atleast_2d(alt).T  ,dim_theta, axis = 1)[:,:,np.newaxis],dim_phi, axis = 2) 
        x_vals   = altitude/np.tan(theta) 
        y_vals   = np.repeat(np.repeat(np.atleast_2d(Y) ,dim_theta, axis = 0)[np.newaxis,:,:],dim_alt, axis = 0)  
        z_vals   = altitude    
    
    else: # dynamic microhpone locations  
        X        = np.linspace(pos[:,0][0],pos[:,0][-1],N)  
        altitude = np.repeat(np.repeat(np.atleast_2d(alt).T  ,dim_theta, axis = 1)[:,:,np.newaxis],dim_phi, axis = 2) 
        x_vals   = np.repeat(np.repeat(np.atleast_2d(X).T ,dim_phi  , axis = 1)[np.newaxis,:,:],dim_alt, axis = 0)   
        y_vals   = np.repeat(np.repeat(np.atleast_2d(Y) ,dim_theta, axis = 0)[np.newaxis,:,:],dim_alt, axis = 0) 
        z_vals   = altitude       
        
    # store microphone locations  
    PHI                  = np.repeat(np.atleast_2d(gm_phi),ctrl_pts,axis=0)
    THETA                = np.repeat(np.atleast_2d(gm_theta),ctrl_pts,axis=0) 
    mic_locations        = np.zeros((dim_alt,num_mic,3))   
    mic_locations[:,:,0] = x_vals.reshape(dim_alt,num_mic) 
    mic_locations[:,:,1] = y_vals.reshape(dim_alt,num_mic) 
    mic_locations[:,:,2] = z_vals.reshape(dim_alt,num_mic) 
    
    if type(ucml) is np.ndarray: # urban canyon microphone locations
        UCML           = np.repeat(ucml[np.newaxis,:,:],ctrl_pts,axis=0)
        num_b_mic      = len(ucml)
        UCML_ALT       = np.repeat(np.atleast_2d(alt).T,num_b_mic , axis = 1) - UCML[:,:,-2]
        b_theta        = np.arctan(UCML_ALT/UCML[:,:,0]) 
        b_phi          = np.arctan(UCML_ALT/UCML[:,:,1])  
        mic_locations  = np.concatenate((mic_locations,UCML),axis = 1) 
        THETA          = np.concatenate((THETA,b_theta),axis = 1) 
        PHI            = np.concatenate((PHI,b_phi),axis = 1)            
        num_mic       += num_b_mic
     
    return THETA,PHI,mic_locations,num_mic 