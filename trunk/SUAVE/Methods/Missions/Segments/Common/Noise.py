## @ingroup Methods-Missions-Segments-Common
# Noise.py
# 
# Created:  Oct 2020, M. Clarke

# ----------------------------------------------------------------------
#  Update Noise
# ----------------------------------------------------------------------
import numpy as np 

## @ingroup Methods-Missions-Segments-Common
def compute_noise(segment):
    """ Evaluates the energy network to find the thrust force and mass rate

        Inputs -
            segment.analyses.energy_network    [Function]

        Outputs -
            state.conditions:
               frames.body.thrust_force_vector [Newtons]
               weights.vehicle_mass_rate       [kg/s]


        Assumptions -


    """    
    
    # unpack
    conditions  = segment.state.conditions 
    alt         = -conditions.frames.inertial.position_vector[:,2] 
    dist        = conditions.frames.inertial.position_vector[:,0] 
    gma         = segment.ground_microphone_angles
    noise_model = segment.analyses.noise
    
    dim_alt = len(alt)
    dim_mic = len(gma)  
    
    angles   = np.repeat(np.atleast_2d(gma), dim_alt, axis = 0)
    altitude = np.repeat(np.atleast_2d(alt).T, dim_mic, axis = 1)
    
    mic_locations        = np.zeros((dim_alt,dim_mic,3)) 
    mic_locations[:,:,1] = np.tan(angles)*altitude  
    
    conditions.noise.microphone_angles     = gma
    conditions.noise.microphone_locations  = mic_locations
    conditions.noise.number_of_microphones = dim_mic 
 
    if noise_model:
        noise_model.evaluate_noise(conditions)    