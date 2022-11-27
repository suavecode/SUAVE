## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# compute_source_coordinates.py
# 
# Created:  Mar 2021, M. Clarke
# Modified: Feb 2022, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------
from SUAVE.Core import to_jnumpy
from jax import  jit
import jax.numpy as jnp 
import numpy as np 
from SUAVE.Core.Utilities import jjv
from SUAVE.Core import Data
import scipy as sp
# ----------------------------------------------------------------------
#  Source Coordinates 
# ---------------------------------------------------------------------

## @ingroup Methods-Noise-Fidelity_One-Propeller 
@jit
def compute_point_source_coordinates(conditions,rotors,mls,settings):  
    """This calculated the position vector from a point source to the observer 
            
    Assumptions:
        N/A

    Source:
        N/A 
        
    Inputs:  
        conditions        - flight conditions            [None]  
        mls               - microphone locations         [m] 
        rotors            - rotors on network            [None]  
        settings          - noise calculation settings   [None]
        
    Outputs: 
        position vector   - position vector of points    [m]
        
    Properties Used:
        N/A       
    """  
    
    # aquire dimension of matrix
    num_cpt     = conditions._size
    num_mic     = len(mls[0,:,0])  
    num_rot     = len(rotors)  
    rot_origins = []
    for rotor in rotors:
        rot_origins.append(rotor.origin[0])
    rot_origins = jnp.array(rot_origins)  
        
    # Get the rotation matrix
    prop2body   = rotor.prop_vel_to_body()

    # [control point, microphone , propeller , 2D geometry matrix ]
    # rotation of propeller about y axis by thrust angle (one extra dimension for translations)
    rotation_1    = jnp.zeros((num_cpt,4,4))
    rotation_1    = rotation_1.at[:,0:3,0:3].set(prop2body)  
    rotation_1    = rotation_1.at[:,3,3].set(1)   
    rotation_1    = jnp.repeat(rotation_1[:,None,:,:], num_rot, axis=1)
    rotation_1    = jnp.repeat(rotation_1[:,None,:,:,:], num_mic, axis=1)

    # translation to location on propeller
    I             = jnp.atleast_3d(jnp.eye(4)).T
    translation_1 = jnp.tile(I[None,None,:,:,:],(num_cpt,num_mic,num_rot,1,1))
    translation_1 = translation_1.at[:,:,:,0,3].set(to_jnumpy(np.tile(rot_origins[:,0][None,None,:],(num_cpt,num_mic,1))))
    translation_1 = translation_1.at[:,:,:,1,3].set(to_jnumpy(np.tile(rot_origins[:,1][None,None,:],(num_cpt,num_mic,1))))
    translation_1 = translation_1.at[:,:,:,2,3].set(to_jnumpy(np.tile(rot_origins[:,2][None,None,:],(num_cpt,num_mic,1))))

    # rotation of vehicle about y axis by AoA 
    rotation_2    = jnp.zeros((num_cpt,num_mic,num_rot,4,4))
    rotation_2    = rotation_2.at[0:num_cpt,:,:,0:3,0:3].set(conditions.frames.body.transform_to_inertial[:,jnp.newaxis,jnp.newaxis,:,:])
    rotation_2    = rotation_2.at[:,:,:,3,3].set(1) 

    # translation of vehicle to air  
    translation_2 = jnp.tile(I[None,None,:,:,:],(num_cpt,num_mic,num_rot,1,1))    
    translation_2 = translation_2.at[:,:,:,0,3].set(jnp.tile(mls[:,:,0][:,:,None],(1,1,num_rot)))
    translation_2 = translation_2.at[:,:,:,1,3].set(jnp.tile(mls[:,:,1][:,:,None],(1,1,num_rot))) 
    translation_2 = translation_2.at[:,:,:,2,3].set(jnp.tile(mls[:,:,2][:,:,None],(1,1,num_rot))) 

    # identity transformation 
    I0    = jnp.atleast_3d(jnp.array([[0,0,0,1]]))
    I0    = jnp.array(I0)  
    mat_0 = jnp.tile(I0[None,None,:,:,:],(num_cpt,num_mic,num_rot,1,1))

    # execute operation  
    mat_1 = jnp.matmul(rotation_1,mat_0) 
    mat_2 = jnp.matmul(translation_1,mat_1)
    mat_3 = jnp.matmul(rotation_2,mat_2) 
    mat_4 = jnp.matmul(translation_2,mat_3)
    mat_4 = -mat_4

    # store points
    propeller_position_vector = jnp.zeros((num_cpt,num_mic,num_rot,3))
    propeller_position_vector = propeller_position_vector.at[:,:,:,0].set(mat_4[:,:,:,0,0])
    propeller_position_vector = propeller_position_vector.at[:,:,:,1].set(mat_4[:,:,:,1,0])
    propeller_position_vector = propeller_position_vector.at[:,:,:,2].set(mat_4[:,:,:,2,0])
     
    return propeller_position_vector

## @ingroup Methods-Noise-Fidelity_One-Noise_Tools  @jut
@jit
def compute_blade_section_source_coordinates(AoA,acoustic_outputs,rotors,mls,settings):  
    """This calculated the position vector from a point source to the observer 
            
    Assumptions:
        N/A
 
    Source:
        N/A  
 
    Inputs:  
        AoA                            - aircraft angle ofattack                     [rad]
        acoustic_outputs               - outputs from propeller aerodynamic analysis [None]   
        mls                            - microphone locations                        [m]   
        rotors                         - rotors on network                           [None]  
        settings                       - noise calculation settings                  [None]
 
    Outputs: 
        blade_section_position_vectors - position vector of rotor blade sections     [m]
 
    Properties Used:
        N/A       
    """
    
    # aquire dimension of matrix 
    num_cpt     = len(AoA)
    num_mic     = len(mls[0,:,0])   
    num_rot     = len(rotors)  
    rot_origins = []
    for rotor in rotors:
        rot_origins.append(rotor.origin[0])
    rot_origins = to_jnumpy(np.array(rot_origins))
            
    rotor          = rotors[list(rotors.keys())[0]]
    num_cf         = len(settings.center_frequencies)
    r              = to_jnumpy(rotor.radius_distribution)
    num_sec        = len(r)
    phi_2d0        = to_jnumpy(acoustic_outputs.disc_azimuthal_distribution)
    alpha_eff0     = to_jnumpy(acoustic_outputs.disc_effective_angle_of_attack)
    num_azi        = len(phi_2d0[0,0,:])  
    orientation    = jnp.array(rotor.orientation_euler_angles) * 1 
    orientation[1] = orientation[1] + jnp.pi/2 # rotor tilt angle between the rotor hub plane and the vehicle hub plane
    body2thrust    = sp.spatial.transform.Rotation.from_rotvec(orientation).as_matrix()

    # Update dimensions for computation   
    r                    = jnp.tile(r[None,None,None,:,None,None],(num_cpt,num_mic,num_rot,1,num_azi,num_cf))
    sin_phi              = jnp.tile(jnp.sin(phi_2d0)[:,None,None,:,:,None,None],(1,num_mic,num_rot,1,1,num_cf,1))
    cos_phi              = jnp.tile(jnp.cos(phi_2d0)[:,None,None,:,:,None,None],(1,num_mic,num_rot,1,1,num_cf,1))
    sin_alpha_eff        = jnp.tile(jnp.sin(alpha_eff0)[:,None,None,:,:,None,None],(1,num_mic,num_rot,1,1,num_cf,1))
    cos_alpha_eff        = jnp.tile(jnp.cos(alpha_eff0)[:,None,None,:,:,None,None],(1,num_mic,num_rot,1,1,num_cf,1))
    cos_t_v              = jnp.tile(jnp.cos(-to_jnumpy(AoA))[:,None,None,None,None,None,:],(1,num_mic,num_rot,num_sec,num_azi,num_cf,1))
    sin_t_v              = jnp.tile(jnp.sin(-to_jnumpy(AoA))[:,None,None,None,None,None,:],(1,num_mic,num_rot,num_sec,num_azi,num_cf,1))  
    cos_t_v_t_r          = jnp.tile(to_jnumpy(np.array([body2thrust[0,0]]))[:,None,None,None,None,None,None],(num_cpt,num_mic,num_rot,num_sec,num_azi,num_cf,1))  
    sin_t_v_t_r          = jnp.tile(to_jnumpy(np.array([body2thrust[0,2]]))[:,None,None,None,None,None,None],(num_cpt,num_mic,num_rot,num_sec,num_azi,num_cf,1))  
    M_hub                = jnp.tile(rot_origins[None,None,:,None,None,None,:,None],(num_cpt,num_mic,1,num_sec,num_azi,num_cf,1,1))
    POS_2                = jnp.tile(to_jnumpy(mls)[:,:,None,None,None,None,:,None],(1,1,num_rot,num_sec,num_azi,num_cf,1,1))

    # ------------------------------------------------------------
    # ****** COORDINATE TRANSFOMRATIONS ******  
    M_t      = jnp.zeros((num_cpt,num_mic,num_rot,num_sec,num_azi,num_cf,3,3))
    M_phi    = jnp.zeros((num_cpt,num_mic,num_rot,num_sec,num_azi,num_cf,3,3))
    M_theta  = jnp.zeros((num_cpt,num_mic,num_rot,num_sec,num_azi,num_cf,3,3))
    M_tv     = jnp.zeros((num_cpt,num_mic,num_rot,num_sec,num_azi,num_cf,3,3))

    M_tv     = M_tv.at[:,:,:,:,:,:,0,0].set( cos_t_v[:,:,:,:,:,:,0])
    M_tv     = M_tv.at[:,:,:,:,:,:,0,2].set( sin_t_v[:,:,:,:,:,:,0])
    M_tv     = M_tv.at[:,:,:,:,:,:,1,1].set( 1)
    M_tv     = M_tv.at[:,:,:,:,:,:,2,0].set(-sin_t_v[:,:,:,:,:,:,0])
    M_tv     = M_tv.at[:,:,:,:,:,:,2,2].set( cos_t_v[:,:,:,:,:,:,0])
    POS_1    = jnp.matmul(M_tv,(POS_2 + M_hub)) # rotor hub position relative to center of aircraft

    # twist angle matrix
    M_theta  = M_theta.at[:,:,:,:,:,:,0,0].set( cos_alpha_eff[:,:,:,:,:,:,0])
    M_theta  = M_theta.at[:,:,:,:,:,:,0,2].set( sin_alpha_eff[:,:,:,:,:,:,0])
    M_theta  = M_theta.at[:,:,:,:,:,:,1,1].set(1)
    M_theta  = M_theta.at[:,:,:,:,:,:,2,0].set(-sin_alpha_eff[:,:,:,:,:,:,0])
    M_theta  = M_theta.at[:,:,:,:,:,:,2,2].set( cos_alpha_eff[:,:,:,:,:,:,0])

    # azimuth motion matrix
    M_phi    = M_phi.at[:,:,:,:,:,:,0,0].set(  sin_phi[:,:,:,:,:,:,0])
    M_phi    = M_phi.at[:,:,:,:,:,:,0,1].set( -cos_phi[:,:,:,:,:,:,0])
    M_phi    = M_phi.at[:,:,:,:,:,:,1,0].set(  cos_phi[:,:,:,:,:,:,0])
    M_phi    = M_phi.at[:,:,:,:,:,:,1,1].set(  sin_phi[:,:,:,:,:,:,0])
    M_phi    = M_phi.at[:,:,:,:,:,:,2,2].set( 1)

    # tilt motion matrix 
    M_t      = M_t.at[:,:,:,:,:,:,0,0].set(  cos_t_v_t_r[:,:,:,:,:,:,0])
    M_t      = M_t.at[:,:,:,:,:,:,0,2].set(  sin_t_v_t_r[:,:,:,:,:,:,0])
    M_t      = M_t.at[:,:,:,:,:,:,1,1].set(  1)
    M_t      = M_t.at[:,:,:,:,:,:,2,0].set( -sin_t_v_t_r[:,:,:,:,:,:,0])
    M_t      = M_t.at[:,:,:,:,:,:,2,2].set(  cos_t_v_t_r[:,:,:,:,:,:,0])
    
    # transformation of geographical global reference frame to the sectional local coordinate
    mat0    = jnp.matmul(M_t,POS_1)   
    mat1    = jnp.matmul(M_phi,mat0)
    POS     = jnp.matmul(M_theta,mat1)

    blade_section_position_vectors = Data()
    blade_section_position_vectors.blade_section_coordinate_sys    = POS 
    blade_section_position_vectors.vehicle_coordinate_sys          = POS_2
    blade_section_position_vectors.cos_phi                         = jnp.repeat(cos_phi,2,axis = 6)  
    blade_section_position_vectors.sin_alpha_eff                   = jnp.repeat(sin_alpha_eff,2,axis = 6)     
    blade_section_position_vectors.cos_alpha_eff                   = jnp.repeat(cos_alpha_eff,2,axis = 6) 
    blade_section_position_vectors.M_hub_X                         = jnp.repeat(M_hub[:,:,:,:,:,:,0,:],2,axis = 6)
    blade_section_position_vectors.M_hub_Y                         = jnp.repeat(M_hub[:,:,:,:,:,:,2,:],2,axis = 6)
    blade_section_position_vectors.M_hub_Z                         = jnp.repeat(M_hub[:,:,:,:,:,:,2,:],2,axis = 6)
    blade_section_position_vectors.cos_t_v                         = jnp.repeat(cos_t_v,2,axis = 6)
    blade_section_position_vectors.sin_t_v                         = jnp.repeat(sin_t_v,2,axis = 6)
    blade_section_position_vectors.cos_t_v_t_r                     = jnp.repeat(cos_t_v_t_r,2,axis = 6)
    blade_section_position_vectors.sin_t_v_t_r                     = jnp.repeat(sin_t_v_t_r,2,axis = 6)

    return blade_section_position_vectors