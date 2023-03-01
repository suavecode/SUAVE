## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# compute_source_coordinates.py
# 
# Created:  Mar 2021, M. Clarke
# Modified: Feb 2022, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------
import numpy as np 
from MARC.Core import Data, Units
import scipy as sp
# ----------------------------------------------------------------------
#  Source Coordinates 
# ---------------------------------------------------------------------

## @ingroup Methods-Noise-Fidelity_One-Propeller 
def compute_rotor_point_source_coordinates(conditions,rotors,mls,settings):  
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
    rot_origins = np.array(rot_origins)  
        
    # Get the rotation matrix
    prop2body   = rotor.prop_vel_to_body()

    # [control point, microphone , propeller , 2D geometry matrix ]
    # rotation of propeller about y axis by thrust angle (one extra dimension for translations)
    rotation_1            = np.zeros((num_cpt,4,4))
    rotation_1[:,0:3,0:3] = prop2body   
    rotation_1[:,3,3]     = 1     
    rotation_1            = np.repeat(rotation_1[:,None,:,:], num_rot, axis=1)
    rotation_1            = np.repeat(rotation_1[:,None,:,:,:], num_mic, axis=1)

    # translation to location on propeller
    I                         = np.atleast_3d(np.eye(4)).T
    translation_1             = np.tile(I[None,None,:,:,:],(num_cpt,num_mic,num_rot,1,1))
    translation_1[:,:,:,0,3]  = np.tile(rot_origins[:,0][None,None,:],(num_cpt,num_mic,1))
    translation_1[:,:,:,1,3]  = np.tile(rot_origins[:,1][None,None,:],(num_cpt,num_mic,1))     
    translation_1[:,:,:,2,3]  = np.tile(rot_origins[:,2][None,None,:],(num_cpt,num_mic,1))

    # rotation of vehicle about y axis by AoA 
    rotation_2                        = np.zeros((num_cpt,num_mic,num_rot,4,4))
    rotation_2[0:num_cpt,:,:,0:3,0:3] = conditions.frames.body.transform_to_inertial[:,np.newaxis,np.newaxis,:,:]
    rotation_2[:,:,:,3,3]             = 1   

    # translation of vehicle to air  
    translation_2               = np.tile(I[None,None,:,:,:],(num_cpt,num_mic,num_rot,1,1))    
    translation_2[:,:,:,0,3]    = np.tile(mls[:,:,0][:,:,None],(1,1,num_rot)) 
    translation_2[:,:,:,1,3]    = np.tile(mls[:,:,1][:,:,None],(1,1,num_rot)) 
    translation_2[:,:,:,2,3]    = np.tile(mls[:,:,2][:,:,None],(1,1,num_rot))  

    # identity transformation 
    I0    = np.atleast_3d(np.array([[0,0,0,1]]))
    I0    = np.array(I0)  
    mat_0 = np.tile(I0[None,None,:,:,:],(num_cpt,num_mic,num_rot,1,1))

    # execute operation  
    mat_1 = np.matmul(rotation_1,mat_0) 
    mat_2 = np.matmul(translation_1,mat_1)
    mat_3 = np.matmul(rotation_2,mat_2)   

    # store points
    propeller_position_vector          = np.zeros((num_cpt,num_mic,num_rot,3))
    propeller_position_vector[:,:,:,0] = -np.matmul(translation_2,mat_3)[:,:,:,0,0]
    propeller_position_vector[:,:,:,1] = -np.matmul(translation_2,mat_3)[:,:,:,1,0]
    propeller_position_vector[:,:,:,2] = -np.matmul(translation_2,mat_3)[:,:,:,2,0]
     
    return propeller_position_vector

## @ingroup Methods-Noise-Fidelity_One-Propeller 
def new_compute_rotor_point_source_coordinates(conditions,rotors,mls,settings):  
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
    rot_origins = np.array(rot_origins)    

    rotor          = rotors[list(rotors.keys())[0]] 
    num_blades     = rotor.number_of_blades  
    num_sec        = len(rotor.radius_distribution)   
        
    
    # Get the rotation matrix
    prop2body   = rotor.prop_vel_to_body() 
    body2prop   = rotor.body_to_prop_vel()

    phi            = np.linspace(0,2*np.pi,num_blades+1)[0:num_blades]  
    c              = rotor.chord_distribution 
    r              = rotor.radius_distribution  
    theta          = rotor.twist_distribution  # blade pitch 
    theta_0        = rotor.inputs.pitch_command # collective
    MCA            = rotor.mid_chord_alignment
    theta_tot      = theta + theta_0

    # dimension of matrices [control point, microphone , rotor, number of blades, number of sections , x,y,z coords]  

    # -----------------------------------------------------------------------------------------------------------------------------
    # translation matrix of rotor blade
    # ----------------------------------------------------------------------------------------------------------------------------- 
    I                               = np.atleast_3d(np.eye(4)).T 
    Tranlation_blade_trailing_edge  = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1)) 
    Tranlation_blade_trailing_edge[:,:,:,:,:,0,3] = MCA + 0.5*c  
    Tranlation_blade_trailing_edge[:,:,:,:,:,1,3] = r
    
    rev_Tranlation_blade_trailing_edge  = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1)) 
    rev_Tranlation_blade_trailing_edge[:,:,:,:,:,0,3] = -Tranlation_blade_trailing_edge[:,:,:,:,:,0,3] 
    rev_Tranlation_blade_trailing_edge[:,:,:,:,:,1,3] = -Tranlation_blade_trailing_edge[:,:,:,:,:,1,3] 
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # translation to blade quarter chord location 
    # -----------------------------------------------------------------------------------------------------------------------------
    I                               = np.atleast_3d(np.eye(4)).T 
    Tranlation_blade_quarter_chord  = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1)) 
    Tranlation_blade_quarter_chord[:,:,:,:,:,0,3] = MCA - 0.25*c  
    Tranlation_blade_quarter_chord[:,:,:,:,:,1,3] = r
    
    rev_Tranlation_blade_quarter_chord  = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1)) 
    rev_Tranlation_blade_quarter_chord[:,:,:,:,:,0,3] = -Tranlation_blade_quarter_chord[:,:,:,:,:,0,3]  
    rev_Tranlation_blade_quarter_chord[:,:,:,:,:,1,3] = -Tranlation_blade_quarter_chord[:,:,:,:,:,1,3]         
     

    # -----------------------------------------------------------------------------------------------------------------------------
    # rotation matrix of rotor blade twist
    # -----------------------------------------------------------------------------------------------------------------------------    

    Rotation_blade_twist  = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1))     
    Rotation_blade_twist[:,:,:,:,:,0,0] = np.cos(theta_tot)
    Rotation_blade_twist[:,:,:,:,:,0,2] = np.sin(theta_tot)
    Rotation_blade_twist[:,:,:,:,:,2,0] = -np.sin(theta_tot)
    Rotation_blade_twist[:,:,:,:,:,2,2] = np.cos(theta_tot)
    
    rev_Rotation_blade_twist  = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1))  
    rev_Rotation_blade_twist[:,:,:,:,:,0,2] = -Rotation_blade_twist[:,:,:,:,:,0,2]
    rev_Rotation_blade_twist[:,:,:,:,:,2,0] = -Rotation_blade_twist[:,:,:,:,:,2,0] 
     

    # -----------------------------------------------------------------------------------------------------------------------------
    # rotation matrix of rotor blade about azimuth
    # -----------------------------------------------------------------------------------------------------------------------------    
    # mine 
    Rotation_blade_azi  = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1))     
    Rotation_blade_azi[:,:,:,:,:,0,0] = np.tile(np.cos(phi)[:,None],(1,num_sec))  #np.tile(np.sin(phi)[:,None],(1,num_sec))   # May be wrong 
    Rotation_blade_azi[:,:,:,:,:,0,1] = -np.tile(np.sin(phi)[:,None],(1,num_sec)) #-np.tile(np.cos(phi)[:,None],(1,num_sec))  # May be wrong 
    Rotation_blade_azi[:,:,:,:,:,1,0] = np.tile(np.sin(phi)[:,None],(1,num_sec)) # np.tile(np.cos(phi)[:,None],(1,num_sec))    # May be wrong 
    Rotation_blade_azi[:,:,:,:,:,1,1] = np.tile(np.cos(phi)[:,None],(1,num_sec))   # np.tile(np.sin(phi)[:,None],(1,num_sec))   # May be wrong  
 
    rev_Rotation_blade_azi  = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1)) 
    rev_Rotation_blade_azi[:,:,:,:,:,0,1] = -Rotation_blade_azi[:,:,:,:,:,0,1] 
    rev_Rotation_blade_azi[:,:,:,:,:,1,0] = -Rotation_blade_azi[:,:,:,:,:,1,0]
    
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # rotation matrix of rotor about y axis by thrust angle (one extra dimension for translations)
    # -----------------------------------------------------------------------------------------------------------------------------
    Rotation_thrust_vector_angle                    = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1))
    thrust_vector                                   = np.arccos(prop2body[0][0][0]) - np.pi/2 
    Rotation_thrust_vector_angle[:,:,:,:,:,0,0] = np.cos(thrust_vector)
    Rotation_thrust_vector_angle[:,:,:,:,:,0,2] = np.sin(thrust_vector)
    Rotation_thrust_vector_angle[:,:,:,:,:,2,0] = -np.sin(thrust_vector) 
    Rotation_thrust_vector_angle[:,:,:,:,:,2,2] = np.cos(thrust_vector) 

    rev_Rotation_thrust_vector_angle                    =  np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1)) 
    rev_Rotation_thrust_vector_angle[:,:,:,:,:,0,2] = -Rotation_thrust_vector_angle[:,:,:,:,:,0,2]
    rev_Rotation_thrust_vector_angle[:,:,:,:,:,2,0] = -Rotation_thrust_vector_angle[:,:,:,:,:,2,0]
     

    # -----------------------------------------------------------------------------------------------------------------------------    
    # translation matrix of rotor to the relative location on the vehicle
    # -----------------------------------------------------------------------------------------------------------------------------
    Translation_origin_to_rel_loc                 = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1)) 
    Translation_origin_to_rel_loc[:,:,:,:,:,0,3]  = np.tile(rot_origins[:,0][None,None,None,None,:],(num_cpt,num_mic,1,num_blades,num_sec))
    Translation_origin_to_rel_loc[:,:,:,:,:,1,3]  = np.tile(rot_origins[:,1][None,None,None,None,:],(num_cpt,num_mic,1,num_blades,num_sec))     
    Translation_origin_to_rel_loc[:,:,:,:,:,2,3]  = np.tile(rot_origins[:,2][None,None,None,None,:],(num_cpt,num_mic,1,num_blades,num_sec)) 

    rev_Translation_origin_to_rel_loc                 = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1))
    rev_Translation_origin_to_rel_loc[:,:,:,:,:,0,3]  = -Translation_origin_to_rel_loc[:,:,:,:,:,0,3]
    rev_Translation_origin_to_rel_loc[:,:,:,:,:,1,3]  = -Translation_origin_to_rel_loc[:,:,:,:,:,1,3]   
    rev_Translation_origin_to_rel_loc[:,:,:,:,:,2,3]  = -Translation_origin_to_rel_loc[:,:,:,:,:,2,3]    
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # rotation of vehicle about y axis by AoA 
    # -----------------------------------------------------------------------------------------------------------------------------
    Rotation_AoA                        = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1))
    Rotation_AoA[:,:,:,:,:,0:3,0:3]     = conditions.frames.body.transform_to_inertial[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:] # WRONG 

    rev_Rotation_AoA                    = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1))  
    rev_Rotation_AoA[:,:,:,:,:,0,0]     = Rotation_AoA[:,:,:,:,:,0,0]   
    rev_Rotation_AoA[:,:,:,:,:,0,2]     = -Rotation_AoA[:,:,:,:,:,0,2]      
    rev_Rotation_AoA[:,:,:,:,:,2,0]     = -Rotation_AoA[:,:,:,:,:,2,0]   
    rev_Rotation_AoA[:,:,:,:,:,2,2]     = Rotation_AoA[:,:,:,:,:,2,2]    

    # -----------------------------------------------------------------------------------------------------------------------------
    # translation of vehicle to air  
    # -----------------------------------------------------------------------------------------------------------------------------
    Translation_mic_loc                  = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1)) 
    Translation_mic_loc[:,:,:,:,:,0,3]   = np.tile(mls[:,:,0][:,:,None,None,None],(1,1,num_rot,num_blades,num_sec)) 
    Translation_mic_loc[:,:,:,:,:,1,3]   = np.tile(mls[:,:,1][:,:,None,None,None],(1,1,num_rot,num_blades,num_sec)) 
    Translation_mic_loc[:,:,:,:,:,2,3]   = np.tile(mls[:,:,2][:,:,None,None,None],(1,1,num_rot,num_blades,num_sec))   

    rev_Translation_mic_loc                  = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1))
    rev_Translation_mic_loc[:,:,:,:,:,0,3]   = -Translation_mic_loc[:,:,:,:,:,0,3]
    rev_Translation_mic_loc[:,:,:,:,:,1,3]   = -Translation_mic_loc[:,:,:,:,:,1,3]
    rev_Translation_mic_loc[:,:,:,:,:,2,3]   = -Translation_mic_loc[:,:,:,:,:,2,3]    

    # -----------------------------------------------------------------------------------------------------------------------------
    # identity transformation 
    # -----------------------------------------------------------------------------------------------------------------------------
    I0    = np.atleast_3d(np.array([[0,0,0,1]]))
    I0    = np.array(I0)  
    mat_0 = np.tile(I0[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1))

    # -----------------------------------------------------------------------------------------------------------------------------
    # execute operations  
    # -----------------------------------------------------------------------------------------------------------------------------
    # for trailing edge  
    #mat_0_te = np.matmul(rev_Translation_mic_loc,I0) 
    #mat_1_te = np.matmul(rev_Rotation_AoA,mat_0_te) 
    #mat_2_te = np.matmul(rev_Translation_origin_to_rel_loc,mat_1_te)
    #mat_3_te = np.matmul(rev_Rotation_thrust_vector_angle,mat_2_te) 
    #mat_4_te = np.matmul(rev_Rotation_blade_azi,mat_3_te) 
    #mat_5_te = np.matmul(rev_Rotation_blade_twist,mat_4_te)   
    #mat_6_te = np.matmul(rev_Tranlation_blade_trailing_edge,mat_5_te)  
        
    # last to first 

    mat_0_te = np.matmul(rev_Translation_origin_to_rel_loc,I0)       
    mat_1_te = np.matmul(rev_Rotation_thrust_vector_angle,mat_0_te)     
    mat_2_te = np.matmul(rev_Rotation_blade_azi,mat_1_te )  
    mat_3_te = np.matmul(rev_Rotation_blade_twist,mat_2_te)     
    mat_4_te = np.matmul(rev_Tranlation_blade_trailing_edge,mat_3_te)    
   
    # first to last 
    mat_0_te_2 = np.matmul(rev_Tranlation_blade_trailing_edge,I0)   
    mat_1_te_2 = np.matmul(rev_Rotation_blade_twist,mat_0_te_2)   
    mat_2_te_2 = np.matmul(rev_Rotation_blade_azi,mat_1_te_2) 
    mat_3_te_2 = np.matmul(rev_Rotation_thrust_vector_angle,mat_2_te_2)
    mat_4_te_2 = np.matmul(rev_Translation_origin_to_rel_loc,mat_3_te_2)      
    
    # for points on quarter chord of untwisted blade 
    mat_5_qc = np.matmul(rev_Tranlation_blade_quarter_chord,mat_4_te)
    
    # for position vector from rotor point to obverver
    mat_1_p  = np.matmul(Rotation_thrust_vector_angle,mat_0)
    mat_2_p  = np.matmul(Translation_origin_to_rel_loc,mat_1_p)
    mat_3_p  = np.matmul(Rotation_AoA,mat_2_p) 
    mat_4_p  = np.matmul(Translation_mic_loc,mat_3_p)
    
    # for position vector from rotor blade section to obverver (alligned with wind frame)
    mat_1_bs = np.matmul(Tranlation_blade_quarter_chord,mat_0)
    mat_2_bs = np.matmul(Rotation_blade_twist,mat_1_bs)
    mat_3_bs = np.matmul(Rotation_blade_azi,mat_2_bs)
    mat_4_bs = np.matmul(Rotation_thrust_vector_angle,mat_3_bs)
    mat_5_bs = np.matmul(Translation_origin_to_rel_loc,mat_4_bs)
    mat_6_bs = np.matmul(Rotation_AoA,mat_5_bs)
    mat_7_bs = np.matmul(Translation_mic_loc,mat_6_bs)  
     
    return mat_6_te, mat_5_qc , mat_4_p,  mat_7_bs

## @ingroup Methods-Noise-Fidelity_One-Noise_Tools  
def compute_rotor_blade_section_source_coordinates(AoA,acoustic_outputs,rotors,mls,settings):  
    """This calculated the position vector from a point source to the observer 
            
    Assumptions:
        N/A
 
    Source:
        N/A  
 
    Inputs:  
        AoA                            - aircraft angle of attack                    [rad] 
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
    rot_origins = np.array(rot_origins) 
            
    rotor          = rotors[list(rotors.keys())[0]] 
    num_blades     = rotor.number_of_blades
    num_cf         = len(settings.center_frequencies)
    r              = rotor.radius_distribution
    num_sec        = len(r)
    phi            = np.ones(num_sec)*((2*np.pi)/num_blades)
    phi_2d0        = acoustic_outputs.disc_azimuthal_distribution
    beta_p         = 90*np.ones(num_sec)*Units.degrees  # no flapping 
    theta_tot      = rotor.twist_distribution + rotor.inputs.pitch_command # collective and pitch 
    alpha_eff0     = acoustic_outputs.blade_effective_angle_of_attack 
    orientation    = np.array(rotor.orientation_euler_angles) * 1 
    orientation[1] = orientation[1] + np.pi/2 # rotor tilt angle between the rotor hub plane and the vehicle hub plane
    body2thrust    = sp.spatial.transform.Rotation.from_rotvec(orientation).as_matrix()

    # Update dimensions for computation   
    r                    = np.tile(r[None,None,None,:,None],(num_cpt,num_mic,num_rot,1,num_cf))
    sin_phi              = np.tile(np.sin(phi)[None,None,None,:,None],(num_cpt,num_mic,num_rot,1,num_cf))  
    cos_phi              = np.tile(np.cos(phi)[None,None,None,:,None],(num_cpt,num_mic,num_rot,1,num_cf)) 
    sin_beta_p           = np.tile(np.sin(beta_p)[None,None,None,:,None],(num_cpt,num_mic,num_rot,1,num_cf))  
    cos_beta_p           = np.tile(np.cos(beta_p)[None,None,None,:,None],(num_cpt,num_mic,num_rot,1,num_cf))  
    sin_theta_tot        = np.tile(np.sin(theta_tot)[None,None,None,:,None],(num_cpt,num_mic,num_rot,1,num_cf)) 
    cos_theta_tot        = np.tile(np.cos(theta_tot)[None,None,None,:,None],(num_cpt,num_mic,num_rot,1,num_cf))
    sin_alpha_eff        = np.tile(np.sin(alpha_eff0)[:,None,None,:,None],(1,num_mic,num_rot,1,num_cf))
    cos_alpha_eff        = np.tile(np.cos(alpha_eff0)[:,None,None,:,None],(1,num_mic,num_rot,1,num_cf))
    cos_t_v              = np.tile(np.cos(-AoA)[:,None,None,None,:],(1,num_mic,num_rot,num_sec,num_cf))
    sin_t_v              = np.tile(np.sin(-AoA)[:,None,None,None,:],(1,num_mic,num_rot,num_sec,num_cf))   
    cos_t_v_t_r          = np.tile(np.array([body2thrust[0,0]])[:,None,None,None,None],(num_cpt,num_mic,num_rot,num_sec,num_cf))  
    sin_t_v_t_r          = np.tile(np.array([body2thrust[0,2]])[:,None,None,None,None],(num_cpt,num_mic,num_rot,num_sec,num_cf))  

    # ------------------------------------------------------------
    # ****** COORDINATE TRANSFOMRATIONS ******  
    M_t      = np.zeros((num_cpt,num_mic,num_rot,num_sec,num_cf,3,3))
    M_phi    = np.zeros((num_cpt,num_mic,num_rot,num_sec,num_cf,3,3))
    M_theta  = np.zeros((num_cpt,num_mic,num_rot,num_sec,num_cf,3,3))
    M_tv     = np.zeros((num_cpt,num_mic,num_rot,num_sec,num_cf,3,3)) 
    M_beta_p = np.zeros((num_cpt,num_mic,num_rot,num_sec,num_cf,3,1)) 

    M_tv[:,:,:,:,:,0,0]    = cos_t_v 
    M_tv[:,:,:,:,:,0,2]    = sin_t_v 
    M_tv[:,:,:,:,:,1,1]    = 1
    M_tv[:,:,:,:,:,2,0]    =-sin_t_v 
    M_tv[:,:,:,:,:,2,2]    = cos_t_v 
    

    M_beta_p[:,:,:,:,:,0,0]  = -r* sin_beta_p * cos_phi
    M_beta_p[:,:,:,:,:,1,0]  = -r* sin_beta_p * sin_phi
    M_beta_p[:,:,:,:,:,2,0]  =  r* cos_beta_p 
     

    # twist angle matrix
    M_theta[:,:,:,:,:,0,0] = cos_theta_tot 
    M_theta[:,:,:,:,:,0,2] = sin_theta_tot 
    M_theta[:,:,:,:,:,1,1] = 1
    M_theta[:,:,:,:,:,2,0] = -sin_theta_tot 
    M_theta[:,:,:,:,:,2,2] = cos_theta_tot 

    # azimuth motion matrix
    M_phi[:,:,:,:,:,0,0] = sin_phi 
    M_phi[:,:,:,:,:,0,1] = -cos_phi 
    M_phi[:,:,:,:,:,1,0] = cos_phi 
    M_phi[:,:,:,:,:,1,1] = sin_phi 
    M_phi[:,:,:,:,:,2,2] = 1

    # tilt motion matrix 
    M_t[:,:,:,:,:,0,0] =  cos_t_v_t_r 
    M_t[:,:,:,:,:,0,2] =  sin_t_v_t_r 
    M_t[:,:,:,:,:,1,1] =  1
    M_t[:,:,:,:,:,2,0] = -sin_t_v_t_r 
    M_t[:,:,:,:,:,2,2] =  cos_t_v_t_r 
    
    # transformation of geographical global reference frame to the sectional local coordinate
    M_hub   = np.tile(rot_origins[None,None,:,None,None,:,None],(num_cpt,num_mic,1,num_sec,num_cf,1,1))
    POS_2   = np.tile(mls[:,:,None,None,None,:,None],(num_cpt,1,num_rot,num_sec,num_cf,1,1))
    POS_1   = np.matmul(M_tv,(POS_2 + M_hub))  # eqn 4 and 5 
    mat0    = np.matmul(M_t,(POS_1 + M_beta_p))   
    mat1    = np.matmul(M_phi,mat0)
    POS     = np.matmul(M_theta,mat1)

    blade_section_position_vectors = Data() 
    blade_section_position_vectors.blade_section_coordinate_sys         = POS
    blade_section_position_vectors.vehicle_coordinate_sys               = POS_2 
    blade_section_position_vectors.sin_alpha_eff                        = sin_alpha_eff      
    blade_section_position_vectors.cos_alpha_eff                        = cos_alpha_eff

    return blade_section_position_vectors