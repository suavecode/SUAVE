## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# compute_source_coordinates.py
# 
# Created:  Mar 2021, M. Clarke
# Modified: Feb 2022, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------
import numpy as np 
from MARC.Core import Data  
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

    rotor          = rotors[list(rotors.keys())[0]] 
    num_blades     = rotor.number_of_blades  
    num_sec        = len(rotor.radius_distribution)    
    
    # Get the rotation matrix
    prop2body   = rotor.prop_vel_to_body()  

    phi            = np.linspace(0,2*np.pi,num_blades+1)[0:num_blades]  
    c              = rotor.chord_distribution 
    r              = rotor.radius_distribution  
    beta           = rotor.flap_angle
    theta          = rotor.twist_distribution   # blade pitch 
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
    rev_Tranlation_blade_trailing_edge            = np.linalg.inv(Tranlation_blade_trailing_edge)       
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # translation to blade quarter chord location 
    # -----------------------------------------------------------------------------------------------------------------------------
    I                               = np.atleast_3d(np.eye(4)).T 
    Tranlation_blade_quarter_chord  = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1)) 
    Tranlation_blade_quarter_chord[:,:,:,:,:,0,3] = MCA - 0.25*c  
    Tranlation_blade_quarter_chord[:,:,:,:,:,1,3] = r 
    rev_Tranlation_blade_quarter_chord            = np.linalg.inv(Tranlation_blade_quarter_chord)         
     

    # -----------------------------------------------------------------------------------------------------------------------------
    # rotation matrix of rotor blade twist
    # -----------------------------------------------------------------------------------------------------------------------------    

    Rotation_blade_twist  = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1))     
    Rotation_blade_twist[:,:,:,:,:,0,0] = np.cos(theta_tot)
    Rotation_blade_twist[:,:,:,:,:,0,2] = np.sin(theta_tot)
    Rotation_blade_twist[:,:,:,:,:,2,0] = -np.sin(theta_tot)
    Rotation_blade_twist[:,:,:,:,:,2,2] = np.cos(theta_tot) 
    rev_Rotation_blade_twist            =  np.linalg.inv(Rotation_blade_twist) 

    # -----------------------------------------------------------------------------------------------------------------------------
    # rotation matrix of rotor blade flap
    # -----------------------------------------------------------------------------------------------------------------------------    
    # mine 
    Rotation_blade_flap  = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1))     
    Rotation_blade_flap[:,:,:,:,:,1,1] = np.cos(beta)
    Rotation_blade_flap[:,:,:,:,:,1,2] = -np.sin(beta)
    Rotation_blade_flap[:,:,:,:,:,2,1] = np.sin(beta)
    Rotation_blade_flap[:,:,:,:,:,2,2] = np.cos(beta) 
    rev_Rotation_blade_flap            = np.linalg.inv(Rotation_blade_flap)    
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # rotation matrix of rotor blade about azimuth
    # -----------------------------------------------------------------------------------------------------------------------------    
    # mine 
    Rotation_blade_azi  = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1))     
    Rotation_blade_azi[:,:,:,:,:,0,0] = np.tile(np.cos(phi)[:,None],(1,num_sec))  
    Rotation_blade_azi[:,:,:,:,:,0,1] = -np.tile(np.sin(phi)[:,None],(1,num_sec)) 
    Rotation_blade_azi[:,:,:,:,:,1,0] = np.tile(np.sin(phi)[:,None],(1,num_sec))  
    Rotation_blade_azi[:,:,:,:,:,1,1] = np.tile(np.cos(phi)[:,None],(1,num_sec))  
    rev_Rotation_blade_azi            = np.linalg.inv(Rotation_blade_azi)   
    
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # rotation matrix of rotor about y axis by thrust angle (one extra dimension for translations)
    # -----------------------------------------------------------------------------------------------------------------------------
    Rotation_thrust_vector_angle                    = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1))
    prop2body                                       = rotor.prop_vel_to_body()      
    thrust_vector                                   = np.arccos(prop2body[0][0][0]) - np.pi/2 
    Rotation_thrust_vector_angle[:,:,:,:,:,0,0]     = np.cos(thrust_vector)
    Rotation_thrust_vector_angle[:,:,:,:,:,0,2]     = np.sin(thrust_vector)
    Rotation_thrust_vector_angle[:,:,:,:,:,2,0]     = -np.sin(thrust_vector) 
    Rotation_thrust_vector_angle[:,:,:,:,:,2,2]     = np.cos(thrust_vector)  
    rev_Rotation_thrust_vector_angle                = np.linalg.inv(Rotation_thrust_vector_angle)   

    # -----------------------------------------------------------------------------------------------------------------------------    
    # translation matrix of rotor to the relative location on the vehicle
    # -----------------------------------------------------------------------------------------------------------------------------
    Translation_origin_to_rel_loc                 = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1)) 
    Translation_origin_to_rel_loc[:,:,:,:,:,0,3]  = np.tile(rot_origins[:,0][None,None,None,None,:],(num_cpt,num_mic,1,num_blades,num_sec))
    Translation_origin_to_rel_loc[:,:,:,:,:,1,3]  = np.tile(rot_origins[:,1][None,None,None,None,:],(num_cpt,num_mic,1,num_blades,num_sec))     
    Translation_origin_to_rel_loc[:,:,:,:,:,2,3]  = np.tile(rot_origins[:,2][None,None,None,None,:],(num_cpt,num_mic,1,num_blades,num_sec))  
    rev_Translation_origin_to_rel_loc             = np.linalg.inv(Translation_origin_to_rel_loc) 
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # rotation of vehicle about y axis by AoA 
    # -----------------------------------------------------------------------------------------------------------------------------
    Rotation_AoA                        = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1))
    Rotation_AoA[:,:,:,:,:,0:3,0:3]     = conditions.frames.body.transform_to_inertial[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:]  
    rev_Rotation_AoA                    = np.linalg.inv(Rotation_AoA) 

    # -----------------------------------------------------------------------------------------------------------------------------
    # translation of vehicle to air  
    # -----------------------------------------------------------------------------------------------------------------------------
    Translation_mic_loc                  = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1)) 
    Translation_mic_loc[:,:,:,:,:,0,3]   = np.tile(mls[:,:,0][:,:,None,None,None],(1,1,num_rot,num_blades,num_sec)) 
    Translation_mic_loc[:,:,:,:,:,1,3]   = np.tile(mls[:,:,1][:,:,None,None,None],(1,1,num_rot,num_blades,num_sec)) 
    Translation_mic_loc[:,:,:,:,:,2,3]   = np.tile(mls[:,:,2][:,:,None,None,None],(1,1,num_rot,num_blades,num_sec))   
    rev_Translation_mic_loc              = np.linalg.inv(Translation_mic_loc)   

    # -----------------------------------------------------------------------------------------------------------------------------
    # rotation of vehicle by velocity vector i.ee rotation by bank (roll)  air speed angle of attack to x axis (pitch) and true course angle (yaw)
    # -----------------------------------------------------------------------------------------------------------------------------

    Rotation_RPY                        = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1)) 
    V_vec_pitch                         = np.linalg.inv(conditions.frames.wind.transform_to_inertial[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:])
    V_vec_true_course                   = np.linalg.inv(conditions.frames.planet.true_course_angle[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:]) 
    Rotation_RPY[:,:,:,:,:,0:3,0:3]     = np.matmul(V_vec_true_course,V_vec_pitch) 
    rev_Rotation_RPY                    = np.linalg.inv(Rotation_RPY)  
    

    # -----------------------------------------------------------------------------------------------------------------------------
    # vehicle velocit vector 
    # -----------------------------------------------------------------------------------------------------------------------------    
    M_vec                        = np.tile(I[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1))     
    M_vec[:,:,:,:,:,0,3]         = np.tile((conditions.frames.inertial.velocity_vector[:,0]/conditions.freestream.speed_of_sound[:,0])[:,None,None,None,None],(1,num_mic,num_rot,num_blades,num_sec)) 
    M_vec[:,:,:,:,:,1,3]         = np.tile((conditions.frames.inertial.velocity_vector[:,1]/conditions.freestream.speed_of_sound[:,0])[:,None,None,None,None],(1,num_mic,num_rot,num_blades,num_sec))   
    M_vec[:,:,:,:,:,2,3]         = np.tile((conditions.frames.inertial.velocity_vector[:,2]/conditions.freestream.speed_of_sound[:,0])[:,None,None,None,None],(1,num_mic,num_rot,num_blades,num_sec))     
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # identity transformation 
    # -----------------------------------------------------------------------------------------------------------------------------
    I0    = np.atleast_3d(np.array([[0,0,0,1]]))
    I0    = np.array(I0)  
    mat_0 = np.tile(I0[None,None,None,None,:,:,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1,1))

    # -----------------------------------------------------------------------------------------------------------------------------
    # execute operations to compute matrices for aircraft location in present frame 
    # -----------------------------------------------------------------------------------------------------------------------------
    # for trailing edge  
    
    mat_0_te   = np.matmul(rev_Rotation_RPY,I0)     # we need to do an initial rotation to allign vector in the direction of the vehicle motion
    mat_1_te   = np.matmul(rev_Translation_mic_loc,mat_0_te) 
    mat_2_te   = np.matmul(rev_Rotation_AoA,mat_1_te) 
    mat_3_te   = np.matmul(rev_Translation_origin_to_rel_loc,mat_2_te) 
    mat_4_te   = np.matmul(rev_Rotation_thrust_vector_angle,mat_3_te) 
    mat_5_te   = np.matmul(rev_Rotation_blade_azi,mat_4_te) 
    mat_6_te   = np.matmul(rev_Rotation_blade_flap,mat_5_te)
    mat_7_te   = np.matmul(rev_Rotation_blade_twist,mat_6_te)   
    mat_8_te   = np.matmul(rev_Tranlation_blade_trailing_edge,mat_7_te)  
          
    # for points on quarter chord of untwisted blade 
    mat_7_qc = np.matmul(rev_Tranlation_blade_quarter_chord,mat_6_te)
    
    # for position vector from rotor point to obverver 
    mat_0_p  = np.matmul(Rotation_thrust_vector_angle,mat_0)
    mat_1_p  = np.matmul(Translation_origin_to_rel_loc,mat_0_p)
    mat_2_p  = np.matmul(Rotation_AoA,mat_1_p) 
    mat_3_p  = np.matmul(Translation_mic_loc,mat_2_p)
    mat_4_p  = np.matmul(Rotation_RPY,mat_3_p)
    
    # for position vector from rotor blade section to obverver (alligned with wind frame)
    mat_0_bs = np.matmul(Tranlation_blade_quarter_chord,mat_0)
    mat_1_bs = np.matmul(Rotation_blade_twist,mat_0_bs)
    mat_2_bs = np.matmul(Rotation_blade_flap,mat_1_bs)
    mat_3_bs = np.matmul(Rotation_blade_azi,mat_2_bs)
    mat_4_bs = np.matmul(Rotation_thrust_vector_angle,mat_3_bs)
    mat_5_bs = np.matmul(Translation_origin_to_rel_loc,mat_4_bs)
    mat_6_bs = np.matmul(Rotation_AoA,mat_5_bs)
    mat_7_bs = np.matmul(Translation_mic_loc,mat_6_bs)  
    mat_8_bs = np.matmul(Rotation_RPY,mat_7_bs) 
    
    X_prime  = -mat_8_bs[:,:,:,:,:,0:3,0]
    X_e      = mat_8_te[:,:,:,:,:,0:3,0]
    X        = mat_7_qc[:,:,:,:,:,0:3,0]
    X_hub    = -mat_4_p[:,:,:,:,:,0:3,0] 
      
    # -----------------------------------------------------------------------------------------------------------------------------
    # execute operations to compute matrices for aircraft location in retarded frame
    # ----------------------------------------------------------------------------------------------------------------------------- 
    S              = np.linalg.norm(X, axis =5)     
    S_prime        = np.linalg.norm(X_prime, axis =5)    
    S_e            = np.linalg.norm(X_e, axis =5)    
    S_hub          = np.linalg.norm(X_hub, axis =5)   
    
    theta          = np.arccos(X[:,:,:,:,:,0]/S)  
    theta_prime    = np.arccos(X_prime[:,:,:,:,:,0]/S_prime)  
    theta_e        = np.arccos(X_e[:,:,:,:,:,0]/S_e)  
    theta_hub      = np.arccos(X_hub[:,:,:,:,:,0]/S_hub)  
    
    M_vec_rotated  = np.matmul(Rotation_RPY,M_vec)
    M_x            = np.linalg.norm(M_vec_rotated[:,:,:,:,:,0:3,3], axis =5)      
    
    theta_r        =  np.arccos(np.cos(theta)*np.sqrt(1 - ((M_x**2)*np.sin(theta)**2 )) + M_x*np.sin(theta)**2)
    theta_prime_r  =  np.arccos(np.cos(theta_prime)*np.sqrt(1 - ((M_x**2)*np.sin(theta_prime)**2 )) + M_x*np.sin(theta_prime)**2)
    theta_e_r      =  np.arccos(np.cos(theta_e)*np.sqrt(1 - ((M_x**2)*np.sin(theta_e)**2 )) + M_x*np.sin(theta_e)**2)
    theta_hub_r    =  np.arccos(np.cos(theta_hub)*np.sqrt(1 - ((M_x**2)*np.sin(theta_hub)**2 )) + M_x*np.sin(theta_hub)**2)

    Y              = np.sqrt(X[:,:,:,:,:,1]**2 + X[:,:,:,:,:,2]**2)     
    Y_prime        = np.sqrt(X_prime[:,:,:,:,:,1]**2 + X_prime[:,:,:,:,:,2]**2)          
    Y_e            = np.sqrt(X_e[:,:,:,:,:,1]**2 + X_e[:,:,:,:,:,2]**2)            
    Y_hub          = np.sqrt(X_hub[:,:,:,:,:,1]**2 + X_hub[:,:,:,:,:,2]**2)    
    
    x_r            = Y/(np.tan(theta_r))                             
    x_prime_r      = Y_prime/(np.tan(theta_prime_r))                             
    x_e_r          = Y_e/(np.tan(theta_e_r))                             
    x_hub_r        = Y_hub/(np.tan(theta_hub_r))    

    X_prime_r      = np.zeros_like(X_prime)  
    X_e_r          = np.zeros_like(X_e)   
    X_r            = np.zeros_like(X)   
    X_hub_r        = np.zeros_like(X_hub)  

    # update x coordiates of matrics 
    X_prime_r[:,:,:,:,:,0]  = x_prime_r 
    X_prime_r[:,:,:,:,:,1]  = X_prime[:,:,:,:,:,1] 
    X_prime_r[:,:,:,:,:,2]  = X_prime[:,:,:,:,:,2] 
    X_e_r[:,:,:,:,:,0]      = x_e_r  
    X_e_r[:,:,:,:,:,1]      = X_e[:,:,:,:,:,1]
    X_e_r[:,:,:,:,:,2]      = X_e[:,:,:,:,:,2]
    X_r[:,:,:,:,:,0]        = x_r 
    X_r[:,:,:,:,:,1]        = X[:,:,:,:,:,1]  
    X_r[:,:,:,:,:,2]        = X[:,:,:,:,:,2]  
    X_hub_r[:,:,:,:,:,0]    = x_hub_r   
    X_hub_r[:,:,:,:,:,1]    = X_hub[:,:,:,:,:,1] 
    X_hub_r[:,:,:,:,:,2]    = X_hub[:,:,:,:,:,2] 
     
    coordinates        = Data(
        X_prime        = X_prime,
        X_e            = X_e,
        X              = X,
        X_hub          = X_hub,
        X_prime_r      = X_prime_r,
        X_e_r          = X_e_r,
        X_r            = X_r,
        X_hub_r        = X_hub_r, 
        theta_r        = theta_r,    
        theta_prime_r  = theta_prime_r,
        theta_e        = theta_e, 
        theta_e_r      = theta_e_r,
        theta_hub      = theta_hub,
        theta_hub_r    = theta_hub_r)
    
    return coordinates 