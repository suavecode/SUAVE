## @ingroup Methods-Noise-Fidelity_One-Rotor
# compute_TBL_TE_broadband_noise.py
#
# Created: Feb 2023, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np   

## @ingroup Methods-Noise-Fidelity_One-Rotor 
def compute_TBL_TE_broadband_noise(f,r_e,L,U,M,R_c,Dbar_h,Dbar_l,R_delta_star_p,delta_star_p,delta_star_s,alpha_star):
    '''This computes the turbument boundary layer- trailing edge noise compoment of broadband noise using the method outlined by the 
    Brooks, Pope and Marcolini (BPM) Model
    
    Assumptions:
        BPM models assumes a naca 0012 airfol  
        
    Source:  
        BPM Model:  Brooks, Thomas F., D. Stuart Pope, and Michael A.
        Marcolini. Airfoil self-noise and prediction. No. L-16528. 1989.
    
    Inputs:   
       R_delta_star_p - Reynolds number correlated to boundary layer thicness [m]
       delta_star_p   - boundary layer thickness of pressure side             [m]
       delta_star_s   - boundary layer thickness of suction side              [m]
       R_c            - Reynolds number                                       [-]
       alpha_star     - angle of attack blade section                         [deg]
       delta_p        - boundary layer of pressure section                    [m
       L              - length of blade section                               [m]
       U              - velocity of blade section                             [m/s]
       M              - Mach number                                           [-]
       c              - airfoil section chord                                 [m] 
       f              - frequency spectrum                                    [Hz]
       Dbar_h         - high frequency directivity term                       [-]
       Dbar_l         - low frequency directivity term                        [-]
       r_e            - distance from noise source to observer                [m] 
    
    
    Outputs 
       SPL_TBL_TE   - Sound pressure level of turbument boundary layer        [dB]
       
    Properties Used:
        N/A   
    '''      
     
    # Strouhal number
    St_p                  = f*delta_star_p/U    # eqn 31
    St_s                  = f*delta_star_s/U    # eqn 31
    St_1                  = 0.02*(M**(-0.6))      # eqn 32
    St_2                  = St_1*(10**(0.0054*((alpha_star - 1.33)**2)))
    St_2[alpha_star<1.33] = St_1[alpha_star<1.33]*1
    St_2[alpha_star>12.5] = St_1[alpha_star>12.5]*4.72 
    St_bar_1              = (St_1 + St_2)/2     # eqn 33 
    
    # Spectral Shape Functions
    St_peak   = St_bar_1 # can be St_1,St_2 or St_bar_1 
    A_p       = spectral_shape_function_A(St_p,St_peak,R_c)    # eqns 35 - 40
    A_s       = spectral_shape_function_A(St_s,St_peak,R_c)    # eqns 35 - 40
    A_prime   = spectral_shape_function_A(St_s,St_peak,R_c*3)  # eqns 35 - 40
    B         = spectral_shape_function_B(St_s,St_2,R_c)       # eqns 31 - 46    
    
    # Amplitude function 
    K_1                                  = amplitude_function_K_1(R_c)                             # eqn 47
    delta_K_1                            = amplitude_function_delta_K_1(alpha_star,R_delta_star_p) # eqn 48 
    K_2                                  = amplitude_function_K_2(alpha_star,M,K_1)                # eqn 49   
    SPL_alpha                            = 10*np.log10((delta_star_s*(M**5)*L*Dbar_h)/(r_e**2))  + B   + K_2                    # eqn 27 
    SPL_s                                = 10*np.log10((delta_star_s*(M**5)*L*Dbar_h)/(r_e**2))  + A_s + (K_1 - 3)              # eqn 26
    SPL_p                                = 10*np.log10((delta_star_p*(M**5)*L*Dbar_h)/(r_e**2))  + A_p + (K_1 - 3) + delta_K_1  # eqn 25   
   
    alpha_star_0_bool                    = np.zeros_like(alpha_star , dtype=bool) # 
    K_2_peak                             = np.max(K_2)
    alpha_switch_1                       = np.where(K_2 == K_2_peak)[0]
    alpha_switch_2                       = np.where(alpha_star>12.5)[0]
    alpha_star_0_bool[alpha_switch_1]    = True 
    alpha_star_0_bool[alpha_switch_2]    = True  
    
    SPL_p[alpha_star_0_bool]             = -np.inf # eqn 28
    SPL_s[alpha_star_0_bool]             = -np.inf # eqn 29 
    SPL_alpha[alpha_star_0_bool]         = 10*np.log10((delta_star_s[alpha_star_0_bool]*(M[alpha_star_0_bool]**5)*L*Dbar_l[alpha_star_0_bool])/(r_e**2))  + A_prime[alpha_star_0_bool]+ K_2[alpha_star_0_bool]  
             
    SPL_TBL_TE                           = 10*np.log10( 10**(SPL_alpha/10) + 10**(SPL_s/10) + 10**(SPL_p/10) ) # eqn 24 

    return  SPL_TBL_TE

def spectral_shape_function_A(St,St_peak,R_c):  
    a                  = abs(np.log10(St/St_peak))    # 37  
    a_0                = (-9.57E-13)*((R_c - 8.57E5)**2) + 1.13    # 38 
    a_0[R_c < 9.52E4]  =  0.57                            # 38 
    a_0[R_c > 8.57E5]  =  1.13                            # 38  
    
    A_min     = A_min_function(a)                          # eqn 35 
    A_min_a_0 = A_min_function(a_0)                        # eqn 35 
    A_max     = A_max_function(a)                          # eqn 36 
    A_max_a_0 = A_max_function(a_0)                        # eqn 36 
 
    A_R_a_0   = (-20 - A_min_a_0)/(A_max_a_0 - A_min_a_0)  # eqn 39 
    A         = A_min + A_R_a_0*abs(A_max -A_min )         # eqn 40
    
    return A 

def spectral_shape_function_B(St_s,St_2,R_c): 
    b                   = abs(np.log10(St_s/St_2))                    # 43  
    b_0                 = (-4.48E-13)*((R_c - 8.57E5)**2) + 0.56      # 44
    b_0[R_c < 9.52E4]   = 0.30                         # 44
    b_0[R_c > 8.57E5]   = 0.56                         # 44 
    
    B_min     = B_min_function(b)    # eqn 41
    B_min_b_0 = B_min_function(b_0)  # eqn 41
    B_max     = B_max_function(b)    # eqn 42 
    B_max_b_0 = B_max_function(b_0)  # eqn 42 
 
    B_R_b_0   = (-20 - B_min_b_0)/(B_max_b_0 - B_min_b_0)  # eqn 45
    B         = B_min + B_R_b_0*abs(B_max -B_min )         # eqn 40  
    return B

def A_min_function(a):   
    A_min          = -32.665*a +3.981 # eqn 35 
    A_min[a<0.204] = np.sqrt(67.552-886.788*(a[a<0.204]**2)) - 8.219 # eqn 35 
    A_min[a>0.244] = -142.795*a[a>0.244]**3 +  103.656*a[a>0.244]**2 - 57.757*a[a>0.244] + 6.006 # eqn 35 
   
    return A_min 

def A_max_function(a):    
    A_max          = -15.901*a + 1.098 # eqn 36 
    A_max[a<0.13]  = np.sqrt(67.552-886.788*(a[a<0.13]**2)) - 8.219 # eqn 36 
    A_max[a>0.321] = -4.669*a[a>0.321]**3 + 3.491*a[a>0.321]**2  - 16.699*a[a>0.321] + 1.149 # eqn 36  
    return A_max 


def B_min_function(b):   
    B_min          = -83.607*b + 8.138  # eqn 41
    B_min[b<0.13]  = np.sqrt(16.888-886.788*(b[b<0.13]**2)) - 4.109
    B_min[b>0.145] = -817.810*b[b>0.145]**3 +  355.210*b[b>0.145]**2 - 135.024*b[b>0.145] + 10.619 # eqn 41    
    return B_min 

def B_max_function(b):  
    B_max          = -31*33*b + 1.854  # eqn 42
    B_max[b<0.10]  = np.sqrt(16.888-886.788*(b[b<0.10]**2)) - 4.109 # eqn 42
    B_max[b>0.187] = -80.541*b[b>0.187]**3 +  44.174*b[b>0.187]**2 - 39.381*b[b>0.187] + 2.344 # eqn 42  
    return B_max 


def amplitude_function_K_1(R_c): 
    K_1             = - 9.0*np.log10(R_c) + 181.6   # eqn 47 
    K_1[R_c<2.47E5] = -4.31*np.log10(R_c[R_c<2.47E5]) + 156.3  # eqn 47
    K_1[R_c>8.0E5]  = 128.5   # eqn 47 
    return K_1

def amplitude_function_delta_K_1(alpha_star,R_delta_star_p):  
    delta_K_1                        = alpha_star*(1.43*np.log10(R_delta_star_p) - 5.29)   # eqn 48 
    delta_K_1[R_delta_star_p>5000] = 0                                                       # eqn 48     
    return delta_K_1

def amplitude_function_K_2(alpha_star,M,K_1):    
    gamma   = 27.094*M + 3.31  # eqn 50  
    gamma_0 = 23.43*M + 4.651  # eqn 50  
    beta    = 72.65*M + 10.74  # eqn 50
    beta_0  = -34.19*M - 13.82 # eqn 50   
    K_2                               = K_1 + np.sqrt(beta**2 - ((beta/gamma)**2)*((alpha_star-gamma_0)**2)) + beta_0      # eqn 49 
    K_2[alpha_star< (gamma_0-gamma)]  = K_1[alpha_star< (gamma_0-gamma)] -1000  # eqn 49 
    K_2[alpha_star> (gamma_0+gamma)]  = K_1[alpha_star> (gamma_0+gamma)] -12 # eqn 49 
    return K_2

