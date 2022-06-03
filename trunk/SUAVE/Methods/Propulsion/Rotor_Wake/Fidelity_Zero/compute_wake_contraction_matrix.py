## @defgroup Methods-Propulsion-Rotor_Wake-Fidelity_Zero
# compute_wake_contraction_matrix.py
# 
# Created:  Sep 2020, M. Clarke 
#           Jul 2021, E. Botero
#           Sep 2021, R. Erhard
#           May 2021, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import jax.numpy as jnp

## @defgroup Methods-Propulsion-Rotor_Wake-Fidelity_Zero
def compute_wake_contraction_matrix(prop,Nr,m,nts,X_pts,prop_outputs):
    """ This computes slipstream development factor for all points 
    along slipstream

    Assumptions: 
    Fixed wake with helical shape  

    Source:  
    Stone, R. Hugh. "Aerodynamic modeling of the wing-propeller 
    interaction for a tail-sitter unmanned air vehicle." Journal 
    of Aircraft 45.1 (2008): 198-210.
    
    Inputs: 
    prop     - propeller/rotor data structure       
    Nr       - discretization on propeller/rotor [Unitless] 
    m        - control points in segemnt         [Unitless] 
    nts      - number of timesteps               [Unitless] 
    X_pts    - location of wake points           [meters] 

    Properties Used:
    N/A
    """    
    r                 = prop.radius_distribution  
    rdim              = Nr-1
    B                 = prop.number_of_blades
    va                = jnp.mean(prop_outputs.disc_axial_induced_velocity, axis=2)  # induced velocitied averaged around the azimuth
    R0                = prop.hub_radius 
    R_p               = prop.tip_radius  
    s                 = X_pts[0,:,0,-1,:] - prop.origin[0][0]    #  ( control point, blade number,  location on blade, time step)  
    s2                = 1 + s/(jnp.sqrt(s**2 + R_p**2))
    Kd                = jnp.repeat(jnp.atleast_2d(s2)[:, None, :], rdim , axis = 1)  
    
    # TO DO: UPDATE FOR ANGLES SO THAT VELOCITY IS COMPONENT IN ROTOR AXIS FRAME
    VX                = jnp.repeat(jnp.repeat(jnp.atleast_2d(prop_outputs.velocity[:,0]).T, rdim, axis = 1)[:, :, None], nts , axis = 2) # dimension (num control points, propeller distribution, wake points )
   
    prop_dif          = jnp.atleast_2d(va[:,1:] +  va[:,:-1])
    prop_dif          = jnp.repeat(prop_dif[:,  :, None], nts, axis=2) 
     
    Kv                = (2*VX + prop_dif) /(2*VX + Kd*prop_dif)  
    
    r_diff            = jnp.ones((m,rdim))*(r[1:]**2 - r[:-1]**2 )
    r_diff            = jnp.repeat(jnp.atleast_2d(r_diff)[:, :, None], nts, axis = 2) 
    r_prime           = jnp.zeros((m,Nr,nts))                
    r_prime           = r_prime.at[:,0,:].set(R0)
    for j in range(rdim):
        r_prime = r_prime.at[:,1+j,:].set(jnp.sqrt(r_prime[:,j,:]**2 + (r_diff*Kv)[:,j,:]))                       
    
    wake_contraction  = jnp.repeat((r_prime/jnp.repeat(jnp.atleast_2d(r)[:, :, None], nts, axis = 2))[:,None,:,:], B, axis = 1)            
    
    return wake_contraction 
            
