## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
# compute_fidelity_one_inflow_velocities.py
#
# Created:  Sep 2021, R. Erhard
# Modified: Jan 2022, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_wake_induced_velocity import compute_wake_induced_velocity

# package imports
import jax.numpy as jnp
from jax.lax import fori_loop as fori

## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
def compute_fidelity_one_inflow_velocities( wake, prop, WD ):
    """
    Computs the inflow velocities
    
    Assumptions:
        None

    Source:
        N/A
    Inputs:
        wake - rotor wake
        prop - rotor instance
        WD   - wake vortex distribution
    Outputs:
        Va   - axial velocity, shape (ctrl_pts, Nr, Na); axis 2 in direction of rotation       [m/s]
        Vt   - tangential velocity, shape (ctrl_pts, Nr, Na); axis 2 in direction of rotation    [m/s]
    """
    
    VD                       = prop.vortex_distribution
    omega                    = prop.inputs.omega
    init_timestep_offset     = wake.wake_settings.initial_timestep_offset

    # use results from prior bevw iteration
    prop_outputs  = prop.outputs
    cpts          = len(prop_outputs.velocity)
    Na            = prop.number_azimuthal_stations
    Nr            = len(prop.chord_distribution)
    r             = prop.radius_distribution
    rot           = prop.rotation
    WD            = wake.vortex_distribution

    # compute radial blade section locations based on initial timestep offset
    azi_step = 2*jnp.pi/(Na+1)
    dt       = azi_step/omega[0][0]
    t0       = dt*init_timestep_offset

    # set shape of velocity arrays
    Va = jnp.zeros((cpts,Nr,Na))
    Vt = jnp.zeros((cpts,Nr,Na))
    
    r_midpts = (r[1:] + r[:-1])/2
    r_midpts = r_midpts[jnp.newaxis,...]
    r_midpts = jnp.broadcast_to(r_midpts,(cpts,Nr-1))    
    
    r_eval   = r[jnp.newaxis,...]
    r_eval   = jnp.broadcast_to(r_eval,(cpts,Nr))
    
    # Clear the XC, YC, and ZC values
    VD.XC   = jnp.zeros((Nr-1))
    VD.YC   = jnp.zeros((Nr-1))
    VD.ZC   = jnp.zeros((Nr-1))
    VD.n_cp = Nr-1

    VD.Wake_collapsed = WD    
    
    prop.vortex_distribution = VD
    
    # Compute induced velocities at blade from the helical fixed wake
    inits   = (Va,Vt,VD)
    
    function = lambda i, inits: Na_loop(i,inits,rot,omega,t0,Na,prop,wake,cpts,WD,Nr,r_eval,r_midpts)
    
    outnits = fori(0,Na,function,inits)
    
    Va, Vt, VD = outnits    
    
    prop.vortex_distribution = VD

    return Va, Vt, prop

def Na_loop(i,inits,rot,omega,t0,Na,prop,wake,cpts,WD,Nr,r_eval,r_midpts):
    
    Va,Vt,VD = inits
    
    # increment blade angle to new azimuthal position 
    blade_angle   = -rot*(omega[0]*t0 + i*(2*jnp.pi/(Na)))  # axial view of rotor, negative rotation --> positive blade angle

    #----------------------------------------------------------------
    #Compute the wake-induced velocities at propeller blade
    #----------------------------------------------------------------
    #set the evaluation points in the vortex distribution: (ncpts, nblades, Nr, Ntsteps)
    Yb   = wake.vortex_distribution.reshaped_wake.Yblades_cp[i,0,0,:,0]
    Zb   = wake.vortex_distribution.reshaped_wake.Zblades_cp[i,0,0,:,0]
    Xb   = wake.vortex_distribution.reshaped_wake.Xblades_cp[i,0,0,:,0]
    
    VD.XC = (Xb[1:] + Xb[:-1])/2
    VD.YC = (Yb[1:] + Yb[:-1])/2
    VD.ZC = (Zb[1:] + Zb[:-1])/2

    VD.n_cp = jnp.size(VD.YC)
    
    V_ind   = compute_wake_induced_velocity(WD, VD, cpts, azi_start_idx=i)
    
    # velocities in vehicle frame
    u       = V_ind[:,:,0]    # velocity in vehicle x-frame
    v       = V_ind[:,:,1]    # velocity in vehicle y-frame
    w       = V_ind[:,:,2]    # velocity in vehicle z-frame
    
    ## rotate from vehicle to prop frame:
    rot_to_prop = prop.vec_to_prop_body()
    uprop       = u*rot_to_prop[:,0,0][:,None] + w*rot_to_prop[:,0,2][:,None]
    vprop       = v
    wprop       = u*rot_to_prop[:,2,0][:,None] + w*rot_to_prop[:,2,2][:,None]     
    
    # interpolate to get values at rotor radial stations
    up = jnp.zeros((cpts,Nr))
    vp = jnp.zeros((cpts,Nr))
    wp = jnp.zeros((cpts,Nr))
    
    up = interp1d_rotor(r_eval, r_midpts,uprop)
    vp = interp1d_rotor(r_eval, r_midpts,vprop)
    wp = interp1d_rotor(r_eval, r_midpts,wprop)


    # Update velocities at the disc
    Va = Va.at[:,:,i].set(up)
    Vt = Vt.at[:,:,i].set(-rot*(vp*(jnp.cos(blade_angle)) - wp*(jnp.sin(blade_angle)) ))  # velocity component in direction of rotation       

    
    return (Va, Vt, VD)


# Interpolator

def interp1d_rotor(eval_x,x_i,y_i):
    
    """
    Does a 1D interpolation on 2D data. The second index is the one that is interpolated on. This really only works on
    the rotor
    
    Assumptions:
        X IS ALREADY SORTED!!!

    Inputs:
        x_i   - input x vector to use for interpolation
        y_i   - input y vector to use for interpolation
    Outputs:
       eval_y - interpolated Y values of shape eval_x
    """
    
    # find the inidices less than x_eval, closest point
    i = jnp.clip(jnp.apply_along_axis(jnp.searchsorted,1,x_i,eval_x,side='right')-1,0,x_i.shape[1]-1)[0,0,:]
    
    # Evaluate the points
    eval_y = y_i[:,i] + (eval_x[:,:]-x_i[:,i])*(y_i[:,i+1]-y_i[:,i])/(x_i[:,i+1]-x_i[:,i])
    
    
    # if the values are outside the range extrapolate
    # above
    max_x   = x_i[:,-1]
    max_y   = y_i[:,-1]
    max_xm1 = x_i[:,-2]
    max_ym1 = y_i[:,-2]
    slope   = (max_y-max_ym1)/(max_x-max_xm1)
    
    eval_y = eval_y.at[:,-1].set(max_y+(eval_x[:,-1]-max_x)*slope)

    return eval_y    
    
    
    