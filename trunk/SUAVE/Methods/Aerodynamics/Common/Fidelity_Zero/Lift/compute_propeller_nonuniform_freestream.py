## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# compute_propeller_nonuniform_freestream.py
# 
# Created:   April 2021, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np
import scipy as sp
from SUAVE.Methods.Geometry.Three_Dimensional import  orientation_product, orientation_transpose


def compute_propeller_nonuniform_freestream(prop, upstream_wake,conditions):
    """ Computes the inflow velocities in the frame of the rotating propeller
    
    Inputs:
       prop.                               SUAVE propeller 
            tip_radius                     - propeller radius                         [m]
            rotation                       - propeller rotation direction             [-]
            thrust_angle                   - thrust angle of prop                     [rad]
            number_radial_stations         - number of propeller radial stations      [-]
            number_azimuthal_stations      - number of propeller azimuthal stations   [-]
       upstream_wake.
          u_velocities                     - Streamwise velocities from upstream wake
          v_velocities                     - Spanwise velocities from upstream wake
          w_velocities                     - Downwash velocities from upstream wake
          VD                               - Vortex distribution from upstream wake
       conditions.
          frames
       
    Outputs:
       Va                     Axial velocities at propeller             [m/s]
       Vt                     Tangential velocities at propeller        [m/s]
       Vr                     Radial velocities at propeller            [m/s]
    """
    # unpack propeller parameters
    Vv       = conditions.frames.inertial.velocity_vector 
    R        = prop.tip_radius    
    rotation = prop.rotation
    c        = prop.chord_distribution
    Na       = prop.number_azimuthal_stations 
    Nr       = len(c)
    
    ua_wing  = upstream_wake.u_velocities
    uv_wing  = upstream_wake.v_velocities
    uw_wing  = upstream_wake.w_velocities
    VD       = upstream_wake.VD
    
    # Velocity in the Body frame
    T_body2inertial = conditions.frames.body.transform_to_inertial
    T_inertial2body = orientation_transpose(T_body2inertial)
    V_body          = orientation_product(T_inertial2body,Vv)
    body2thrust     = prop.body_to_prop_vel()
    T_body2thrust   = orientation_transpose(np.ones_like(T_body2inertial[:])*body2thrust)  
    V_thrust        = orientation_product(T_body2thrust,V_body) 
    
    
    # azimuth distribution 
    psi       = np.linspace(0,2*np.pi,Na+1)[:-1]
    psi_2d    = np.tile(np.atleast_2d(psi),(Nr,1))   

    # 2 dimensiona radial distribution non dimensionalized
    chi     = prop.radius_distribution /R
    
    # Reframe the wing induced velocities:
    y_center = prop.origin[0][1] 
    
    # New points to interpolate data: (corresponding to r,phi locations on propeller disc)
    points  = np.array([[VD.YC[i], VD.ZC[i]] for i in range(len(VD.YC))])
    ycoords = np.reshape((R*chi*np.cos(psi_2d).T).T,(Nr*Na,))
    zcoords = prop.origin[0][2]  + np.reshape((R*chi*np.sin(psi_2d).T).T,(Nr*Na,))
    xi      = np.array([[y_center+ycoords[i],zcoords[i]] for i in range(len(ycoords))])
    
    ua_w = sp.interpolate.griddata(points,ua_wing,xi,method='linear')
    uv_w = sp.interpolate.griddata(points,uv_wing,xi,method='linear')
    uw_w = sp.interpolate.griddata(points,uw_wing,xi,method='linear') 
    
    ua_wing = np.reshape(ua_w,(Nr,Na))
    uw_wing = np.reshape(uw_w,(Nr,Na))
    uv_wing = np.reshape(uv_w,(Nr,Na))    
    
    if rotation == [1]:
        Vt_2d =  V_thrust[:,0]*( -np.array(uw_wing)*np.cos(psi_2d) + np.array(uv_wing)*np.sin(psi_2d)  )  # velocity tangential to the disk plane, positive toward the trailing edge eqn 6.34 pg 165           
        Vr_2d =  V_thrust[:,0]*( -np.array(uw_wing)*np.sin(psi_2d) - np.array(uv_wing)*np.cos(psi_2d)  )  # radial velocity , positive outward              
        Va_2d =  V_thrust[:,0]*   np.array(ua_wing)                                                       # velocity perpendicular to the disk plane, positive downward  eqn 6.36 pg 166  
    else:     
        Vt_2d =  V_thrust[:,0]*(  np.array(uw_wing)*np.cos(psi_2d) - np.array(uv_wing)*np.sin(psi_2d)  )  # velocity tangential to the disk plane, positive toward the trailing edge       
        Vr_2d =  V_thrust[:,0]*( -np.array(uw_wing)*np.sin(psi_2d) - np.array(uv_wing)*np.cos(psi_2d)  )  # radial velocity , positive outward               
        Va_2d =  V_thrust[:,0]*   np.array(ua_wing)                                                       # velocity perpendicular to the disk plane, positive downward
    
    # Append velocities to propeller
    prop.tangential_velocities_2d = Vt_2d
    prop.radial_velocities_2d     = Vr_2d
    prop.axial_velocities_2d      = Va_2d    
    
    return prop