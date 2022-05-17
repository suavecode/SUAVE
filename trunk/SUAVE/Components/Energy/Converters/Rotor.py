## @ingroup Components-Energy-Converters
# Rotor.py
#
# Created:  Jun 2014, E. Botero
# Modified: Jan 2016, T. MacDonald
#           Feb 2019, M. Vegh
#           Mar 2020, M. Clarke
#           Sep 2020, M. Clarke
#           Mar 2021, R. Erhard
#           Apr 2021, M. Clarke
#           Jul 2021, E. Botero
#           Jul 2021, R. Erhard
#           Sep 2021, R. Erhard
#           Feb 2022, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Analyses.Propulsion.Rotor_Wake_Fidelity_Zero import Rotor_Wake_Fidelity_Zero
from SUAVE.Analyses.Propulsion.Rotor_Wake_Fidelity_One import Rotor_Wake_Fidelity_One
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.BET_calculations \
     import compute_airfoil_aerodynamics,compute_inflow_and_tip_loss
from SUAVE.Methods.Geometry.Three_Dimensional \
     import  orientation_product_jax, orientation_transpose

# package imports
import numpy as np
import scipy as sp
from jax.tree_util import register_pytree_node_class
from jax import jit, lax
import jax.numpy as jnp
from functools import partial

# ----------------------------------------------------------------------
#  Generalized Rotor Class
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
@register_pytree_node_class
class Rotor(Energy_Component):
    """This is a general rotor component.

    Assumptions:
    None

    Source:
    None
    """
    def __defaults__(self):
        """This sets the default values for the component to function.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """

        self.tag                          = 'rotor'
        self.number_of_blades             = 0.0
        self.tip_radius                   = 0.0
        self.hub_radius                   = 0.0
        self.twist_distribution           = 0.0
        self.sweep_distribution           = 0.0         # quarter chord offset from quarter chord of root airfoil
        self.chord_distribution           = 0.0 
        self.thickness_to_chord           = 0.0
        self.blade_solidity               = 0.0
        self.design_power                 = None
        self.design_thrust                = None
        self.airfoil_geometry             = None
        self.airfoil_polars               = None
        self.airfoil_polar_stations       = None
        self.airfoil_cl_surrogates        = None
        self.airfoil_cd_surrogates        = None
        self.radius_distribution          = None
        self.rotation                     = 1.        
        self.orientation_euler_angles     = [0.,0.,0.]   # This is X-direction thrust in vehicle frame
        self.ducted                       = False
        self.number_azimuthal_stations    = 1
        self.vtk_airfoil_points           = 40
        self.induced_power_factor         = 1.48         # accounts for interference effects
        self.profile_drag_coefficient     = .03
        self.sol_tolerance                = 1e-8
        self.design_power_coefficient     = 0.01


        self.nonuniform_freestream     = False
        self.axial_velocities_2d       = None     # user input for additional velocity influences at the rotor
        self.tangential_velocities_2d  = None     # user input for additional velocity influences at the rotor
        self.radial_velocities_2d      = None     # user input for additional velocity influences at the rotor
        
        self.start_angle               = 0.0      # angle of first blade from vertical
        self.inputs.y_axis_rotation    = 0.
        self.inputs.pitch_command      = 0.
        self.variable_pitch            = False
        
        # JAX static args
        self.static_keys               = ['number_azimuthal_stations','airfoil_geometry','airfoil_polars','number_of_blades','vtk_airfoil_points']
        
        # Initialize the default wake set to Fidelity Zero
        self.Wake                      = Rotor_Wake_Fidelity_Zero()

        
    #@jit
    def spin(self,conditions):
        """Analyzes a general rotor given geometry and operating conditions.
    
        Assumptions:
        per source
    
        Source:
        Drela, M. "Qprop Formulation", MIT AeroAstro, June 2006
        http://web.mit.edu/drela/Public/web/qprop/qprop_theory.pdf
    
        Leishman, Gordon J. Principles of helicopter aerodynamics
        Cambridge university press, 2006.
    
        Inputs:
        self.inputs.omega                    [radian/s]
        conditions.freestream.
          density                            [kg/m^3]
          dynamic_viscosity                  [kg/(m-s)]
          speed_of_sound                     [m/s]
          temperature                        [K]
        conditions.frames.
          body.transform_to_inertial         (rotation matrix)
          inertial.velocity_vector           [m/s]
        conditions.propulsion.
          throttle                           [-]
    
        Outputs:
        conditions.propulsion.outputs.
           number_radial_stations            [-]
           number_azimuthal_stations         [-]
           disc_radial_distribution          [m]
           speed_of_sound                    [m/s]
           density                           [kg/m-3]
           velocity                          [m/s]
           disc_tangential_induced_velocity  [m/s]
           disc_axial_induced_velocity       [m/s]
           disc_tangential_velocity          [m/s]
           disc_axial_velocity               [m/s]
           drag_coefficient                  [-]
           lift_coefficient                  [-]
           omega                             [rad/s]
           disc_circulation                  [-]
           blade_dQ_dR                       [N/m]
           blade_dT_dr                       [N]
           blade_thrust_distribution         [N]
           disc_thrust_distribution          [N]
           thrust_per_blade                  [N]
           thrust_coefficient                [-]
           azimuthal_distribution            [rad]
           disc_azimuthal_distribution       [rad]
           blade_dQ_dR                       [N]
           blade_dQ_dr                       [Nm]
           blade_torque_distribution         [Nm]
           disc_torque_distribution          [Nm]
           torque_per_blade                  [Nm]
           torque_coefficient                [-]
           power                             [W]
           power_coefficient                 [-]
    
        Properties Used:
        self.
          number_of_blades                   [-]
          tip_radius                         [m]
          twist_distribution                 [radians]
          chord_distribution                 [m]
          orientation_euler_angles           [rad, rad, rad]
        """
    
        # Unpack rotor blade parameters
        B       = self.number_of_blades
        R       = self.tip_radius
        beta_0  = self.twist_distribution
        c       = self.chord_distribution
        sweep   = self.sweep_distribution     # quarter chord distance from quarter chord of root airfoil
        r_1d    = self.radius_distribution
        tc      = self.thickness_to_chord
    
        # Unpack rotor airfoil data
        a_geo   = self.airfoil_geometry
        a_loc   = self.airfoil_polar_stations
        cl_sur  = self.airfoil_cl_surrogates
        cd_sur  = self.airfoil_cd_surrogates
    
        # Unpack rotor inputs and conditions
        omega                 = self.inputs.omega
        Na                    = self.number_azimuthal_stations
        nonuniform_freestream = self.nonuniform_freestream
        pitch_c               = self.inputs.pitch_command
    
        # Check for variable pitch
        vp_cond = jnp.any(pitch_c !=0)
        self.variable_pitch = lax.cond(vp_cond,lambda:True,lambda:False)
    
        # Unpack freestream conditions
        rho     = conditions.freestream.density[:,0,None]
        mu      = conditions.freestream.dynamic_viscosity[:,0,None]
        a       = conditions.freestream.speed_of_sound[:,0,None]
        T       = conditions.freestream.temperature[:,0,None]
        Vv      = conditions.frames.inertial.velocity_vector
        nu      = mu/rho
        rho_0   = rho
    
        # Number of radial stations and segment control points
        Nr       = len(c)
        ctrl_pts = len(Vv)
        
        # Helpful shorthands
        pi      = jnp.pi
    
        # Calculate total blade pitch
        total_blade_pitch = beta_0 + pitch_c
    
        # Velocity in the rotor frame
        T_body2inertial = conditions.frames.body.transform_to_inertial
        T_inertial2body = orientation_transpose(T_body2inertial)
        V_body          = orientation_product_jax(T_inertial2body,Vv)
        body2thrust     = self.body_to_prop_vel()
        
        T_body2thrust   = orientation_transpose(jnp.ones_like(T_body2inertial[:])*body2thrust)
        V_thrust        = orientation_product_jax(T_body2thrust,V_body)
    
        # Check and correct for hover
        V = V_thrust[:,0,None]
        V = jnp.where(V==0.0,1E-6,V)
    
        # Non-dimensional radial distribution and differential radius
        chi    = r_1d/R
        deltar = jnp.gradient(r_1d)
        
        # Calculating rotational parameters
        omegar   = jnp.outer(omega,r_1d)
        n        = omega/(2.*pi)   # Rotations per second
    
        # 2 dimensional radial distribution non dimensionalized
        chi_2d         = jnp.tile(chi[:, None],(1,Na))
        chi_2d         = jnp.repeat(chi_2d[None,:,:], ctrl_pts, axis=0)
        r_dim_2d       = jnp.tile(r_1d[:, None] ,(1,Na))
        r_dim_2d       = jnp.repeat(r_dim_2d[None,:,:], ctrl_pts, axis=0)
        c_2d           = jnp.tile(c[:, None] ,(1,Na))
        c_2d           = jnp.repeat(c_2d[None,:,:], ctrl_pts, axis=0)
    
        # Azimuthal distribution of stations (in direction of rotation)
        psi            = jnp.linspace(0,2*pi,Na+1)[:-1]
        psi_2d         = jnp.tile(jnp.atleast_2d(psi),(Nr,1))
        psi_2d         = jnp.repeat(psi_2d[None, :, :], ctrl_pts, axis=0)

        # apply blade sweep to azimuthal position, if it's zero it will add zero
        sweep               = jnp.atleast_3d(sweep)
        sweep_2d            = jnp.repeat(sweep,Na,axis=2)
        sweep_offset_angles = jnp.tan(sweep_2d/r_dim_2d)
        psi_2d             += sweep_offset_angles      
    
        # Starting with uniform freestream
        ua       = 0
        ut       = 0
        ur       = 0
    
        # Include velocities introduced by rotor incidence angles
    
        # y-component of freestream in the propeller cartesian plane
        Vy  = V_thrust[:,1,None,None]
        Vy  = jnp.repeat(Vy, Nr,axis=1)
        Vy  = jnp.repeat(Vy, Na,axis=2)

        # z-component of freestream in the propeller cartesian plane
        Vz  = V_thrust[:,2,None,None]
        Vz  = jnp.repeat(Vz, Nr,axis=1)
        Vz  = jnp.repeat(Vz, Na,axis=2)

        # compute resulting radial and tangential velocities in polar frame
        utz =  -Vz*jnp.sin(psi_2d)
        urz =   Vz*jnp.cos(psi_2d)
        uty =  -Vy*jnp.cos(psi_2d)
        ury =   Vy*jnp.sin(psi_2d)

        ut +=  (utz + uty)  # tangential velocity in direction of rotor rotation
        ur +=  (urz + ury)  # radial velocity (positive toward tip)
        ua +=  jnp.zeros_like(ut)
        
        # Include external velocities introduced by user
        if nonuniform_freestream:
    
            # include additional influences specified at rotor sections, shape=(ctrl_pts,Nr,Na)
            ua += self.axial_velocities_2d
            ut += self.tangential_velocities_2d
            ur += self.radial_velocities_2d
    
        # 2-D freestream velocity and omega*r
        V_2d   = V_thrust[:,0,None,None]
        V_2d   = jnp.repeat(V_2d, Na,axis=2)
        V_2d   = jnp.repeat(V_2d, Nr,axis=1)
        omegar = (jnp.repeat(jnp.outer(omega,r_1d)[:,:,None], Na, axis=2))
    
        # total velocities
        Ua     = V_2d + ua
    
        # 2-D blade pitch and radial distributions
        if np.size(pitch_c)>1:
            # control variable is the blade pitch, repeat around azimuth
            beta = jnp.repeat(total_blade_pitch[:,:,None], Na, axis=2)
        else:
            beta = jnp.tile(total_blade_pitch[None,:,None],(ctrl_pts,1,Na ))
    
        r      = jnp.tile(r_1d[None,:,None], (ctrl_pts, 1, Na))
        c      = jnp.tile(c[None,:,None], (ctrl_pts, 1, Na))
        deltar = jnp.tile(deltar[None,:,None], (ctrl_pts, 1, Na))
    
        # 2-D atmospheric properties
        a   = jnp.tile(jnp.atleast_2d(a),(1,Nr))
        a   = jnp.repeat(a[:, :, None], Na, axis=2)
        nu  = jnp.tile(jnp.atleast_2d(nu),(1,Nr))
        nu  = jnp.repeat(nu[:,  :, None], Na, axis=2)
        rho = jnp.tile(jnp.atleast_2d(rho),(1,Nr))
        rho = jnp.repeat(rho[:,  :, None], Na, axis=2)
        T   = jnp.tile(jnp.atleast_2d(T),(1,Nr))
        T   = jnp.repeat(T[:, :, None], Na, axis=2)
    
        # Total velocities
        Ut     = omegar - ut
        U      = jnp.sqrt(Ua*Ua + Ut*Ut + ur*ur)
        
        
        #---------------------------------------------------------------------------
        # COMPUTE WAKE-INDUCED INFLOW VELOCITIES AND RESULTING ROTOR PERFORMANCE
        #---------------------------------------------------------------------------
        # pack inputs
        wake_inputs                       = Data()
        wake_inputs.velocity_total        = U
        wake_inputs.velocity_axial        = Ua
        wake_inputs.velocity_tangential   = Ut
        wake_inputs.ctrl_pts              = ctrl_pts
        wake_inputs.Nr                    = Nr
        wake_inputs.Na                    = Na        
        wake_inputs.twist_distribution    = beta
        wake_inputs.chord_distribution    = c
        wake_inputs.radius_distribution   = r
        wake_inputs.speed_of_sounds       = a
        wake_inputs.dynamic_viscosities   = nu      
    
        va, vt = self.Wake.evaluate(self,wake_inputs,conditions)
        
        # compute new blade velocities
        Wa   = va + Ua
        Wt   = Ut - vt
        
        lamdaw, F, _ = compute_inflow_and_tip_loss(r,R,Wa,Wt,B)
    
        # Compute aerodynamic forces based on specified input airfoil or surrogate
        Cl, Cdval, alpha, Ma,W = compute_airfoil_aerodynamics(beta,c,r,R,B,Wa,Wt,a,nu,a_loc,a_geo,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc)
        
        # compute HFW circulation at the blade
        Gamma = 0.5*W*c*Cl  
    
        #---------------------------------------------------------------------------            
                
        # tip loss correction for velocities, since tip loss correction is only applied to loads in prior BET iteration
        va     = F*va
        vt     = F*vt
        lamdaw = r*(va+Ua)/(R*(Ut-vt))
    
        # More Cd scaling from Mach from AA241ab notes for turbulent skin friction
        Tw_Tinf     = 1. + 1.78*(Ma*Ma)
        Tp_Tinf     = 1. + 0.035*(Ma*Ma) + 0.45*(Tw_Tinf-1.)
        Tp          = (Tp_Tinf)*T
        Rp_Rinf     = (Tp_Tinf**2.5)*(Tp+110.4)/(T+110.4)
        Cd          = ((1/Tp_Tinf)*(1/Rp_Rinf)**0.2)*Cdval
    
        epsilon                  = Cd/Cl
        epsilon                  = jnp.where(epsilon==jnp.inf,10.,epsilon)
    
        # thrust and torque and their derivatives on the blade.
        blade_T_distribution     = rho*(Gamma*(Wt-epsilon*Wa))*deltar
        blade_Q_distribution     = rho*(Gamma*(Wa+epsilon*Wt)*r)*deltar
        blade_dT_dr              = rho*(Gamma*(Wt-epsilon*Wa))
        blade_dQ_dr              = rho*(Gamma*(Wa+epsilon*Wt)*r)
 
        blade_T_distribution_2d = blade_T_distribution
        blade_Q_distribution_2d = blade_Q_distribution
        blade_dT_dr_2d          = blade_dT_dr
        blade_dQ_dr_2d          = blade_dQ_dr
        blade_Gamma_2d          = Gamma
        alpha_2d                = alpha
    
        Va_2d = Wa
        Vt_2d = Wt
        Va_avg = jnp.average(Wa, axis=2)      # averaged around the azimuth
        Vt_avg = jnp.average(Wt, axis=2)      # averaged around the azimuth
    
        Va_ind_2d  = va
        Vt_ind_2d  = vt
        Vt_ind_avg = jnp.average(vt, axis=2)
        Va_ind_avg = jnp.average(va, axis=2)
    
        # set 1d blade loadings to be the average:
        blade_T_distribution    = jnp.mean((blade_T_distribution_2d), axis = 2)
        blade_Q_distribution    = jnp.mean((blade_Q_distribution_2d), axis = 2)
        blade_dT_dr             = jnp.mean((blade_dT_dr_2d), axis = 2)
        blade_dQ_dr             = jnp.mean((blade_dQ_dr_2d), axis = 2)
    
        # compute the hub force / rotor drag distribution along the blade
        dL_2d    = 0.5*rho*c_2d*Cd*omegar**2*deltar
        dD_2d    = 0.5*rho*c_2d*Cl*omegar**2*deltar
    
        rotor_drag_distribution = jnp.mean(dL_2d*jnp.sin(psi_2d) + dD_2d*jnp.cos(psi_2d),axis=2)
    
        # forces
        thrust                  = jnp.atleast_2d((B * jnp.sum(blade_T_distribution, axis = 1))).T
        torque                  = jnp.atleast_2d((B * jnp.sum(blade_Q_distribution, axis = 1))).T
        rotor_drag              = jnp.atleast_2d((B * jnp.sum(rotor_drag_distribution, axis=1))).T
        power                   = omega*torque

        # calculate coefficients
        D        = 2*R
        Cq       = torque/(rho_0*(n*n)*(D*D*D*D*D))
        Ct       = thrust/(rho_0*(n*n)*(D*D*D*D))
        Cp       = power/(rho_0*(n*n*n)*(D*D*D*D*D))
        Crd      = rotor_drag/(rho_0*(n*n)*(D*D*D*D))
        etap     = V*thrust/power
        A        = jnp.pi*(R**2 - self.hub_radius**2)
        FoM      = thrust*jnp.sqrt(thrust/(2*rho_0*A))    /power  
    
        # prevent things from breaking
        O_cond     = omega==0.0
        T_cond     = conditions.propulsion.throttle<=0.0
        Cq         = jnp.where(Cq<0,0,Cq)
        Ct         = jnp.where(Ct<0,0,Ct)
        Cp         = jnp.where(Cp<0,0,Cp)
        power      = jnp.where(T_cond,0,power)
        thrust     = jnp.where(T_cond,0,thrust)
        torque     = jnp.where(T_cond,0,torque)
        rotor_drag = jnp.where(T_cond,0,rotor_drag )
        thrust     = jnp.where(O_cond,0,thrust)
        torque     = jnp.where(O_cond,0,torque)
        rotor_drag = jnp.where(O_cond,0,rotor_drag)
        Cp         = jnp.where(O_cond,0,Cp)
        Ct         = jnp.where(O_cond,0,Ct)        
        etap       = jnp.where(O_cond,0,etap)      
        thrust     = jnp.where(omega<0.,-thrust,thrust)        
        
        # Make the thrust a 3D vector
        thrust_prop_frame      = jnp.repeat(jnp.atleast_2d([1,0,0]),repeats=ctrl_pts,axis=0)*thrust
        thrust_vector          = orientation_product_jax(orientation_transpose(T_body2thrust),thrust_prop_frame)
    
        # Assign efficiency to network
        conditions.propulsion.etap = etap
    
    
        # Store data
        self.azimuthal_distribution                   = psi
        results_conditions                            = Data
        outputs                                       = results_conditions(
                    number_radial_stations            = Nr,
                    number_azimuthal_stations         = Na,
                    disc_radial_distribution          = r_dim_2d,
                    speed_of_sound                    = conditions.freestream.speed_of_sound,
                    density                           = conditions.freestream.density,
                    velocity                          = Vv,
                    blade_tangential_induced_velocity = Vt_ind_avg,
                    blade_axial_induced_velocity      = Va_ind_avg,
                    blade_tangential_velocity         = Vt_avg,
                    blade_axial_velocity              = Va_avg,
                    disc_tangential_induced_velocity  = Vt_ind_2d,
                    disc_axial_induced_velocity       = Va_ind_2d,
                    disc_tangential_velocity          = Vt_2d,
                    disc_axial_velocity               = Va_2d,
                    drag_coefficient                  = Cd,
                    lift_coefficient                  = Cl,
                    omega                             = omega,
                    disc_circulation                  = blade_Gamma_2d,
                    blade_dT_dr                       = blade_dT_dr,
                    disc_dT_dr                        = blade_dT_dr_2d,
                    blade_thrust_distribution         = blade_T_distribution,
                    disc_thrust_distribution          = blade_T_distribution_2d,
                    disc_effective_angle_of_attack    = alpha_2d,
                    thrust_per_blade                  = thrust/B,
                    thrust_coefficient                = Ct,
                    disc_azimuthal_distribution       = psi_2d,
                    blade_dQ_dr                       = blade_dQ_dr,
                    disc_dQ_dr                        = blade_dQ_dr_2d,
                    blade_torque_distribution         = blade_Q_distribution,
                    disc_torque_distribution          = blade_Q_distribution_2d,
                    torque_per_blade                  = torque/B,
                    torque_coefficient                = Cq,
                    power                             = power,
                    power_coefficient                 = Cp,
                    converged_inflow_ratio            = lamdaw,
                    propeller_efficiency              = etap,
                    blade_H_distribution              = rotor_drag_distribution,
                    rotor_drag                        = rotor_drag,
                    rotor_drag_coefficient            = Crd,
                    figure_of_merit                   = FoM,
                    tip_mach                          = omega * R / conditions.freestream.speed_of_sound
            )
        self.outputs = outputs
    
        return thrust_vector, torque, power, Cp, outputs , etap
        
        

    
    def vec_to_vel(self):
        """This rotates from the propellers vehicle frame to the propellers velocity frame

        Assumptions:
        There are two propeller frames, the vehicle frame describing the location and the propeller velocity frame
        velocity frame is X out the nose, Z towards the ground, and Y out the right wing
        vehicle frame is X towards the tail, Z towards the ceiling, and Y out the right wing

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """

        rot_mat = sp.spatial.transform.Rotation.from_rotvec([0,np.pi,0]).as_matrix()

        return rot_mat
    

    def body_to_prop_vel(self):
        """This rotates from the systems body frame to the propellers velocity frame

        Assumptions:
        There are two propeller frames, the vehicle frame describing the location and the propeller velocity frame
        velocity frame is X out the nose, Z towards the ground, and Y out the right wing
        vehicle frame is X towards the tail, Z towards the ceiling, and Y out the right wing

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """

        # Go from body to vehicle frame
        body_2_vehicle = sp.spatial.transform.Rotation.from_rotvec([0,np.pi,0]).as_matrix()

        # Go from vehicle frame to propeller vehicle frame: rot 1 including the extra body rotation
        cpts       = len(jnp.atleast_1d(self.inputs.y_axis_rotation))
        rots       = jnp.array(self.orientation_euler_angles) * 1.
        rots       = jnp.repeat(rots[None,:], cpts, axis=0)
        rots.at[:,1].add(jnp.atleast_2d(self.inputs.y_axis_rotation)[:,0])
        
        #vehicle_2_prop_vec = sp.spatial.transform.Rotation.from_rotvec(rots).as_matrix()
        T1 = rots[:,0]
        T2 = rots[:,1]
        T3 = rots[:,2]
        
        Z   = jnp.zeros_like(T1)
        O   = jnp.ones_like(T1)
        CT1 = jnp.cos(T1)
        CT2 = jnp.cos(T2)
        CT3 = jnp.cos(T3)
        ST1 = jnp.sin(T1)
        ST2 = jnp.sin(T2)
        ST3 = jnp.sin(T3)
        
        R1  = jnp.array([[O,Z,Z],[Z,CT1,-ST1],[Z,ST1,CT1]]).T
        R2  = jnp.array([[CT2,Z,ST2],[Z,O,Z],[-ST2,Z,CT2]]).T
        R3  = jnp.array([[CT3,-ST3,Z],[ST3,CT3,Z],[Z,Z,O]]).T
        
        R2_R1 = jnp.matmul(R2,R1)
        vehicle_2_prop_vec = jnp.matmul(R3,R2_R1)

        # Go from the propeller vehicle frame to the propeller velocity frame: rot 2
        prop_vec_2_prop_vel = self.vec_to_vel()

        # Do all the matrix multiplies
        rot1    = jnp.matmul(body_2_vehicle,vehicle_2_prop_vec)
        rot_mat = jnp.matmul(rot1,prop_vec_2_prop_vel)


        return rot_mat


    def prop_vel_to_body(self):
        """This rotates from the propeller's velocity frame to the system's body frame

        Assumptions:
        There are two propeller frames, the vehicle frame describing the location and the propeller velocity frame
        velocity frame is X out the nose, Z towards the ground, and Y out the right wing
        vehicle frame is X towards the tail, Z towards the ceiling, and Y out the right wing

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """

        body2propvel = self.body_to_prop_vel()

        r = sp.spatial.transform.Rotation.from_matrix(body2propvel)
        r = r.inv()
        rot_mat = r.as_matrix()

        return rot_mat
    
    def vec_to_prop_body(self):
        return self.prop_vel_to_body()
