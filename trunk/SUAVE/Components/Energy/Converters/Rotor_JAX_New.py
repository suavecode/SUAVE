## @ ingroup Component-Energy-Converters
# Rotor_JAX.py
#
# Created:  Feb 2022, J. Smart
# Modified:

#-----------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------

from SUAVE.Core import Data, Units
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Methods.Geometry.Three_Dimensional \
    import orientation_product, orientation_transpose
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_HFW_inflow_velocties \
    import compute_HFW_inflow_velocities

import jax.numpy as np
import jax.lax as lax
import numpy as onp
import scipy as sp

from copy import  deepcopy
#-----------------------------------------------------------------------
# Differentiable & Accelerable Rotor Class
#-----------------------------------------------------------------------

class Rotor(Energy_Component):
    """This is a modification of SUAVE's basic rotor class made to be
    compatible with JAX's autodifferentiation and GPU acceleration
    capabilities. It is presenently maintained separately from the
    primary rotor class to allow for reference development.

    Assumptions:
    None

    Soure:
    None
    """
    def __defaults__(self):
        """This sets the default values for the component to function.

        Assumptions:
        None

        Source:
        None

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """

        self.tag                            = 'rotor'
        self.number_of_blades               = 0.0
        self.tip_radius                     = 0.0
        self.hub_radius                     = 0.0
        self.twist_distribution             = 0.0
        self.sweep_distribution             = 0.0           # Quarter-chord offset from quarter-chord of root airfoil
        self.chord_distribution             = 0.0
        self.mid_chord_alignment            = 0.0
        self.thickness_to_chord             = 0.0
        self.blade_solidity                 = 0.0
        self.design_power                   = None
        self.design_thrust                  = None
        self.airfoil_geometry               = None
        self.airfoil_polars                 = None
        self.airfoil_polar_stations         = None
        self.radius_distribution            = None
        self.rotation                       = 1
        self.azimuthal_offset_angle         = 0.0
        self.orientation_euler_angles       = [0.,0.,0.]    # This is x-direction thrust in vehicle frame
        self.ducted                         = False
        self.number_azimuthal_stations      = 24
        self.number_points_around_airfoil   = 40
        self.induced_power_factor           = 1.48          # Accounts for interference effects
        self.profile_drag_coefficient       = .03

        self.use_2d_analysis                = False         # True if rotor is at an angle relative to freestream or if there is a non-uniform freestream
        self.nonuniform_freestream          = False
        self.axial_velocities_2d            = None          # User input for additional velocity influences at the rotor
        self.tangential_velocities_2d       = None          # User input for additional velocity influences at the rotor
        self.radial_velocities_2d           = None          # User input for additional velocity influences at the rotor

        self.Wake_VD                        = Data()
        self.wake_method                    = "momentum"
        self.number_rotor_rotations         = 6
        self.number_steps_per_rotation      = 100
        self.wake_settings                  = Data()

        self.wake_settings.initial_timestep_offset  = 0     # Initial timestep
        self.wake_settings.wake_development_time    = 0.05  # Total simulation time required for wake development
        self.wake_settings.number_of_wake_timesteps = 30    # Total number of time steps in wake development
        self.start_angle                            = 0.0   # Angle of first blade from vertical

        self.inputs.y_axis_rotation         = 0.
        self.inputs.pitch_command           = 0.
        self.variable_pitch                 = False

    def spin(self, conditions):
        """Analyzes a general rotor given geometry and operating conditions.
        Numerically modified from basic SUAVE rotor.

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
        sweep   = self.sweep_distribution
        r_1d    = self.radius_distribution
        tc      = self.thickness_to_chord

        # Unpack rotor airfoil data
        a_geo   = self.airfoil_geometry
        a_loc   = self.airfoil_polar_surrogates
        cl_sur  = self.airfoil_cl_surrogates
        cd_sur  = self.airfoil_cd_surrogates

        # Unpack rotor inputs and conditions
        omega                   = self.inputs.omega
        Na                      = self.number_azimuthal_stations
        nonuniform_freestream   = self.nonuniform_freestream
        use_2d_analysis         = self.use_2d_analysis
        wake_method             = self.wake_method
        rotation                = self.rotation
        pitch_c                 = self.inputs.pitch_command

        # Check for variable pitch

        # if np.any(pitch_c !=0) and not self.variable_pitch:
        #
        #     print("Warning: Pitch commanded for a fixed-pitch rotor. \
        #     Changing to variable pitch rotor for weights analysis.")
        #
        #     self.variable_pitch = True

        self.variable_pitch = lax.cond(np.any(pitch_c !=0) and not self.variable_pitch,
                                       lambda _: True,
                                       lambda _: self.variable_pitch,
                                       None)

        # Unpack freestream conditions
        rho     = conditions.freestream.density[:,0,None]
        mu      = conditions.freestream.dynamic_viscosity[:,0,None]
        a       = conditions.freestream.speed_of_sound[:,0,None]
        T       = conditions.freestream.temperature[0,:,None]
        Vv      = conditions.frames.inertia.velocity_vector
        nu      = mu/rho
        rho_0   = rho

        # Helpful shorthands
        pi      = np.pi

        # Calculate total blade pitch
        total_blade_pitch   = beta_0 + pitch_c

        # Velocity in the rotor frame
        T_body2inertial     = conditions.frames.body.transform_to_inertial
        T_inertial2body     = orientation_transpose(T_body2inertial)
        V_body              = orientation_product(T_inertial2body, Vv)
        body2thrust         = self.body_to_prop_vel()
        T_body2thrust       = orientation_transpose(np.ones_like(T_body2inertial[:])*body2thrust)
        V_thrust            = orientation_product(T_body2thrust, V_body)

        # Check and correct for hover
        V               = V_thrust[:,0,None]
        V.at[V==0.0].set(1e-6)

        # Number of radial stations and segment control points
        Nr          = len(c)
        ctrl_pts    = len(Vv)

        # Non-dimensional radial distribution and differential radius
        chi             = r_1d/R
        diff_r          = np.diff(r_1d)
        deltar          = np.zeros(len(r_1d))
        deltar.at[1:-1] = diff_r[0:-1]/2 + diff_r[1:]/2
        deltar.at[0]    = diff_r[0]/2
        deltar.at[-1]   = diff_r[-1]/2

        # Calculating rotational parameters
        omegar  = np.outer(omega, r_1d)
        n       = omega/(2.*pi)             # Rotations per second

        # 2 dimensional radial distribution non-dimensionalized
        chi_2d      = np.tile(chi[:,None], (1,Na))
        chi_2d      = np.repeat(chi_2d[None,:,:], ctrl_pts, axis=0)
        r_dim_2d    = np.tile(r_1d[:,None], (1,Na))
        r_dim_2d    = np.repeat(r_dim_2d[None,:,:], ctrl_pts, axis=0)
        c_2d        = np.tile(c[:,None], (1,Na))
        c_2d        = np.repeat(c_2d[None,:,:], ctrl_pts, axis=0)

        # Azimuthal distribution of stations
        psi         = np.linspace(0,2*pi,Na+1)[:-1]
        psi_2d      = np.tile(np.atleast_2d(psi), (Nr,1))
        psi_2d      = np.repeat(psi_2d[None,:,:], ctrl_pts, axis=0)

        # Apply blade sweep to azimuthal position
        # if np.any(np.array([sweep])!=0):
        #     use_2d_analysis     = True
        #     sweep_2d            = np.repeat(sweep[:,None], (1,Na))
        #     sweep_offset_angles = np.tan(sweep_2d/r_dim_2d)
        #     psi_2d              += sweep_offset_angles

        def non_zero_sweep(sweep, Na, r_dim_2d, psi_2d):

            sweep_2d            = np.repeat(sweep[:,None], (1,Na))
            sweep_offset_angles = np.tan(sweep_2d/r_dim_2d)

            return True, psi_2d+sweep_offset_angles


        use_2d_analysis, psi_2d = lax.cond(np.any(np.array([sweep])!=0),
                                           non_zero_sweep,
                                           lambda _ : use_2d_analysis, psi_2d,
                                           sweep, Na, r_dim_2d, psi_2d)

        # Starting with uniform freestream
        ua  = 0
        ut  = 0
        ur  = 0

        # rotation = lax.cond(rotation == -1,
        #                     lambda _ : -1,
        #                     lambda _ : 1,
        #                     None)

        # Include velocities introduced by rotor incidence angles
        if (np.any(abs(V_thrust[:,1]) > 1e-3) or np.any(abs(V_thrust[:,2]) > 1e-3)) and use_2d_analysis:

            # y-component of freestream in the propeller Cartesian plane

            Vy = V_thrust[:,1,None,None]
            Vy = np.repeat(Vy, Nr, axis=1)
            Vy = np.repeat(Vy, Na, axis=2)

            # z-component of freestream in the propeller Cartesian plane

            Vz = V_thrust[:,2,None,None]
            Vz = np.repeat(Vz, Nr, axis=1)
            Vz = np.repeat(Vz, Na, axis=2)

            # Check for invalid rotation angle
            if (rotation==1) or (rotation==-1):
                pass
            else:
                print("Invalid rotation direction. Setting to 1.")
                rotation = 1

            # Compute resulting radial and tangential velocities in polar frame
            utz = Vz*np.cos(psi_2d*rotation)
            urz = Vz*np.sin(psi_2d*rotation)
            uty = Vy*np.sin(psi_2d*rotation)
            ury = Vy*np.cos(psi_2d*rotation)

            ua += (utz + uty)
            ut += (urz + ury)
            ua += np.zeros_like(ut)

        # Include external velocities introduced by user
        if nonuniform_freestream:
            use_2d_analysis = True

            # Include additional influences specified at rotor sections, shape=(ctrl_pts,Nr,Na)
            ua += self.axial_velocities_2d
            ut += self.tangential_velocities_2d
            ur += self.radial_velocities_2d

        if use_2d_analysis:
            # Make everything 2D with shape (ctrl_pts,Nr,Na)
            size    = (ctrl_pts,Nr,Na)
            PSI     = np.ones(size)
            PSIold  = np.zeros(size)

            # 2D freestream velocity and omega*r
            V_2d    = V_thrust[:,0,None,None]
            V_2d    = np.repeat(V_2d, Na, axis=2)
            V_2d    = np.repeat(V_2d, Nr, axis=1)
            omegar  = (np.repeat(np.outer(omega, r_1d)[:,:,None], Na, axis=2))

            # Total velocities
            Ua      = V_2d + ua

            # 2D blade pitch and radial distributions
            if np.size(pitch_c)>1:
                # Control variable is the blade pitch, repeat around azimuth
                beta = np.repeat(total_blade_pitch[:,:,None], Na, axis=2)
            else:
                beta = np.tile(total_blade_pitch[None,:,None], (ctrl_pts, 1, Na))

            r       = np.tile(r_1d[None,:,None], (ctrl_pts, 1, Na))
            c       = np.tile(c[None,:,None], (ctrl_pts, 1, Na))
            deltar  = np.tile(deltar[None,:,None], (ctrl_pts, 1, Na))

            # 2D atmospheric properties
            a   = np.tile(np.atleast_2d(a), (1, Nr))
            a   = np.repeat(a[:,:,None], Na, axis=2)
            nu = np.tile(np.atleast_2d(nu), (1, Nr))
            nu = np.repeat(nu[:, :, None], Na, axis=2)
            rho = np.tile(np.atleast_2d(rho), (1, Nr))
            rho = np.repeat(rho[:, :, None], Na, axis=2)
            T = np.tile(np.atleast_2d(T), (1, Nr))
            T = np.repeat(T[:, :, None], Na, axis=2)

        else:
            # Total velocities
            r       = r_1d
            Ua      = np.outer((V + ua), np.ones_like(r))
            beta    = total_blade_pitch

            # Things that will change with iteration
            size    = (ctrl_pts, Nr)
            PSI     = np.ones(size)
            PSIold  = np.zeros(size)

        # Total velocities
        Ut  = omegar - ut
        U   = np.sqrt(Ua*Ua + Ut*Ut + ur*ur)

        # Momentum Wake Method (TODO: Implement Helical Fixed Wake)

        diff    = 1.
        tol     = 1E-6
        ii      = 0

        while (diff > tol):

            # Compute velocities
            sin_psi = np.sin(PSI)
            cos_psi = np.cos(PSI)
            Wa      = 0.5*Ua + 0.5*U*sin_psi
            Wt      = 0.5*Ut + 0.5*U*cos_psi
            va      = Wa - Ua
            vt      = Ut - Wt

            # Compute blade airfoil forces and properties
            Cl, Cdval, alpha, Ma, W     = compute_airfoil_aerodynamics(beta,c,r,R,B,
                                                                       Wa,Wt,a,nu,
                                                                       a_loc,a_geo,cl_sur,cd_sur,
                                                                       ctrl_pts,Nr,Na,tc,
                                                                       use_2d_analysis)

            


def compute_airfoil_aerodynamics(beta,c,r,R,B,
                                Wa,Wt,a,nu,
                                a_loc,a_geo,cl_sur,cd_sur,
                                ctrl_pts,Nr,Na,tc,
                                use_2d_analysis):
    """
    Cl, Cdval = compute_airfoil_aerodynamics( beta,c,r,R,B,
                                              Wa,Wt,a,nu,
                                              a_loc,a_geo,cl_sur,cd_sur,
                                              ctrl_pts,Nr,Na,tc,use_2d_analysis )

    Computes the aerodynamic forces at sectional blade locations. If airfoil
    geometry and locations are specified, the forces are computed using the
    airfoil polar lift and drag surrogates, accounting for the local Reynolds
    number and local angle of attack.

    If the airfoils are not specified, an approximation is used.

    Modified for use in JAX-based autodifferentiation and GPU acceleration.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
       beta                       blade twist distribution                        [-]
       c                          chord distribution                              [-]
       r                          radius distribution                             [-]
       R                          tip radius                                      [-]
       B                          number of rotor blades                          [-]

       Wa                         axial velocity                                  [-]
       Wt                         tangential velocity                             [-]
       a                          speed of sound                                  [-]
       nu                         viscosity                                       [-]

       a_loc                      Locations of specified airfoils                 [-]
       a_geo                      Geometry of specified airfoil                   [-]
       cl_sur                     Lift Coefficient Surrogates                     [-]
       cd_sur                     Drag Coefficient Surrogates                     [-]
       ctrl_pts                   Number of control points                        [-]
       Nr                         Number of radial blade sections                 [-]
       Na                         Number of azimuthal blade stations              [-]
       tc                         Thickness to chord                              [-]
       use_2d_analysis            Specifies 2d disc vs. 1d single angle analysis  [Boolean]

    Outputs:
       Cl                       Lift Coefficients                         [-]
       Cdval                    Drag Coefficients  (before scaling)       [-]
       alpha                    section local angle of attack             [rad]

    """

    alpha   = beta - np.arctan2(Wa,Wt)
    W       = (Wa*Wa + Wt*Wt)**0.5
    Ma      = W/a
    Re      = (W*c)/nu

    # If propeller airfoils are defined, use airfoil surrogate
    if a_loc != None:

        print("Use of specified airfoils for aerodynamics is currently unsupported. Defaulting to estimation method.")

        # # Compute blade Cl and Cd distribution from the airfoil data
        # dim_sur = len(cl_sur)
        # if use_2d_analysis:
        #     # Return the 2D Cl and CDval of shape (ctrl_pts, Nr, Na)
        #     Cl      = np.zeros((ctrl_pts, Nr, Na))
        #     Cdval   = np.zeros((ctrl_pts, Nr, Na))
        #     for jj in range(dim_sur):
        #         Cl_af       = cl_sur[a_geo[jj]](Re, alpha, grid=False)
        #         Cdval_af    = cd_sur[a_geo[jj]](Re, alpha, grid=False)
        #         locs        = np.where(np.array(a_loc) == jj)
        #
        #         Cl.at[:,locs,:].set(Cl_af[:,locs,:])
        #         Cdval.at[:,locs,:].set(Cdval_af[:,locs,:])
        #
        # else:
        #     # Return the 1D Cl and CDval of shape (ctrl_pts, Nr)
        #     Cl      = np.zeros((ctrl_pts, Nr))
        #     Cdval   = np.zeros((ctrl_pts, Nr))
        #
        #     for jj in range(dim_sur):
        #         Cl_af       = cl_sur[a_geo[jj]](Re, alpha, grid=False)
        #         Cdval_af    = cd_sur[a_geo[jj]](Re, alpha, grid=False)
        #         locs        = np.where(np.array(a_loc) == jj)
        #
        #         Cl.at[:,locs].set(Cl_af[:, locs])
        #         Cdval.at[:,locs].set(Cdval_af[:,locs])

    Cl_max_ref  = -0.0009*tc**3 + 0.0217*tc**2 - 0.0442*tc + 0.7005
    Re_ref      = 9E6
    Cl1maxp     = Cl_max_ref * (Re/Re_ref)**0.1

    Cl          = 2.*np.pi*alpha
    Cl.at[Cl > Cl1maxp].set(Cl1maxp[0][[Cl > Cl1maxp][0][0]])
    Cl.at[alpha >= np.pi / 2].set(0.)
    Cl.at[Ma[:, :] < 1.].set(Cl[Ma[:, :] < 1.] / ((1 - Ma[Ma[:, :] < 1.] * Ma[Ma[:, :] < 1.]) ** 0.5 + (
                (Ma[Ma[:, :] < 1.] * Ma[Ma[:, :] < 1.]) / (1 + (1 - Ma[Ma[:, :] < 1.] * Ma[Ma[:, :] < 1.]) ** 0.5)) *
                                                  Cl[Ma < 1.] / 2))
    Cl.at[Ma[:, :] >= 1.].set(Cl[Ma[:, :] >= 1.])

    Cdval = (0.108 * (Cl * Cl * Cl * Cl) - 0.2612 * (Cl * Cl * Cl) + 0.181 * (Cl * Cl) - 0.0139 * Cl + 0.0278) * (
                (50000. / Re) ** 0.2)
    Cdval.at[alpha >= np.pi / 2].set(2.)

    return Cl, Cdval
















