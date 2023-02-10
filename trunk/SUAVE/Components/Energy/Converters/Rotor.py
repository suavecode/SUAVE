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
from SUAVE.Core import Data, Units
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Analyses.Propulsion.Rotor_Wake_Fidelity_Zero import Rotor_Wake_Fidelity_Zero
from SUAVE.Analyses.Propulsion.Rotor_Wake_Fidelity_One import Rotor_Wake_Fidelity_One
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.BET_calculations \
     import compute_airfoil_aerodynamics,compute_inflow_and_tip_loss
from SUAVE.Methods.Geometry.Three_Dimensional \
     import  orientation_product, orientation_transpose

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wing_induced_velocity import compute_wing_induced_velocity
# package imports
import numpy as np
import scipy as sp
import copy
import pandas as pd
from pykrige.uk import UniversalKriging
from pykrige.ok import OrdinaryKriging as Kriging
from csv import writer


# ----------------------------------------------------------------------
#  Generalized Rotor Class
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
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
        self.mid_chord_alignment          = 0.0 
        self.blade_solidity               = 0.0
        self.design_power                 = None
        self.design_thrust                = None
        self.airfoil_geometry             = None
        self.airfoil_data                 = None
        self.airfoil_polars               = None
        self.airfoil_polar_stations       = None
        self.radius_distribution          = None
        self.rotation                     = 1        
        self.orientation_euler_angles     = [0.,0.,0.]   # This is X-direction thrust in vehicle frame
        self.ducted                       = False
        self.number_azimuthal_stations    = 24
        self.vtk_airfoil_points           = 40
        self.induced_power_factor         = 1.48         # accounts for interference effects
        self.profile_drag_coefficient     = .03
        self.sol_tolerance                = 1e-8
        self.design_power_coefficient     = 0.01

        
        self.use_2d_analysis                    = True    # True if rotor is at an angle relative to freestream or nonuniform freestream
        self.nonuniform_freestream              = False
        self.external_axial_disc_velocity       = None     # user input for additional velocity influences at the rotor
        self.external_tangential_disc_velocity  = None     # user input for additional velocity influences at the rotor
        self.external_radial_disc_velocity      = None     # user input for additional velocity influences at the rotor
        
        self.start_angle               = 0.0      # angle of first blade from vertical
        self.start_angle_idx           = 0        # azimuthal index at which the blade is started
        self.inputs.y_axis_rotation    = 0.
        self.inputs.pitch_command      = 0.
        self.variable_pitch            = False
        self.surrogate_spin_flag       = False
        
        # Initialize the default wake set to Fidelity Zero
        self.Wake                      = Rotor_Wake_Fidelity_Zero()
        

    def spin(self,conditions,VD=None):
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
        Rh      = self.hub_radius
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
        use_2d_analysis       = self.use_2d_analysis
        pitch_c               = self.inputs.pitch_command
        
        # 2d analysis required for wake fid1
        if isinstance(self.Wake, Rotor_Wake_Fidelity_One):
            use_2d_analysis=True

        # Check for variable pitch
        if np.any(pitch_c !=0) and not self.variable_pitch:
            print("Warning: pitch commanded for a fixed-pitch rotor. Changing to variable pitch rotor for weights analysis.")
            self.variable_pitch = True

        # Unpack freestream conditions
        rho     = conditions.freestream.density[:,0,None]
        mu      = conditions.freestream.dynamic_viscosity[:,0,None]
        a       = conditions.freestream.speed_of_sound[:,0,None]
        T       = conditions.freestream.temperature[:,0,None]
        Vv      = conditions.frames.inertial.velocity_vector
        nu      = mu/rho
        rho_0   = rho
        T_0     = T

        # Number of radial stations and segment control points
        Nr       = len(c)
        ctrl_pts = len(Vv)
        
        # Helpful shorthands
        pi      = np.pi

        # Calculate total blade pitch
        total_blade_pitch = beta_0 + pitch_c

        # Velocity in the rotor frame
        T_body2inertial = conditions.frames.body.transform_to_inertial
        T_inertial2body = orientation_transpose(T_body2inertial)
        V_body          = orientation_product(T_inertial2body,Vv)
        body2thrust     = self.body_to_prop_vel()
        
        T_body2thrust   = orientation_transpose(np.ones_like(T_body2inertial[:])*body2thrust)
        V_thrust        = orientation_product(T_body2thrust,V_body)

        # Check and correct for hover
        V         = V_thrust[:,0,None]
        V[V==0.0] = 1E-6

        # Non-dimensional radial distribution and differential radius
        chi           = r_1d/R
        diff_r        = np.diff(r_1d)
        deltar        = np.zeros(len(r_1d))
        deltar[1:-1]  = diff_r[0:-1]/2 + diff_r[1:]/2
        deltar[0]     = diff_r[0]/2
        deltar[-1]    = diff_r[-1]/2

        # Calculating rotational parameters
        omegar   = np.outer(omega,r_1d)
        n        = omega/(2.*pi)   # Rotations per second

        # 2 dimensional radial distribution non dimensionalized
        chi_2d         = np.tile(chi[:, None],(1,Na))
        chi_2d         = np.repeat(chi_2d[None,:,:], ctrl_pts, axis=0)
        r_dim_2d       = np.tile(r_1d[:, None] ,(1,Na))
        r_dim_2d       = np.repeat(r_dim_2d[None,:,:], ctrl_pts, axis=0)
        c_2d           = np.tile(c[:, None] ,(1,Na))
        c_2d           = np.repeat(c_2d[None,:,:], ctrl_pts, axis=0)

        # Azimuthal distribution of stations (in direction of rotation)
        psi            = np.linspace(0,2*pi,Na+1)[:-1]
        psi_2d         = np.tile(np.atleast_2d(psi),(Nr,1))
        psi_2d         = np.repeat(psi_2d[None, :, :], ctrl_pts, axis=0)

        # apply blade sweep to azimuthal position
        if np.any(np.array([sweep])!=0):
            use_2d_analysis     = True
            sweep_2d            = np.repeat(sweep[:, None], (1,Na))
            sweep_offset_angles = np.tan(sweep_2d/r_dim_2d)
            psi_2d             += sweep_offset_angles

        # Starting with uniform freestream
        ua       = 0
        ut       = 0
        ur       = 0

        # Include velocities introduced by rotor incidence angles
        if (np.any(abs(V_thrust[:,1]) >1e-3) or np.any(abs(V_thrust[:,2]) >1e-3)) and use_2d_analysis:

            # y-component of freestream in the propeller cartesian plane
            Vy  = V_thrust[:,1,None,None]
            Vy  = np.repeat(Vy, Nr,axis=1)
            Vy  = np.repeat(Vy, Na,axis=2)

            # z-component of freestream in the propeller cartesian plane
            Vz  = V_thrust[:,2,None,None]
            Vz  = np.repeat(Vz, Nr,axis=1)
            Vz  = np.repeat(Vz, Na,axis=2)

            # compute resulting radial and tangential velocities in polar frame
            utz =  -Vz*np.sin(psi_2d)
            urz =   Vz*np.cos(psi_2d)
            uty =  -Vy*np.cos(psi_2d)
            ury =   Vy*np.sin(psi_2d)

            ut +=  (utz + uty)  # tangential velocity in direction of rotor rotation
            ur +=  (urz + ury)  # radial velocity (positive toward tip)
            ua +=  np.zeros_like(ut)
            

        # Include external velocities introduced by user
        if nonuniform_freestream:
            use_2d_analysis   = True

            # include additional influences specified at rotor sections, shape=(ctrl_pts,Nr,Na)
            ua += self.external_axial_disc_velocity
            ut += self.external_tangential_disc_velocity
            ur += self.external_radial_disc_velocity

        if use_2d_analysis:
            # make everything 2D with shape (ctrl_pts,Nr,Na)

            # 2-D freestream velocity and omega*r
            V_2d   = V_thrust[:,0,None,None]
            V_2d   = np.repeat(V_2d, Na,axis=2)
            V_2d   = np.repeat(V_2d, Nr,axis=1)
            omegar = (np.repeat(np.outer(omega,r_1d)[:,:,None], Na, axis=2))

            # total velocities
            Ua     = V_2d + ua

            # 2-D blade pitch and radial distributions
            if np.size(pitch_c)>1:
                # control variable is the blade pitch, repeat around azimuth
                beta = np.repeat(total_blade_pitch[:,:,None], Na, axis=2)
            else:
                beta = np.tile(total_blade_pitch[None,:,None],(ctrl_pts,1,Na ))

            r    = np.tile(r_1d[None,:,None], (ctrl_pts, 1, Na))
            c    = np.tile(c[None,:,None], (ctrl_pts, 1, Na))
            deltar = np.tile(deltar[None,:,None], (ctrl_pts, 1, Na))

            # 2-D atmospheric properties
            a   = np.tile(np.atleast_2d(a),(1,Nr))
            a   = np.repeat(a[:, :, None], Na, axis=2)
            nu  = np.tile(np.atleast_2d(nu),(1,Nr))
            nu  = np.repeat(nu[:,  :, None], Na, axis=2)
            rho = np.tile(np.atleast_2d(rho),(1,Nr))
            rho = np.repeat(rho[:,  :, None], Na, axis=2)
            T   = np.tile(np.atleast_2d(T),(1,Nr))
            T   = np.repeat(T[:, :, None], Na, axis=2)

        else:
            # total velocities
            r      = r_1d
            Ua     = np.outer((V + ua),np.ones_like(r))
            beta   = total_blade_pitch

        # Total velocities
        Ut     = omegar - ut
        U      = np.sqrt(Ua*Ua + Ut*Ut + ur*ur)
        
        
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
        wake_inputs.use_2d_analysis       = use_2d_analysis        
        wake_inputs.twist_distribution    = beta
        wake_inputs.chord_distribution    = c
        wake_inputs.radius_distribution   = r
        wake_inputs.speed_of_sounds       = a
        wake_inputs.dynamic_viscosities   = nu
        wake_inputs.azimuthal_distribution = psi_2d

        va, vt, self = self.Wake.evaluate(self,wake_inputs,conditions)
        
        # compute new blade velocities
        Wa   = va + Ua
        Wt   = Ut - vt

        lamdaw, F, _ = compute_inflow_and_tip_loss(r,R, Rh,Wa,Wt,B)

        # Compute aerodynamic forces based on specified input airfoil or surrogate
        Cl, Cdval, alpha, Ma,W = compute_airfoil_aerodynamics(beta,c,r,R,B,F,Wa,Wt,a,nu,a_loc,a_geo,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc,use_2d_analysis)
        
        
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
        epsilon[epsilon==np.inf] = 10.

        # thrust and torque and their derivatives on the blade.
        blade_T_distribution     = rho*(Gamma*(Wt-epsilon*Wa))*deltar
        blade_Q_distribution     = rho*(Gamma*(Wa+epsilon*Wt)*r)*deltar
        blade_dT_dr              = rho*(Gamma*(Wt-epsilon*Wa))
        blade_dQ_dr              = rho*(Gamma*(Wa+epsilon*Wt)*r)


        if use_2d_analysis:
            blade_T_distribution_2d = blade_T_distribution
            blade_Q_distribution_2d = blade_Q_distribution
            blade_dT_dr_2d          = blade_dT_dr
            blade_dQ_dr_2d          = blade_dQ_dr
            blade_Gamma_2d          = Gamma
            alpha_2d                = alpha

            Va_2d = Wa
            Vt_2d = Wt
            Va_avg = np.average(Wa, axis=2)      # averaged around the azimuth
            Vt_avg = np.average(Wt, axis=2)      # averaged around the azimuth

            Va_ind_2d  = va
            Vt_ind_2d  = vt
            Vt_ind_avg = np.average(vt, axis=2)
            Va_ind_avg = np.average(va, axis=2)

            # set 1d blade loadings to be the average:
            blade_T_distribution    = np.mean((blade_T_distribution_2d), axis = 2)
            blade_Q_distribution    = np.mean((blade_Q_distribution_2d), axis = 2)
            blade_dT_dr             = np.mean((blade_dT_dr_2d), axis = 2)
            blade_dQ_dr             = np.mean((blade_dQ_dr_2d), axis = 2)

            # compute the hub force / rotor drag distribution along the blade
            dL_2d    = 0.5*rho*c_2d*Cd*omegar**2*deltar
            dD_2d    = 0.5*rho*c_2d*Cl*omegar**2*deltar

            rotor_drag_distribution = np.mean(dL_2d*np.sin(psi_2d) + dD_2d*np.cos(psi_2d),axis=2)

        else:
            Va_2d   = np.repeat(Wa[ :, :, None], Na, axis=2)
            Vt_2d   = np.repeat(Wt[ :, :, None], Na, axis=2)

            blade_T_distribution_2d  = np.repeat(blade_T_distribution[:, :, None], Na, axis=2)
            blade_Q_distribution_2d  = np.repeat(blade_Q_distribution[:, :, None], Na, axis=2)
            blade_dT_dr_2d           = np.repeat(blade_dT_dr[:, :, None], Na, axis=2)
            blade_dQ_dr_2d           = np.repeat(blade_dQ_dr[:, :, None], Na, axis=2)
            blade_Gamma_2d           = np.repeat(Gamma[ :, :, None], Na, axis=2)
            alpha_2d                 = np.repeat(alpha[ :, :, None], Na, axis=2)

            Vt_avg                  = Wt
            Va_avg                  = Wa
            Vt_ind_avg              = vt
            Va_ind_avg              = va
            Va_ind_2d               = np.repeat(va[ :, :, None], Na, axis=2)
            Vt_ind_2d               = np.repeat(vt[ :, :, None], Na, axis=2)

            # compute the hub force / rotor drag distribution along the blade
            dL    = 0.5*rho*c*Cd*omegar**2*deltar
            dL_2d = np.repeat(dL[:, :, None], Na, axis=2)
            dD    = 0.5*rho*c*Cl*omegar**2*deltar
            dD_2d = np.repeat(dD[:, :, None], Na, axis=2)

            rotor_drag_distribution = np.mean(dL_2d*np.sin(psi_2d) + dD_2d*np.cos(psi_2d),axis=2)

        # forces
        thrust                  = np.atleast_2d((B * np.sum(blade_T_distribution, axis = 1))).T
        torque                  = np.atleast_2d((B * np.sum(blade_Q_distribution, axis = 1))).T
        rotor_drag              = np.atleast_2d((B * np.sum(rotor_drag_distribution, axis=1))).T
        power                   = omega*torque

        # calculate coefficients
        D        = 2*R
        Cq       = torque/(rho_0*(n*n)*(D*D*D*D*D))
        Ct       = thrust/(rho_0*(n*n)*(D*D*D*D))
        Cp       = power/(rho_0*(n*n*n)*(D*D*D*D*D))
        Crd      = rotor_drag/(rho_0*(n*n)*(D*D*D*D))
        etap     = V*thrust/power
        A        = np.pi*(R**2 - self.hub_radius**2)
        FoM      = thrust*np.sqrt(thrust/(2*rho_0*A))    /power  

        # prevent things from breaking
        #Cq[Cq<0]                                               = 0.
        #Ct[Ct<0]                                               = 0.
        #Cp[Cp<0]                                               = 0.
        thrust[conditions.propulsion.throttle[:,0] <=0.0]      = 0.0
        power[conditions.propulsion.throttle[:,0]  <=0.0]      = 0.0
        torque[conditions.propulsion.throttle[:,0]  <=0.0]     = 0.0
        rotor_drag[conditions.propulsion.throttle[:,0]  <=0.0] = 0.0
        thrust[omega<0.0]                                      = -thrust[omega<0.0]
        thrust[omega==0.0]                                     = 0.0
        power[omega==0.0]                                      = 0.0
        torque[omega==0.0]                                     = 0.0
        rotor_drag[omega==0.0]                                 = 0.0
        Ct[omega==0.0]                                         = 0.0
        Cp[omega==0.0]                                         = 0.0
        etap[omega==0.0]                                       = 0.0


        # Make the thrust a 3D vector
        thrust_prop_frame      = np.zeros((ctrl_pts,3))
        thrust_prop_frame[:,0] = thrust[:,0]
        thrust_vector          = orientation_product(orientation_transpose(T_body2thrust),thrust_prop_frame)

        # Assign efficiency to network
        conditions.propulsion.etap = etap


        # Store data
        self.azimuthal_distribution                   = psi
        results_conditions                            = Data
        outputs                                       = results_conditions(
                    number_radial_stations            = Nr,
                    number_azimuthal_stations         = Na,
                    disc_radial_distribution          = r_dim_2d,
                    disc_azimuthal_distribution       = psi_2d,
                    speed_of_sound                    = conditions.freestream.speed_of_sound,
                    density                           = conditions.freestream.density,
                    velocity                          = Vv,
                    blade_tangential_induced_velocity = Vt_ind_avg,
                    blade_axial_induced_velocity      = Va_ind_avg,
                    blade_tangential_velocity         = Vt_avg,
                    blade_axial_velocity              = Va_avg,
                    external_axial_disc_velocity      = self.external_axial_disc_velocity,
                    external_tangential_disc_velocity = self.external_tangential_disc_velocity,
                    external_radial_disc_velocity     = self.external_radial_disc_velocity,
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
                    tip_mach                          = omega * R / conditions.freestream.speed_of_sound,
                    wake_inputs                       = wake_inputs
            )
        self.outputs = outputs

        # DEBUG
        # append sample point to csv (V, Alpha, J, CT, CQ)
        Alpha = np.atleast_2d(self.inputs.y_axis_rotation)
        Vinf = np.atleast_2d(np.linalg.norm(Vv,axis=1)).T
        J = Vinf / (n * D)        
        for i in range(ctrl_pts):
            newRow = [i, Vinf[i][0], Alpha[i][0], J[i][0], outputs.thrust_coefficient[i][0], outputs.torque_coefficient[i][0], thrust[i][0], torque[i][0]]
            with open('/Users/rerha/Desktop/mission_sampled_points.csv', 'a') as file:
                
                writerObj = writer(file)
                writerObj.writerow(newRow)
                file.close()
                
        return thrust_vector, torque, power, Cp, outputs , etap
    
    def spin_surrogate(self, conditions):
        # get force coefficients from lookup table with interpolation
        try:
            #surrogate_data_file = self.surrogate_data_file
            surrogate_data = self.surrogate_data #pd.read_csv(surrogate_data_file)
            #CT_ok3d = self.Kriging_CT
            #CQ_ok3d = self.Kriging_CQ
        except:
            raise("Error: No surrogate data found.")
        
        # unpack
        omega = self.inputs.omega
        n     = omega/(2.*np.pi) 
        D     = 2 * self.tip_radius
        rho   = conditions.freestream.density[:,0,None]
        Vvec  = conditions.frames.inertial.velocity_vector
        cpts  = len(Vvec)
        
        
        
        # compute T, Q for current prop state
        V_sim = np.linalg.norm(Vvec,axis=1)
        A_sim_tilt = self.inputs.y_axis_rotation #np.repeat(self.orientation_euler_angles[1] , cpts)
        J_sim = V_sim / (n.T[0] * D)

        # compute total prop angle to freestream (tilt + pitch)
        try:
            aircraft_pitch  = conditions.aerodynamics.angle_of_attack
            A_sim = A_sim_tilt + aircraft_pitch
        except:
            A_sim = np.atleast_2d(np.repeat(self.orientation_euler_angles[1] , cpts))
            A_sim_tilt = A_sim
                
        # extract surrogate data
        V_sur = surrogate_data.V.to_numpy()
        A_sur = surrogate_data.AlphaP.to_numpy() * Units.deg
        J_sur = surrogate_data.J.to_numpy()    
        CT_sur = surrogate_data.CT.to_numpy()
        CQ_sur = surrogate_data.CQ.to_numpy()
        
        
        thrust = np.zeros_like(V_sim)
        torque = np.zeros_like(V_sim)
        
        # Use Kriging interpolation for (J-Alpha) space with linear interpolation in velocity dimension
        for i, V_sim_i in enumerate(V_sim):
            v_low = V_sur[np.round(V_sur,3) >= np.round(V_sim_i,3)].min()
            v_high = V_sur[np.round(V_sur,3) <= np.round(V_sim_i,3)].max()
        
            vmodel = "linear" #"exponential" "spherical" "linear" "power" "gaussian"
            if v_low==v_high:
                # use exact V_sim_i
                ids = np.where(np.round(V_sur,3) == np.round(v_low,3))
                Kriging_CT = Kriging(A_sur[ids], J_sur[ids], CT_sur[ids], variogram_model=vmodel)
                Kriging_CQ = Kriging(A_sur[ids], J_sur[ids], CQ_sur[ids], variogram_model=vmodel)
        
                CT_k, CT_ss = Kriging_CT.execute("points", A_sim[i], J_sim[i]) #,n_closest_points=20
                thrust[i] = CT_k * (rho[0][0]*(n[i]*n[i])*(D*D*D*D))        
        
                CQ_k, CQ_ss = Kriging_CQ.execute("points", A_sim[i], J_sim[i]) #,n_closest_points=20
                torque[i] = CQ_k * (rho[0][0]*(n[i]*n[i])*(D*D*D*D*D))       
            else:
                # interpolate between the two 
                ids_low = np.where(np.round(V_sur,3) == np.round(v_low,3))
                Kriging_CT_low = Kriging(A_sur[ids_low], J_sur[ids_low], CT_sur[ids_low], variogram_model=vmodel)
                Kriging_CQ_low = Kriging(A_sur[ids_low], J_sur[ids_low], CQ_sur[ids_low], variogram_model=vmodel)
                
                CT_k_l, CT_ss_l = Kriging_CT_low.execute("points", A_sim[i], J_sim[i]) #,n_closest_points=20
                thrust_low = CT_k_l[0] * (rho[0][0]*(n[i]*n[i])*(D*D*D*D))          
                CQ_k_l, CQ_ss_l = Kriging_CQ_low.execute("points", A_sim[i], J_sim[i]) #,n_closest_points=20
                torque_low = CQ_k_l[0] * (rho[0][0]*(n[i]*n[i])*(D*D*D*D*D))                 
    
                ids_high = np.where(np.round(V_sur,4) == np.round(v_high,4))
                Kriging_CT_high = Kriging(A_sur[ids_high], J_sur[ids_high], CT_sur[ids_high], variogram_model=vmodel)
                Kriging_CQ_high = Kriging(A_sur[ids_high], J_sur[ids_high], CQ_sur[ids_high], variogram_model=vmodel)  
                
    
                CT_k_h, CT_ss_h = Kriging_CT_high.execute("points", A_sim[i], J_sim[i]) #,n_closest_points=20
                thrust_high = CT_k_h[0] * (rho[0][0]*(n[i]*n[i])*(D*D*D*D))          
                CQ_k_h, CQ_ss_h = Kriging_CQ_high.execute("points", A_sim[i], J_sim[i]) #,n_closest_points=20
                torque_high = CQ_k_h[0] * (rho[0][0]*(n[i]*n[i])*(D*D*D*D*D))         
                
                Kriging_T_fun = sp.interpolate.interp1d(np.array([v_low, v_high]), np.array([thrust_low[0], thrust_high[0]]))
                thrust[i] = Kriging_T_fun(V_sim_i)
    
                Kriging_Q_fun = sp.interpolate.interp1d(np.array([v_low, v_high]), np.array([torque_low[0], torque_high[0]]))
                torque[i] = Kriging_Q_fun(V_sim_i) 
        
        thrust = np.atleast_2d(thrust).T
        torque = np.atleast_2d(torque).T      
                
        
        power = omega*torque
        
        # Thrust in the rotor frame
        T_body2inertial = conditions.frames.body.transform_to_inertial
        body2thrust     = self.body_to_prop_vel()
        T_body2thrust   = orientation_transpose(np.ones_like(T_body2inertial[:])*body2thrust)
        
        # Make the thrust a 3D vector
        thrust_prop_frame      = np.zeros((cpts,3))
        thrust_prop_frame[:,0] = thrust[:,0]
        thrust_vector          = orientation_product(orientation_transpose(T_body2thrust),thrust_prop_frame)
        
        etap     = np.atleast_2d(V_sim).T*thrust/power
        A        = np.pi*(self.tip_radius**2 - self.hub_radius**2)
        FoM      = thrust*np.sqrt(thrust/(2*rho*A))    /power  
        Cp       = power/(rho*(n*n*n)*(D*D*D*D*D))
        
        results_conditions     = Data
        outputs                = results_conditions( 
            figure_of_merit    = FoM,
            thrust_coefficient = thrust / (rho*(n*n)*(D*D*D*D)),
            torque_coefficient = torque / (rho*(n*n)*(D*D*D*D*D)),
            power_coefficient  = power / (rho*(n*n*n)*(D*D*D*D*D))
        )
        
        ## DEBUG
        ## append sample point to csv (V, Alpha, J, CT, CQ)
        #for i in range(cpts):
            #newRow = [i, V_sim[i], A_sim_tilt[i][0], J_sim[i], outputs.thrust_coefficient[i][0], outputs.torque_coefficient[i][0], thrust[i][0], torque[i][0]]
            #with open('/Users/rerha/Desktop/mission_sampled_points_sur.csv', 'a') as file:
                
                #writerObj = writer(file)
                #writerObj.writerow(newRow)
                #file.close()
        
        return thrust_vector, torque, power, Cp, outputs , etap
    
    
    def vec_to_vel(self):
        """This rotates from the propeller's vehicle frame to the propeller's velocity frame

        Assumptions:
        There are two propeller frames, the propeller vehicle frame and the propeller velocity frame. When propeller
        is axially aligned with the vehicle body:
           - The velocity frame is X out the nose, Z towards the ground, and Y out the right wing
           - The vehicle frame is X towards the tail, Z towards the ceiling, and Y out the right wing

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
        """This rotates from the system's body frame to the propeller's velocity frame

        Assumptions:
        There are two propeller frames, the vehicle frame describing the location and the propeller velocity frame.
        Velocity frame is X out the nose, Z towards the ground, and Y out the right wing
        Vehicle frame is X towards the tail, Z towards the ceiling, and Y out the right wing

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """

        # Go from velocity to vehicle frame
        body_2_vehicle = sp.spatial.transform.Rotation.from_rotvec([0,np.pi,0]).as_matrix()

        # Go from vehicle frame to propeller vehicle frame: rot 1 including the extra body rotation
        cpts       = len(np.atleast_1d(self.inputs.y_axis_rotation))
        rots       = np.array(self.orientation_euler_angles) * 1.
        rots       = np.repeat(rots[None,:], cpts, axis=0)
        rots[:,1] += np.atleast_2d(self.inputs.y_axis_rotation)[:,0]
        
        vehicle_2_prop_vec = sp.spatial.transform.Rotation.from_rotvec(rots).as_matrix()

        # GO from the propeller vehicle frame to the propeller velocity frame: rot 2
        prop_vec_2_prop_vel = self.vec_to_vel()

        # Do all the matrix multiplies
        rot1    = np.matmul(body_2_vehicle,vehicle_2_prop_vec)
        rot_mat = np.matmul(rot1,prop_vec_2_prop_vel)


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
    
    def vehicle_body_to_prop_vel(self):
        """This rotates from the system's body frame to the propeller's velocity frame

        Assumptions:
        There are two propeller frames, the vehicle frame describing the location and the propeller velocity frame.
        Velocity frame is X out the nose, Z towards the ground, and Y out the right wing
        Vehicle frame is X towards the tail, Z towards the ceiling, and Y out the right wing

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """

        # Go from velocity to vehicle frame
        body_2_vehicle = sp.spatial.transform.Rotation.from_rotvec([0,np.pi,0]).as_matrix()

        # Go from vehicle frame to propeller vehicle frame: rot 1 including the extra body rotation
        cpts       = len(np.atleast_1d(self.inputs.y_axis_rotation))
        rots       = np.array(self.orientation_euler_angles) * 1.
        rots       = np.repeat(rots[None,:], cpts, axis=0)
        
        vehicle_2_prop_vec = sp.spatial.transform.Rotation.from_rotvec(rots).as_matrix()

        # GO from the propeller vehicle frame to the propeller velocity frame: rot 2
        prop_vec_2_prop_vel = self.vec_to_vel()

        # Do all the matrix multiplies
        rot1    = np.matmul(body_2_vehicle,vehicle_2_prop_vec)
        rot_mat = np.matmul(rot1,prop_vec_2_prop_vel)


        return rot_mat
    
    def prop_vel_to_vehicle_body(self):
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

        body2propvel = self.vehicle_body_to_prop_vel()

        r = sp.spatial.transform.Rotation.from_matrix(body2propvel)
        r = r.inv()
        rot_mat = r.as_matrix()

        return rot_mat    
    
    def vec_to_prop_body(self):
        return self.prop_vel_to_body()
    
    def vehicle_body_to_prop_body(self):
        """This rotates from the system's body frame to the propeller's velocity frame

        Assumptions:
        There are two propeller frames, the vehicle frame describing the location and the propeller velocity frame.
        Velocity frame is X out the nose, Z towards the ground, and Y out the right wing
        Vehicle frame is X towards the tail, Z towards the ceiling, and Y out the right wing

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """

        # Go from velocity to vehicle frame
        body_2_vehicle = sp.spatial.transform.Rotation.from_rotvec([0,np.pi,0]).as_matrix()

        # Go from vehicle frame to propeller vehicle frame: rot 1 including the extra body rotation
        cpts       = len(np.atleast_1d(self.inputs.y_axis_rotation))
        rots = np.array([0., self.inputs.y_axis_rotation, 0.])
        rots = np.repeat(rots[None,:], cpts, axis=0)
        
        vehicle_2_prop_vec = sp.spatial.transform.Rotation.from_rotvec(rots).as_matrix()

        # GO from the propeller vehicle frame to the propeller velocity frame: rot 2
        prop_vec_2_prop_vel = self.vec_to_vel()

        # Do all the matrix multiplies
        rot1    = np.matmul(body_2_vehicle,vehicle_2_prop_vec)
        rot_mat = np.matmul(rot1,prop_vec_2_prop_vel)


        return rot_mat   
    
    def prop_body_to_vehicle_body(self):
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

        body2propvel = self.vehicle_body_to_prop_body()

        r = sp.spatial.transform.Rotation.from_matrix(body2propvel)
        r = r.inv()
        rot_mat = r.as_matrix()

        return rot_mat        

    def compute_VLM_influence_at_rotor(self, vehicle, conditions):
        """This takes a vehicle with a vortex distribution after VLM analysis and computes
         the VLM-induced velocities at the rotor wake. These external velocities are then 
         applied in the rotor BET / spin function.

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
        #----------------------------------------------------------------------------------------------
        # unpack
        VD  = vehicle.vortex_distribution
        Na  = self.number_azimuthal_stations
        Nr  = len(self.radius_distribution) 
        O   = self.origin[0]
        rot = self.rotation
        #----------------------------------------------------------------------------------------------
                
        # extract location of rotor blade elements from rotor
        r   = self.radius_distribution
        psi = np.linspace(0,2*np.pi,Na+1)[:-1]
        
        r_2d   = np.tile(r[:,None], (1, Na))
        psi_2d = np.tile(psi[None,:],(Nr,1))   
        #----------------------------------------------------------------------------------------------
                
        # convert positions from polar to cartesian (prop frame)
        xE = 0
        yE = r_2d * np.sin(psi_2d)
        zE = r_2d * np.cos(psi_2d)

        #----------------------------------------------------------------------------------------------
                
        # rotate about y-axis (put into vehicle frame)
        rot_mat = self.prop_body_to_vehicle_body()[0]
        VD_temp = copy.deepcopy(VD)
        Positions = np.matmul(rot_mat, np.array([np.ravel(O[0] + xE), np.ravel(O[1] + yE), np.ravel(O[2] + zE) ]))
        VD_temp.XC = Positions[0][:]
        VD_temp.YC = Positions[1][:]
        VD_temp.ZC = Positions[2][:]
        #----------------------------------------------------------------------------------------------
                
        # compute induced velocity at rotor disk elements
        C_mn, _, _, _ = compute_wing_induced_velocity(VD_temp, mach=np.array([0]))
        Vx_inviscid = np.reshape( (C_mn[:,:,:,0]@VD_temp.gamma.T)[0,:,0] , np.shape(r_2d) )
        Vy_inviscid = np.reshape( (C_mn[:,:,:,1]@VD_temp.gamma.T)[0,:,0] , np.shape(r_2d) )
        Vz_inviscid = np.reshape( (C_mn[:,:,:,2]@VD_temp.gamma.T)[0,:,0] , np.shape(r_2d) )
        
        #----------------------------------------------------------------------------------------------
        # Impart the wake deficit from BL of wing if x is behind the wing
        #----------------------------------------------------------------------------------------------
        
        wing_tags = list(vehicle.wings.keys())
        main_wing = wing_tags[0] # assumes main wing is the first in the list        
        x0_wing = vehicle.wings[main_wing].origin[0][0]
        span = vehicle.wings[main_wing].spans.projected
        croot = vehicle.wings[main_wing].chords.root
        ctip = vehicle.wings[main_wing].chords.tip
        rho     = conditions.freestream.density
        mu      = conditions.freestream.dynamic_viscosity 
        Vv      = conditions.freestream.velocity[0][0]
        nu      = (mu/rho)[0][0]
        
        Va_deficit = np.zeros_like(VD_temp.YC)
        # if prop point is behind the wing leading edge
        invalid_ids = np.where(VD_temp.XC <x0_wing)
        
            
        # impart viscous wake to grid points within the span of the wing
        y_inside            = abs(VD_temp.YC)<0.5*span
        chord_distribution  = croot - (croot-ctip)*(abs(VD_temp.YC[y_inside])/(0.5*span))
        
        # Reynolds number developed at x-plane:
        Rex_prop_plane     = Vv*(VD_temp.XC[y_inside]-x0_wing)/nu
        
        # boundary layer development distance
        x_dev      = (VD_temp.XC[y_inside]-x0_wing) * np.ones_like(chord_distribution)
        
        # For turbulent flow
        theta_turb  = 0.036*x_dev/(Rex_prop_plane**(1/5))
        x_theta     = (x_dev-chord_distribution)/theta_turb

        # axial velocity deficit due to turbulent BL from the wing (correlation from Ramaprian et al.)
        W0  = Vv/np.sqrt(4*np.pi*0.032*x_theta)
        b   = 2*theta_turb*np.sqrt(16*0.032*np.log(2)*x_theta)
        Va_deficit[y_inside] = W0*np.exp(-4*np.log(2)*(abs(VD_temp.ZC[y_inside])/b)**2)
            
        Va_deficit[invalid_ids] = 0
        
        Vx = Vx_inviscid - np.reshape(Va_deficit, np.shape(Vx_inviscid))
        Vy = Vy_inviscid
        Vz = Vz_inviscid
        
        #----------------------------------------------------------------------------------------------
        # convert velocities from vehicle to prop frame
        rot_mat2 = self.vehicle_body_to_prop_body()[0]
        VC_p = np.matmul(rot_mat2, np.array([np.ravel(Vx), np.ravel(Vy), np.ravel(Vz)]))
        Vx_p = np.reshape(VC_p[0][:], np.shape(Vx))
        Vy_p = np.reshape(VC_p[1][:], np.shape(Vx))
        Vz_p = np.reshape(VC_p[2][:], np.shape(Vx))
        
        Va_p = Vx_p
        Vt_p = -rot*(Vy_p*(np.cos(psi_2d)) - Vz_p*(np.sin(psi_2d)) ) 
        Vr_p = -rot*(Vy_p*(np.sin(psi_2d)) + Vz_p*(np.cos(psi_2d)) ) 
        
        #----------------------------------------------------------------------------------------------
        # append to rotor, shape=(ctrl_pts,Nr,Na)
        self.external_axial_disc_velocity = Va_p[None,:,:]
        self.external_tangential_disc_velocity = Vt_p[None,:,:]
        self.external_radial_disc_velocity = Vr_p[None,:,:]
        self.nonuniform_freestream = True
        

        return
        