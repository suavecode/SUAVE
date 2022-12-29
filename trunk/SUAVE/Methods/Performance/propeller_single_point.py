## @ingroup Methods-Performance
# propeller_single_point.py
#
# Created:  Jan 2021, J. Smart
# Modified: Feb 2022, R. Erhard
#           Jun 2022, R. Erhard

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

import SUAVE

from SUAVE.Core import Data

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------------------
#   Propeller Single Point
# ------------------------------------------------------------------------------

## @ingroup Methods-Performance
def propeller_single_point(prop,
                           pitch,
                           omega,
                           altitude,
                           delta_isa,
                           speed,
                           analyses=None,
                           plots=False,
                           print_results=False):
    """propeller_single_point(prop,
                              pitch,
                              omega,
                              altitude,
                              delta_isa,
                              speed,
                              analyses=None,
                              plots=False,
                              print_results=False):

        Uses SUAVE's BEVW propeller model to evaluate propeller performance at a
        single altitude, pitch command, and angular velocity. Can be used indep-
        endently, or as part of creation of a propller maps or flight envelopes.

        Sources:
        N/A

        Assumptions:

        Assumes use of Battery Propeller Energy Network, All Assumptions of
        the BEVW model.

        Inputs:

            prop                 SUAVE Propeller Data Structure
            pitch                Propeller Pitch/Collective                    [User Set]
            omega                Test Angular Velocity                         [User Set]
            altitude             Test Altitude                                 [User Set]
            delta_isa            Atmosphere Temp Offset                        [K]
            speed                Propeller Intake Speed                        [User Set]
            HFW                  Flag for use of helical fixed wake for rotor  [Boolean]
            analyses             SUAVE Analyses Structure
                .atmosphere      SUAVE Atmosphere Analysis Object
            plots                Flag for Plot Generation                      [Boolean]
            print_results        Flag for Terminal Output                      [Boolean]

        Outputs:

            results                             SUAVE Data Object
                .thrust                         BEVW Thrust Prediction      [N]
                .torque                         BEVW Torque Prediction      [N-m]
                .power                          BEVW Power Prediction       [W]
                .power_coefficient              BEVW Cp Prediction          [Unitless]
                .efficiency                     BEVW Efficiency Prediction  [Unitless]
                .induced_axial_velocity         BEVW Ind. V_a Prediction    [m/s]
                .induced_tangential_velocity    BEVW Ind. V_tPrediction     [m/s]
                .radial_distribution            BEVW Radial Stations        [m]
                .thrust_distribution            BEVW T Dist. Prediction     [N/m]
                .torque_distribution            BEVW Q Dist. Prediction     [(N-m)/m]
                .tangential_velocity            BEVW V_t Prediction         [m/s]
                .axial_velocity                 BEVW V_a Prediction         [m/s]
    """
    # Set atmosphere
    if analyses==None:
        # setup standard US 1976 atmosphere
        analyses   = SUAVE.Analyses.Vehicle()
        atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        analyses.append(atmosphere)           
        
    # Unpack Inputs
    prop.inputs.pitch_command   = pitch

    atmo_data           = analyses.atmosphere.compute_values(altitude, delta_isa)
    T                   = atmo_data.temperature
    a                   = atmo_data.speed_of_sound
    density             = atmo_data.density
    dynamic_viscosity   = atmo_data.dynamic_viscosity

    # Setup Pseudo-Mission for Prop Evaluation
    ctrl_pts = 1
    prop.inputs.omega                               = np.ones((ctrl_pts, 1)) * omega
    conditions                                      = SUAVE.Analyses.Mission.Segments.Conditions.Conditions()
    conditions.freestream                           = Data()
    conditions.propulsion                           = Data()
    conditions.noise                                = Data()
    conditions.noise.sources                        = Data()
    conditions.noise.sources.propellers             = Data()
    conditions.frames                               = Data()
    conditions.frames.inertial                      = Data()
    conditions.frames.body                          = Data()  
    conditions.frames.planet                        = Data()    
    conditions.freestream.density                   = np.ones((ctrl_pts, 1)) * density
    conditions.freestream.dynamic_viscosity         = np.ones((ctrl_pts, 1)) * dynamic_viscosity
    conditions.freestream.speed_of_sound            = np.ones((ctrl_pts, 1)) * a
    conditions.freestream.temperature               = np.ones((ctrl_pts, 1)) * T
    conditions.freestream.mach_number               = speed / a
    velocity_vector                                 = np.array([[speed, 0., 0.]])
    conditions.propulsion.throttle                  = np.ones((ctrl_pts, 1)) * 1.
    conditions.frames.inertial.velocity_vector      = np.tile(velocity_vector, (ctrl_pts, 1))
    conditions.frames.body.transform_to_inertial    = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]])
    conditions.frames.planet.true_course_rotation   = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]])  

    # Run Propeller BEVW
    F, Q, P, Cp, outputs, etap = prop.spin(conditions)
        
    va_ind_BEVW         = outputs.disc_axial_induced_velocity[0, :, 0]
    vt_ind_BEVW         = outputs.disc_tangential_induced_velocity[0, :, 0]
    r_BEVW              = outputs.disc_radial_distribution[0, :, 0]
    T_distribution_BEVW = outputs.disc_thrust_distribution[0, :, 0]
    vt_BEVW             = outputs.disc_tangential_velocity[0, :, 0]
    va_BEVW             = outputs.disc_axial_velocity[0, :, 0]
    Q_distribution_BEVW = outputs.disc_torque_distribution[0, :, 0]

    if print_results:
        print('Total Thrust:    {} N'.format(F[0][0]))
        print('Total Torque:    {} N-m'.format(Q[0][0]))
        print('Total Power:     {} W'.format(P[0][0]))
        print('Prop Efficiency: {}'.format(etap[0][0]))

    # ----------------------------------------------------------------------------
    # 2D - Plots  Plots
    # ----------------------------------------------------------------------------

    if plots:
        plt.figure(1)
        plt.plot(r_BEVW, va_BEVW, 'ro-', label='axial BEVW')
        plt.plot(r_BEVW, vt_BEVW, 'bo-', label='tangential BEVW')
        plt.xlabel('Radial Location')
        plt.ylabel('Velocity')
        plt.legend(loc='lower right')

        plt.figure(2)
        plt.plot(r_BEVW, T_distribution_BEVW, 'ro-')
        plt.xlabel('Radial Location')
        plt.ylabel('Thrust, N')

        plt.figure(3)
        plt.plot(r_BEVW, Q_distribution_BEVW, 'ro-')
        plt.xlabel('Radial Location')
        plt.ylabel('Torque, N-m')

        plt.figure(4)
        plt.plot(r_BEVW, va_ind_BEVW, 'ro-', label='Axial')
        plt.plot(r_BEVW, vt_ind_BEVW, 'bo-', label='Tangential')
        plt.xlabel('Radial Location')
        plt.ylabel('Induced Velocity') 

    # Pack Results

    results                             = Data()
    results.thrust                      = F[0][0]
    results.torque                      = Q[0][0]
    results.power                       = P[0][0]
    results.power_coefficient           = Cp[0][0]
    results.efficiency                  = etap[0][0]
    results.induced_axial_velocity      = va_ind_BEVW
    results.induced_tangential_velocity = vt_ind_BEVW
    results.radial_distribution         = r_BEVW
    results.thrust_distribution         = T_distribution_BEVW
    results.torque_distribution         = Q_distribution_BEVW
    results.tangential_velocity         = vt_BEVW
    results.axial_velocity              = va_BEVW
    results.outputs                     = outputs

    return prop, results