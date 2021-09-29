## @ingroup Methods-Performance
# propeller_single_point.py
#
# Created: Jan 2021, J. Smart
# Modified:

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

import SUAVE

from SUAVE.Core import Units, Data

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------------------
#   Propeller Single Point
# ------------------------------------------------------------------------------

## @ingroup Methods-Performance
def propeller_single_point(energy_network,
                           analyses,
                           pitch,
                           omega,
                           altitude,
                           delta_isa,
                           speed,
                           HFW=False,
                           plots=False,
                           print_results=False):
    """propeller_single_point(energy_network,
                              analyses,
                              pitch,
                              omega,
                              altitude,
                              delta_isa,
                              speed,
                              plots=False,
                              print_results=False):

        Uses SUAVE's BEMT propeller model to evaluate propeller performance at a
        single altitude, pitch command, and angular velocity. Can be used indep-
        endently, or as part of creation of a propller maps or flight envelopes.

        Sources:
        N/A

        Assumptions:

        Assumes use of Battery Propeller Energy Network, All Assumptions of
        the BEMT model.

        Inputs:

            energy_network       SUAVE Energy Network
                .propeller       SUAVE Propeller Data Structure

            analyses             SUAVE Analyses Structure
                .atmosphere      SUAVE Atmosphere Analysis Object

            pitch                Propeller Pitch/Collective                    [User Set]
            omega                Test Angular Velocity                         [User Set]
            altitude             Test Altitude                                 [User Set]
            delta_isa            Atmosphere Temp Offset                        [K]
            speed                Propeller Intake Speed                        [User Set]
            HFW                  Flag for use of helical fixed wake for rotor  [Boolean]
            plots                Flag for Plot Generation                      [Boolean]
            print_results        Flag for Terminal Output                      [Boolean]

        Outputs:

            results                             SUAVE Data Object
                .thrust                         BEMT Thrust Prediction      [N]
                .torque                         BEMT Torque Prediction      [N-m]
                .power                          BEMT Power Prediction       [W]
                .power_coefficient              BEMT Cp Prediction          [Unitless]
                .efficiency                     BEMT Efficiency Prediction  [Unitless]
                .induced_axial_velocity         BEMT Ind. V_a Prediction    [m/s]
                .induced_tangential_velocity    BEMT Ind. V_tPrediction     [m/s]
                .radial_distribution            BEMT Radial Stations        [m]
                .thrust_distribution            BEMT T Dist. Prediction     [N/m]
                .torque_distribution            BEMT Q Dist. Prediction     [(N-m)/m]
                .tangential_velocity            BEMT V_t Prediction         [m/s]
                .axial_velocity                 BEMT V_a Prediction         [m/s]
    """
    # Check if the propellers are identical
    if not energy_network.identical_propellers:
        assert('This script only works with identical propellers')
    

    # Unpack Inputs
    prop_key                    = list(energy_network.propellers.keys())[0]
    prop                        = energy_network.propellers[prop_key]
    prop.inputs.pitch_command   = pitch
    energy_network.propeller    = prop

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
    conditions.freestream.density                   = np.ones((ctrl_pts, 1)) * density
    conditions.freestream.dynamic_viscosity         = np.ones((ctrl_pts, 1)) * dynamic_viscosity
    conditions.freestream.speed_of_sound            = np.ones((ctrl_pts, 1)) * a
    conditions.freestream.temperature               = np.ones((ctrl_pts, 1)) * T
    velocity_vector                                 = np.array([[speed, 0., 0.]])
    conditions.propulsion.throttle                  = np.ones((ctrl_pts, 1)) * 1.
    conditions.frames.inertial.velocity_vector      = np.tile(velocity_vector, (ctrl_pts, 1))
    conditions.frames.body.transform_to_inertial    = np.array([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])

    # Run Propeller BEMT
    if HFW:
        F, Q, P, Cp, outputs, etap = prop.spin_HFW(conditions)
    else:
        F, Q, P, Cp, outputs, etap = prop.spin(conditions)
        
    va_ind_BEMT         = outputs.disc_axial_induced_velocity[0, :, 0]
    vt_ind_BEMT         = outputs.disc_tangential_induced_velocity[0, :, 0]
    r_BEMT              = outputs.disc_radial_distribution[0, :, 0]
    T_distribution_BEMT = outputs.disc_thrust_distribution[0, :, 0]
    vt_BEMT             = outputs.disc_tangential_velocity[0, :, 0]
    va_BEMT             = outputs.disc_axial_velocity[0, :, 0]
    Q_distribution_BEMT = outputs.disc_torque_distribution[0, :, 0]

    if print_results:
        print('Total Thrust:    {} N'.format(F[0][0]))
        print('Total Torque:    {} N-m'.format(Q[0][0]))
        print('Total Power:     {} W'.format(P[0][0]))
        print('Prop Efficiency: {}'.format(etap[0][0]))

    # ----------------------------------------------------------------------------
    # 2D - Plots  Plots
    # ----------------------------------------------------------------------------

    if plots:
        fig = plt.figure(1)
        plt.plot(r_BEMT, va_BEMT, 'ro-', label='axial BEMT')
        plt.plot(r_BEMT, vt_BEMT, 'bo-', label='tangential BEMT')
        plt.xlabel('Radial Location')
        plt.ylabel('Velocity')
        plt.legend(loc='lower right')

        fig = plt.figure(2)
        plt.plot(r_BEMT, T_distribution_BEMT, 'ro-')
        plt.xlabel('Radial Location')
        plt.ylabel('Thrust, N')

        fig = plt.figure(3)
        plt.plot(r_BEMT, Q_distribution_BEMT, 'ro-')
        plt.xlabel('Radial Location')
        plt.ylabel('Torque, N-m')

        fig = plt.figure(4)
        plt.plot(r_BEMT, va_ind_BEMT, 'ro-', label='Axial')
        plt.plot(r_BEMT, vt_ind_BEMT, 'bo-', label='Tangential')
        plt.xlabel('Radial Location')
        plt.ylabel('Induced Velocity')

        plt.show()

    # Pack Results

    results                             = Data()
    results.thrust                      = F[0][0]
    results.torque                      = Q[0][0]
    results.power                       = P[0][0]
    results.power_coefficient           = Cp[0][0]
    results.efficiency                  = etap[0][0]
    results.induced_axial_velocity      = va_ind_BEMT
    results.induced_tangential_velocity = vt_ind_BEMT
    results.radial_distribution         = r_BEMT
    results.thrust_distribution         = T_distribution_BEMT
    results.torque_distribution         = Q_distribution_BEMT
    results.tangential_velocity         = vt_BEMT
    results.axial_velocity              = va_BEMT
    results.outputs                     = outputs

    return results