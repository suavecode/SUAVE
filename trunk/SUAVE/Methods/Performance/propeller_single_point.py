## @ingroup Methods-Performance
# propeller_single_point.py
#
# Created: Jan 2021, J. Smart
# Modified:

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Units, Data

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------------------
#   Propeller Single Point
# ------------------------------------------------------------------------------

def propeller_single_point(energy_network,
                           atmosphere,
                           pitch,
                           omega,
                           altitude,
                           delta_isa,
                           speed,
                           plots=False,
                           print_results=False):
    '''
    TODO: Add docstring
    '''

    # Unpack Inputs

    prop                        = energy_network.prop
    prop.pitch_command          = pitch
    energy_network.propeller    = prop

    atmo_data           = atmosphere.compute_values(altitude, delta_isa)
    T                   = atmo_data.temperature
    a                   = atmo_data.speed_of_sound
    density             = atmo_data.density
    dynamic_viscosity   = atmo_data.dynamic_viscosity

    # Setup Pseudo-Mission for Prop Evaluation

    ctrl_pts = 3
    prop.inputs.omega                               = np.ones((ctrl_pts, 1)) * omega * Units.rpm
    conditions                                      = Data()
    conditions.freestream                           = Data()
    conditions.propulsion                           = Data()
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

    F, Q, P, Cp, outputs, etap = prop.spin(conditions)
    va_ind_BEMT         = outputs.disc_axial_induced_velocity[0, 0, :]
    vt_ind_BEMT         = outputs.disc_tangential_induced_velocity[0, 0, :]
    r_BEMT              = outputs.disc_radial_distribution[0, 0, :]
    T_distribution_BEMT = outputs.disc_thrust_distribution[0]
    vt_BEMT             = outputs.disc_tangential_velocity[0, 0, :]
    va_BEMT             = outputs.disc_axial_velocity[0, 0, :]
    Q_distribution_BEMT = outputs.disc_torque_distribution[0]

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

    return results