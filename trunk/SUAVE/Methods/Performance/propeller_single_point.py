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
import time
# ------------------------------------------------------------------------------
#   Propeller Single Point
# ------------------------------------------------------------------------------

## @ingroup Methods-Performance
def propeller_single_point(prop,
                           pitch,
                           tilt,
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
    ctrl_pts = len(omega)
    
    conditions = prop.set_conditions_single_point(pitch, tilt, omega, speed, ctrl_pts, altitude, delta_isa)

    # Initialize and Run Propeller BEVW
    F, Q, P, Cp, outputs, etap = prop.spin(conditions)
    
    va_ind_BEVW         = outputs.disc_axial_induced_velocity[0, :, 0]
    vt_ind_BEVW         = outputs.disc_tangential_induced_velocity[0, :, 0]
    r_BEVW              = outputs.disc_radial_distribution[0, :, 0]
    T_distribution_BEVW = outputs.disc_thrust_distribution[0, :, 0]
    vt_BEVW             = outputs.disc_tangential_velocity[0, :, 0]
    va_BEVW             = outputs.disc_axial_velocity[0, :, 0]
    Q_distribution_BEVW = outputs.disc_torque_distribution[0, :, 0]

    if print_results:
        print('Thrust Coefficient:    {} N'.format(outputs.thrust_coefficient[0][0]))
        print('Torque Coefficient:    {} N-m'.format(outputs.torque_coefficient[0][0]))
        print('Power Coefficient:     {} W'.format(outputs.power_coefficient[0][0]))
        print('Prop Efficiency: {}'.format(etap[0][0]))
        print('Prop FOM: {}'.format(outputs.figure_of_merit[0][0]))

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

    return prop, results, conditions