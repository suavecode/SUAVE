## @ingroup Methods-Performance
# V_n_diagram.py
#
# Created: Dec 2020, J. Smart
# Modified:

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from SUAVE.Core import Units, Data
from SUAVE.Analyses.Mission.Segments.Conditions import Aerodynamics, Numerics
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import  compute_max_lift_coeff

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Compute V-h Diagram
#------------------------------------------------------------------------------

def V_h_diagram(vehicle,
                analyses,
                load_factor = 1.,
                grid_points = 50.,
                altitude_ceiling = 5e4 * Units.ft,
                max_speed = 500 * Units['m/s']
                ):

    """

    Computes excess power contour for a given aircraft load factor (L/W)
    and excess power, plotted over speed and altitude.


    Source:

        Raymer, D. "Aircraft Design: A Conceptual Approach", 6th Edition

    Inputs:

        vehicle
            .propulsors.battery_propeller
                .design_thrust
                .number_of_engines

        analyses
            .atmosphere

        load_factor

        excess_power

        grid_points

        altitude_ceiling

        supersonic

        display_plot

    Outputs:

        excess_power

    Properties Used:


    """

    # Unpack Inputs

    atmo    = analyses.atmosphere
    T       = (vehicle.propulsors.battery_propeller.propeller.design_thrust
                * vehicle.propulsors.battery_propeller.number_of_engines)
    W       = vehicle.mass_properties.max_takeoff
    S       = vehicle.reference_area

    # Specify Altitude Range

    alt_range       = np.linspace(0., altitude_ceiling, num=grid_points, endpoint=True)

    # Specify Airspeed Range

    speed_range     = np.linspace(0., max_speed, num = grid_points, endpoint=False)

    # Initialize Excess Power and Climb Rate

    excess_power = np.zeros((grid_points, grid_points))
    climb_rate = np.zeros((grid_points, grid_points))


    for alt_idx in range(grid_points):

        atmo_data = atmo.compute_values(alt_range[alt_idx])

        for speed_idx in range(grid_points):

            V = speed_range[speed_idx]

            excess_power[speed_idx, alt_idx] = V[T / W - V * 0.02 / (W / S) - load_factor ** 2 * (K / V) * (W / S)]
            climb_rate[speed_idx, alt_idx] = (P-D*V)/w

    mach_space, alt_space = np.meshgrid(speed_range, alt_range)

    if display_plot:

        plt.contour(mach_space, alt_space, excess_power, levels = excess_power_contours)
        plt.xlabel('Mach No.')
        plt.ylabel('Altitude (ft.)')
        plt.title('Excess Power')
        plt.show()

    return excess_power, climb_rate



