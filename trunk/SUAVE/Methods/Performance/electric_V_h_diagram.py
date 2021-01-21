## @ingroup Methods-Performance
# electric_V_h_diagram.py
#
# Created: Jan 2021, J. Smart
# Modified:

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from SUAVE.Core import Units, Data

from SUAVE.Methods.Performance.propeller_single_point import propeller_single_point

import numpy as np
#------------------------------------------------------------------------------
# Flight Envelope Function
#------------------------------------------------------------------------------

def electric_V_h_diagram(vehicle,
                         analyses,
                         grid_points = 20.,
                         altitude_ceiling = 2e4 * Units.ft,
                         max_speed = 130 * Units['m/s'],
                         test_omega = 800. * Units.rpm,
                         display_plot = True,
                         climb_rate_contours = [0.]
                         ):
    '''
    TODO: Add docstring
    '''

    # Unpack Inputs

    g       = analyses.atmosphere.planet.sea_level_gravity
    W       = vehicle.mass_properties.takeoff * g
    S       = vehicle.reference_area

    # Single Point Mission for Drag Determination

    def mini_mission(altitude, speed):

        mission = SUAVE.Analyses.Mission.Sequential_Segments()
        mission.tag = 'the_mission'

        segment = SUAVE.Analyses.Mission.Segments.Single_Point.Set_Speed_Set_Altitude_No_Propulsion()
        segment.tag = 'single_point'
        segment.analyses.extend(analyses)
        segment.altitude = altitude
        segment.air_speed = speed

        mission.append_segment(segment)

        return mission

    # Specify Altitude and Speed Sample Points

    alt_range       = np.linspace(0., altitude_ceiling, num=grid_points, endpoint=True)
    speed_range     = np.linspace(0., max_speed, num = grid_points, endpoint=False)

    # Initialize Climb Rate Grid

    climb_rate = np.zeros((grid_points, grid_points))

    # Loop Through Altitude and Speed Gridpoints

    for alt_idx in range(grid_points):

        altitude    = alt_range[alt_idx]
        atmo_data   = analyses.atmosphere.compute_values(altitude, 15.)
        rho         = atmo_data.density
        Vs          = np.sqrt(2*W/(rho*S*1.4)) # Determine Vehicle Stall Speed

        for speed_idx in range(grid_points):

            V = speed_range[speed_idx]

            if V > Vs: # Only Bother Calculating if Vehicle is Above Stall Speed

                # Determine Vehicle Drag at Altitude and Speed

                mission = mini_mission(altitude, V)
                results = mission.evaluate()

                D = -results.segments.single_point.conditions.frames.wind.drag_force_vector[0][0]

                # Determine Propeller Power at Altitude and Speed

                P = propeller_single_point(vehicle.propulsors.battery_propeller,
                                           analyses.atmosphere,
                                           test_omega,
                                           delta_isa=0.,
                                           speed=V).power

                # Check if Propeller Power Exceeds Max Battery Power, Switch to Max Battery Power if So

                P = np.min([P, vehicle.propulsors.battery_propeller.battery.max_power])

                # Determine Climb Rate (ref. Raymer)

                cr = 1/W * (P - D*V)

                # If Climb Rate is Negative, Replace with 0 for Easy Contour-Finding

                climb_rate[speed_idx, alt_idx] = np.max([0., cr])


    climb_rate  = climb_rate / Units['ft/min']

    if display_plot:

        # Get Speed and Altitude to Agree with Climb Rate Dimensions

        speed_space, alt_space  = np.meshgrid(speed_range, alt_range)
        speed_space             = np.transpose(speed_space)
        alt_space               = np.transpose(alt_space) / Units.ft

        # Make Contour Plot of Climb Rates

        CS = plt.contour(speed_space, alt_space, climb_rate, levels = climb_rate_contours)
        plt.xlabel('Airspeed (m/s)')
        plt.ylabel('Altitude (ft)')
        plt.title('Climb Rate (ft/min)')
        plt.clabel(CS)

        plt.show()

    return climb_rate