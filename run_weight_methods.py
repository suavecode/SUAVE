# run_weight_methods.py
#
# Created:  Aug 2014, SUAVE Team
# Modified: Jan 2017, T. MacDonald
#           Aug 2017, E. Botero

""" File to run all weight estimation methods
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Input_Output.Results import weight_breakdown_to_csv, \
    plot_weight_comparison
import pylab as plt
import sys

sys.path.append('regression/scripts/Vehicles')
from Boeing_737 import vehicle_setup, configs_setup


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    configs, analyses = full_setup()

    configs.finalize()
    analyses.finalize()

    # weight analysis
    weights = analyses.configs.base.weights
    file = 'weight_737.csv'
    analysisType = "SUAVE"
    breakdown = weights.evaluate(method=analysisType)
    weight_breakdown_to_csv(configs.base, filename=file, header=analysisType)

    analysisType = "FLOPS Simple"
    breakdown = weights.evaluate(method=analysisType)
    weight_breakdown_to_csv(configs.base, filename=file, header=analysisType)

    analysisType = "FLOPS Complex"
    breakdown = weights.evaluate(method=analysisType)
    weight_breakdown_to_csv(configs.base, filename=file, header=analysisType)

    analysisType = "New SUAVE"
    breakdown = weights.evaluate(method=analysisType)
    weight_breakdown_to_csv(configs.base, filename=file, header=analysisType)

    analysisType = "Raymer"
    breakdown = weights.evaluate(method=analysisType)
    weight_breakdown_to_csv(configs.base, filename=file, header=analysisType)

    plot_weight_comparison([file], ['Boeing 737'], 'lbs')
    plt.show()
    
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():
    # vehicle data
    vehicle = vehicle_setup()
    configs = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission = mission_setup(configs_analyses)
    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs = configs_analyses
    analyses.missions = missions_analyses

    return configs, analyses


# ----------------------------------------------------------------------
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------

def analyses_setup(configs):
    analyses = SUAVE.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag, config in configs.items():
        analysis = base_analysis(config)
        analyses[tag] = analysis

    return analyses


def base_analysis(vehicle):
    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------
    analyses = SUAVE.Analyses.Vehicle()

    # ------------------------------------------------------------------
    #  Basic Geometry Relations
    sizing = SUAVE.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)

    # ------------------------------------------------------------------
    #  Weights
    weights = SUAVE.Analyses.Weights.Weights_Tube_Wing()
    weights.vehicle = vehicle
    analyses.append(weights)

    return analyses


def simple_sizing(configs):
    base = configs.base
    base.pull_base()

    # zero fuel weight
    base.mass_properties.max_zero_fuel = 0.9 * base.mass_properties.max_takeoff

    # wing areas
    for wing in base.wings:
        wing.areas.wetted = 2.0 * wing.areas.reference
        wing.areas.exposed = 0.8 * wing.areas.wetted
        wing.areas.affected = 0.6 * wing.areas.wetted

    # fuselage seats
    base.fuselages['fuselage'].number_coach_seats = base.passengers

    # diff the new data
    base.store_diff()

    # done!
    return


# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------

def mission_setup(analyses):
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'the_mission'

    # airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude = 0.0 * Units.ft
    airport.delta_isa = 0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport = airport

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment()

    # ------------------------------------------------------------------
    #   First Climb Segment
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"

    segment.analyses.extend(analyses.base)

    ones_row = segment.state.ones_row
    segment.state.unknowns.body_angle = ones_row(1) * 7. * Units.deg

    segment.altitude_start = 0.0 * Units.km
    segment.altitude_end = 3.05 * Units.km
    segment.air_speed = 128.6 * Units['m/s']
    segment.climb_rate = 20.32 * Units['m/s']

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Climb Segment
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_2"

    segment.analyses.extend(analyses.base)

    segment.altitude_end = 4.57 * Units.km
    segment.air_speed = 205.8 * Units['m/s']
    segment.climb_rate = 10.16 * Units['m/s']

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Third Climb Segment: linear Mach
    # ------------------------------------------------------------------

    segment = Segments.Climb.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "climb_3"

    segment.analyses.extend(analyses.base)

    segment.altitude_end = 7.60 * Units.km
    segment.mach_start = 0.64
    segment.mach_end = 1.0
    segment.climb_rate = 5.05 * Units['m/s']

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Fourth Climb Segment: linear Mach
    # ------------------------------------------------------------------

    segment = Segments.Climb.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "climb_4"

    segment.analyses.extend(analyses.base)

    segment.altitude_end = 15.24 * Units.km
    segment.mach_start = 1.0
    segment.mach_end = 2.02
    segment.climb_rate = 5.08 * Units['m/s']

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Fourth Climb Segment
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Mach_Constant_Rate(base_segment)
    segment.tag = "climb_5"

    segment.analyses.extend(analyses.base)

    segment.altitude_end = 18.288 * Units.km
    segment.mach_number = 2.02
    segment.climb_rate = 0.65 * Units['m/s']

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Cruise Segment
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Mach_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    segment.analyses.extend(analyses.base)

    segment.mach = 2.02
    segment.distance = 2000.0 * Units.km

    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   First Descent Segment
    # ------------------------------------------------------------------

    segment = Segments.Descent.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "descent_1"

    segment.analyses.extend(analyses.base)

    segment.altitude_end = 6.8 * Units.km
    segment.mach_start = 2.02
    segment.mach_end = 1.0
    segment.descent_rate = 5.0 * Units['m/s']

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Descent Segment
    # ------------------------------------------------------------------

    segment = Segments.Descent.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "descent_2"

    segment.analyses.extend(analyses.base)

    segment.altitude_end = 3.0 * Units.km
    segment.mach_start = 1.0
    segment.mach_end = 0.65
    segment.descent_rate = 5.0 * Units['m/s']

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Third Descent Segment
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_3"

    segment.analyses.extend(analyses.base)

    segment.altitude_end = 0.0 * Units.km
    segment.air_speed = 130.0 * Units['m/s']
    segment.descent_rate = 5.0 * Units['m/s']

    # append to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Mission definition complete
    # ------------------------------------------------------------------

    return mission


def missions_setup(base_mission):
    # the mission container
    missions = SUAVE.Analyses.Mission.Mission.Container()

    # ------------------------------------------------------------------
    #   Base Mission
    # ------------------------------------------------------------------

    missions.base = base_mission

    # done!
    return missions


if __name__ == '__main__':
    main()
