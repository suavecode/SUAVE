# electric_payload_range.py
#
# Created: Jan 2021, J. Smart
# Modified:

#-------------------------------------------------------------------------------
# Imports
#_______________________________________________________________________________

import SUAVE

from SUAVE.Core import Units, Data
from SUAVE.Methods.Performance.electric_payload_range import electric_payload_range

import numpy as np

import sys
sys.path.append('../Vehicles')

from Stopped_Rotor import vehicle_setup

#-------------------------------------------------------------------------------
# Test Function
#-------------------------------------------------------------------------------

def main():

    vehicle     = vehicle_setup()
    analyses    = base_analysis(vehicle)
    mission     = mission_setup(vehicle, analyses)

    analyses.mission = mission

    analyses.finalize()

    payload_range = electric_payload_range(vehicle, mission, 'cruise', display_plot=True)

    payload_range_r = [     0.,         100763.78024757, 107079.32850161]

    assert (np.abs(payload_range.range[1] - payload_range_r[1]) / payload_range_r[1] < 1e-6), "Payload Range Regression Failed at Max Payload Test"
    assert (np.abs(payload_range.range[2] - payload_range_r[2]) / payload_range_r[2] < 1e-6), "Payload Range Regression Failed at Ferry Range Test"

    return

#-------------------------------------------------------------------------------
# Helper Functions
#-------------------------------------------------------------------------------

def mission_setup(vehicle, analyses):
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Variable_Range_Cruise.Given_State_of_Charge()
    mission.tag = 'the_mission'

    # the cruise tag to vary cruise distance
    mission.cruise_tag = 'cruise'
    mission.target_state_of_charge = 0.5

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment()
    ones_row = base_segment.state.ones_row
    base_segment.state.numerics.number_control_points = 4
    base_segment.process.iterate.conditions.stability = SUAVE.Methods.skip
    base_segment.process.finalize.post_process.stability = SUAVE.Methods.skip
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position = SUAVE.Methods.skip


    # ------------------------------------------------------------------
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    segment.analyses.extend(analyses)

    segment.altitude = 1000.0 * Units.ft
    segment.air_speed = 110. * Units['mph']
    segment.distance = 60. * Units.miles
    segment.battery_energy = vehicle.networks.lift_cruise.battery.max_energy
    segment.state.unknowns.throttle = 0.80 * ones_row(1)
    segment = vehicle.networks.lift_cruise.add_cruise_unknowns_and_residuals_to_segment(segment, initial_prop_power_coefficient = 0.16)


    mission.append_segment(segment)

    return mission

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
    weights = SUAVE.Analyses.Weights.Weights_eVTOL()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.4*vehicle.excrescence_area_spin / vehicle.reference_area
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Energy
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.networks
    analyses.append(energy)

    # ------------------------------------------------------------------
    #  Planet Analysis
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)

    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)

    return analyses

if __name__ == '__main__':
    main()

    print("Payload Range Regression Passed.")