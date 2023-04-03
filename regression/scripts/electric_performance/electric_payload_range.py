# electric_payload_range.py
#
# Created: Jan 2021, J. Smart
# Modified:

#-------------------------------------------------------------------------------
# Imports
#_______________________________________________________________________________

import MARC

from MARC.Core import Units, Data
from MARC.Methods.Performance.electric_payload_range import electric_payload_range

import numpy as np

import sys
sys.path.append('../Vehicles')

from Stopped_Rotor import *

#-------------------------------------------------------------------------------
# Test Function
#-------------------------------------------------------------------------------

def main():

    vehicle     = vehicle_setup() 
    configs     = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses 
    mission  = mission_setup(vehicle,configs_analyses)
    missions_analyses = missions_setup(mission) 

    analyses = MARC.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses
 
    configs.finalize()
    analyses.finalize()   

    # mission analysis
    mission = analyses.missions.base 

    payload_range = electric_payload_range(vehicle, mission, 'cruise', display_plot=True)

    payload_range_r = np.array([ 0.        , 65.93215289136965, 70.17601029180459])

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

    mission = MARC.Analyses.Mission.Variable_Range_Cruise.Given_State_of_Charge()
    mission.tag = 'the_mission'

    # the cruise tag to vary cruise distance
    mission.cruise_tag = 'cruise'
    mission.target_state_of_charge = 0.5

    # unpack Segments module
    Segments = MARC.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment()
    ones_row                                                 = base_segment.state.ones_row    
    base_segment.state.numerics.number_control_points        = 4
    base_segment.process.iterate.conditions.stability        = MARC.Methods.skip
    base_segment.process.finalize.post_process.stability     = MARC.Methods.skip    
    base_segment.process.iterate.conditions.planet_position  = MARC.Methods.skip    
    base_segment.process.initialize.initialize_battery       = MARC.Methods.Missions.Segments.Common.Energy.initialize_battery

    # ------------------------------------------------------------------
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    segment.analyses.extend(analyses.forward_flight)  
    segment.altitude  = 1000.0 * Units.ft
    segment.air_speed = 110.   * Units['mph']
    segment.distance  = 80.    * Units.miles     
    segment.battery_energy = vehicle.networks.battery_electric_rotor.battery.pack.max_energy 
    segment = vehicle.networks.battery_electric_rotor.add_unknowns_and_residuals_to_segment(segment,
                                                                                            initial_throttles = [0.8,0],
                                                                                            initial_rotor_power_coefficients=[0.16,0]) 

    mission.append_segment(segment)

    return mission

# ----------------------------------------------------------------------
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------

def analyses_setup(configs):

    analyses = MARC.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config)
        analyses[tag] = analysis

    return analyses

def base_analysis(vehicle):

    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------
    analyses = MARC.Analyses.Vehicle()

    # ------------------------------------------------------------------
    #  Basic Geometry Relations
    sizing = MARC.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)

    # ------------------------------------------------------------------
    #  Weights
    weights = MARC.Analyses.Weights.Weights_eVTOL()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = MARC.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.4*vehicle.excrescence_area_spin / vehicle.reference_area
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Energy
    energy= MARC.Analyses.Energy.Energy()
    energy.network = vehicle.networks
    analyses.append(energy)

    # ------------------------------------------------------------------
    #  Planet Analysis
    planet = MARC.Analyses.Planets.Planet()
    analyses.append(planet)

    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = MARC.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)

    return analyses


def missions_setup(base_mission):

    # the mission container
    missions = MARC.Analyses.Mission.Mission.Container()

    # ------------------------------------------------------------------
    #   Base Mission
    # ------------------------------------------------------------------

    missions.base = base_mission


    # done!
    return missions  

if __name__ == '__main__':
    main()

    print("Payload Range Regression Passed.")