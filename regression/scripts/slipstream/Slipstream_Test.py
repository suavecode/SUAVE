# Cessna_172.py
# 
# Created:  Mar 2019, M. Clarke

""" setup file for a cruise segment of the NASA X-57 Maxwell (Twin Engine Variant) Electric Aircraft 
"""
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt

from SUAVE.Core import Data , Container
from SUAVE.Methods.Propulsion import propeller_design

import sys
sys.path.append('../Vehicles') 
from X57_Maxwell import vehicle_setup, configs_setup, simple_sizing

import copy

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():

    configs, analyses = full_setup()

    simple_sizing(configs)

    configs.finalize()
    analyses.finalize()  

    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()
     
    # lift coefficient  
    lift_coefficient              = results.segments.cruise.conditions.aerodynamics.lift_coefficient[2][0]
    lift_coefficient_true         = 0.31879931419126495
    print(lift_coefficient)
    diff_CL                       = np.abs(lift_coefficient  - lift_coefficient_true) 
    print('CL difference')
    print(diff_CL)
    assert np.abs((lift_coefficient  - lift_coefficient_true)/lift_coefficient_true) < 1e-6
    
    # sectional lift coefficient check
    sectional_lift_coeff            = results.segments.cruise.conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional_lift[0]
    sectional_lift_coeff_true       = np.array([8.10449702e-02, 7.54016010e-02, 6.43687599e-02, 2.66201837e-02,
                                             4.77585486e-02, 7.56764672e-02, 5.75279914e-02, 5.05558433e-02,
                                             4.48641577e-02, 2.56410823e-02, 5.06247203e-02, 3.41295369e-02,
                                             3.15424389e-02, 2.89157714e-02, 2.61898619e-02, 2.33523357e-02,
                                             2.04128915e-02, 1.74000220e-02, 1.43609316e-02, 1.13583867e-02,
                                             8.46298498e-03, 5.75476361e-03, 3.35577286e-03, 1.45980010e-03,
                                             2.88091664e-04, 8.17941103e-02, 7.78022624e-02, 6.91059517e-02,
                                             3.73633082e-02, 6.35641862e-02, 9.34925571e-02, 6.45194641e-02,
                                             5.47924043e-02, 4.77575898e-02, 2.70206128e-02, 5.28332269e-02,
                                             3.53372162e-02, 3.24846399e-02, 2.96596558e-02, 2.67799691e-02,
                                             2.38203101e-02, 2.07819577e-02, 1.76876794e-02, 1.45807606e-02,
                                             1.15212920e-02, 8.57801097e-03, 5.82971314e-03, 3.39810138e-03,
                                             1.47781185e-03, 2.91601787e-04, 3.25301003e-03, 3.23886941e-03,
                                             3.34503763e-03, 3.56019195e-03, 3.86155288e-03, 4.22535781e-03,
                                             4.62329858e-03, 5.01935830e-03, 5.36947070e-03, 5.62488413e-03,
                                             5.73978095e-03, 5.68281274e-03, 5.44728675e-03, 5.05302969e-03,
                                             4.53883338e-03, 3.95069625e-03, 3.33178496e-03, 2.71696429e-03,
                                             2.13176049e-03, 1.59422076e-03, 1.11786628e-03, 7.14136395e-04,
                                             3.93300783e-04, 1.63922439e-04, 3.17491303e-05, 1.81442840e-03,
                                             2.01942079e-03, 2.26788087e-03, 2.56066842e-03, 2.89374016e-03,
                                             3.25427638e-03, 3.62222571e-03, 3.96952507e-03, 4.26069577e-03,
                                             4.45637371e-03, 4.52057609e-03, 4.43123441e-03, 4.18904405e-03,
                                             3.81837941e-03, 3.35948636e-03, 2.85696148e-03, 2.35018105e-03,
                                             1.86836710e-03, 1.43001818e-03, 1.04508106e-03, 7.18131069e-04,
                                             4.51192705e-04, 2.45392178e-04, 1.01451849e-04, 1.95841648e-05,
                                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                             0.00000000e+00])
    print(sectional_lift_coeff)
    diff_Cl                       = np.abs(sectional_lift_coeff - sectional_lift_coeff_true)
    print('Cl difference')
    print(diff_Cl)
    assert  max(np.abs((sectional_lift_coeff - sectional_lift_coeff_true)/sectional_lift_coeff_true)) < 1e-6 
    
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():

    # vehicle data
    vehicle  = vehicle_setup() 
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = mission_setup(configs_analyses,vehicle) 
    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses

    return configs, analyses

# ----------------------------------------------------------------------
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------
def analyses_setup(configs):

    configs, analyses = full_setup()

    simple_sizing(configs)

    configs.finalize()
    analyses.finalize()

    # weight analysis
    weights = analyses.configs.base.weights
    breakdown = weights.evaluate()      

    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()

    return analyses
# ----------------------------------------------------------------------
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------

def analyses_setup(configs):

    analyses = SUAVE.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in configs.items():
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

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.settings.use_surrogate = False
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Stability Analysis
    stability = SUAVE.Analyses.Stability.Fidelity_Zero()    
    stability.geometry = vehicle
    analyses.append(stability)

    # ------------------------------------------------------------------
    #  Energy
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.propulsors 
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

    # done!
    return analyses    


# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------

def mission_setup(analyses,vehicle):

    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'the_mission'

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment 
    base_segment = Segments.Segment()
    ones_row     = base_segment.state.ones_row
    base_segment.state.numerics.number_control_points = 5 
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
    base_segment.process.iterate.unknowns.network            = vehicle.propulsors.propulsor.unpack_unknowns
    base_segment.process.iterate.residuals.network           = vehicle.propulsors.propulsor.residuals
    base_segment.state.unknowns.propeller_power_coefficient  = 0.005 * ones_row(1) 
    base_segment.state.unknowns.battery_voltage_under_load   = vehicle.propulsors.propulsor.battery.max_voltage * ones_row(1)  
    base_segment.state.residuals.network                     = 0. * ones_row(2)       

    # ------------------------------------------------------------------
    #   Cruise Segment: constant Speed, constant altitude
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    segment.analyses.extend(analyses.base)
    
    segment.altitude  = 8500. * Units.feet
    segment.air_speed = 132.   *Units['mph']  
    segment.distance  = 50.   * Units.nautical_mile
    segment.state.unknowns.throttle   = 0.75 * ones_row(1)  # for slipstream branch
    
    # add to misison
    mission.append_segment(segment)    
    
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
    plt.show()