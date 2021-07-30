# unique_propellers_test.py
# 
# Created:   Jul 2021, R. Erhard
# Modified: 

""" 
Setup file for a cruise segment of the NASA X-57 Maxwell (Mod 4) Electric Aircraft.
Runs the propeller-wing interaction with two different types of propellers. Two tip-
mounted cruise propellers, and 12 additional smaller propellers for lift augmentation
via slipstream interaction during takeoff/landing.
"""
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units,Data

import numpy as np
import pylab as plt
import sys
import time

from SUAVE.Plots.Mission_Plots import *  
from SUAVE.Plots.Geometry_Plots.plot_vehicle import plot_vehicle  
from SUAVE.Plots.Geometry_Plots.plot_vehicle_vlm_panelization  import plot_vehicle_vlm_panelization

from SUAVE.Input_Output.VTK.save_vehicle_vtk import save_vehicle_vtks

sys.path.append('../Vehicles') 
from X57_Mod4 import vehicle_setup, configs_setup 


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    # run test with helical fixed wake model
    helical_fixed_wake_analysis()
    return 

def helical_fixed_wake_analysis():
    ti = time.time()
    # Evaluate wing in propeller wake (using helical fixed-wake model)
    fixed_helical_wake = True
    configs, analyses  = full_setup(fixed_helical_wake) 

    configs.finalize()
    analyses.finalize()  

    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()

    # lift coefficient  
    lift_coefficient              = results.segments.climb_1.conditions.aerodynamics.lift_coefficient[0][0]
    lift_coefficient_true         = 0.5306053340921111

    print(lift_coefficient)
    diff_CL                       = np.abs(lift_coefficient  - lift_coefficient_true) 
    print('CL difference')
    print(diff_CL)

    #Results  = Data()
    #Results['identical'] = False
    #Results['all_prop_outputs'] = results.segments.cruise.conditions.noise.sources.propellers 
    #save_vehicle_vtks(configs.base,Results,time_step=1,save_loc="/Users/rerha/Desktop/mod4/")
    
    telapsed = time.time() - ti
    
    assert np.abs(lift_coefficient  - lift_coefficient_true) < 1e-6

    # sectional lift coefficient check
    sectional_lift_coeff            = results.segments.climb_1.conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional[0]
    sectional_lift_coeff_true       = np.array([ 9.70469690e-01,  1.24765575e+00,  1.17458075e+00,  3.66746565e-01,
                                                 3.10118917e-01,  9.70469746e-01,  1.24765610e+00,  1.17458115e+00,
                                                 3.66747127e-01,  3.10119059e-01,  2.87579592e-01,  3.05938576e-01,
                                                 2.93656626e-01,  2.40733997e-01,  1.53052257e-01,  2.87579597e-01,
                                                 3.05938607e-01,  2.93656721e-01,  2.40734290e-01,  1.53052742e-01,
                                                -2.12352683e-15,  1.80144906e-15,  2.79713246e-15,  3.33525018e-15,
                                                 2.30033709e-15])

    print(sectional_lift_coeff)
    diff_Cl = np.abs(sectional_lift_coeff - sectional_lift_coeff_true)
    print('Cl difference')
    print(diff_Cl)
    assert  np.max(np.abs(sectional_lift_coeff - sectional_lift_coeff_true)) < 1e-6

    return

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup(fixed_helical_wake):

    # vehicle data
    vehicle  = vehicle_setup() 
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs, fixed_helical_wake)

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

def analyses_setup(configs, fixed_helical_wake):

    analyses = SUAVE.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config, fixed_helical_wake)
        analyses[tag] = analysis

    return analyses

def base_analysis(vehicle, fixed_helical_wake):

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
    weights = SUAVE.Analyses.Weights.Weights_Transport()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()    
    if fixed_helical_wake ==True:
        aerodynamics.settings.use_surrogate              = False
        aerodynamics.settings.propeller_wake_model       = True   
        aerodynamics.settings.use_bemt_wake_model        = False      

    aerodynamics.settings.number_spanwise_vortices   = 5
    aerodynamics.settings.number_chordwise_vortices  = 2   
    aerodynamics.settings.spanwise_cosine_spacing    = True  
    aerodynamics.geometry                            = vehicle
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
    mission.tag = 'mission'

    # airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0. * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments 
    
    # base segment
    base_segment = Segments.Segment()
    ones_row     = base_segment.state.ones_row
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
    base_segment.state.numerics.number_control_points        = 2
    
    ## ------------------------------------------------------------------
    ##   Climb 1 : constant Speed, constant rate segment 
    ## ------------------------------------------------------------------ 
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"
    segment.analyses.extend( analyses.base )
    segment.battery_energy            = vehicle.propulsors.battery_propeller.battery.max_energy* 0.89
    segment.altitude_start            = 2500.0  * Units.feet
    segment.altitude_end              = 8012    * Units.feet 
    segment.air_speed                 = 96.4260 * Units['mph'] 
    segment.climb_rate                = 700.034 * Units['ft/min']  
    segment.state.unknowns.throttle   = 0.85 * ones_row(1)
    segment = vehicle.propulsors.battery_propeller.add_unknowns_and_residuals_to_segment(segment)

    # add to misison
    mission.append_segment(segment)
    
    ## ------------------------------------------------------------------
    ##   Cruise Segment: constant Speed, constant altitude
    ## ------------------------------------------------------------------ 
    #segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    #segment.tag = "cruise" 
    #segment.analyses.extend(analyses.base)  
    ##segment.altitude                  = 8012.0  * Units.feet
    #segment.air_speed                 = 135. * Units['mph'] 
    #segment.distance                  = 20.  * Units.nautical_mile  
    #segment.state.unknowns.throttle   = 0.85 *  ones_row(1)
    #segment = vehicle.propulsors.battery_propeller.add_unknowns_and_residuals_to_segment(segment)
    
    ## add to misison
    #mission.append_segment(segment)        
    
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
