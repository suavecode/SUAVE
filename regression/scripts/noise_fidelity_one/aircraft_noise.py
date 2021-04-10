# aircraft_noise.py
#
# Created: Arp 2021, M. Clarke 

""" setup file for the X57-Maxwell Electric Aircraft to valdiate noise in a climb segment
"""
 
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units 
import numpy as np    
from SUAVE.Core import Data 
from SUAVE.Plots.Mission_Plots import *   
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform
import matplotlib.pyplot as plt 

import sys

sys.path.append('../Vehicles')
# the analysis functions 
 
from X57_Maxwell  import vehicle_setup as  X57_vehicle_setup
from X57_Maxwell  import configs_setup as  X57_configs_setup  
from Boeing_737   import vehicle_setup as  B737_vehicle_setup 
from Boeing_737   import configs_setup as  B737_configs_setup 

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():   
    # ----------------------------------------------------------------------
    # SUAVE Frequecy Domain Propeller Aircraft Noise Model 
    # ---------------------------------------------------------------------- 
    configs, analyses = X57_full_setup() 

    configs.finalize()
    analyses.finalize()   
    
    # mission analysis
    mission = analyses.missions.base
    X57_results = mission.evaluate()  
    
    # plot the results
    X57_filename = "X57_Noise"
    plot_results(X57_results,X57_filename)  
    
    # SPL of rotor check during hover
    X57_SPL        = X57_results.segments.ica.conditions.noise.total_SPL_dBA[3][0]
    X57_SPL_true   = 80.33223391658487
    print(X57_SPL) 
    X57_diff_SPL   = np.abs(X57_SPL - X57_SPL_true)
    print('SPL difference')
    print(X57_diff_SPL)
    assert np.abs((X57_SPL - X57_SPL_true)/X57_SPL_true) < 1e-3    
    
    
    
    # ----------------------------------------------------------------------
    # SAE Turbofan Aircraft Noise Model 
    # ---------------------------------------------------------------------- 
    configs, analyses = B737_full_setup() 

    configs.finalize()
    analyses.finalize()   
    
    # mission analysis
    mission       = analyses.missions.base
    B737_results  = mission.evaluate()  
    
    # plot the results
    B737_filename = "B737_Noise"
    plot_results(B737_results,B737_filename)  
    
    # SPL of rotor check during hover
    B737_SPL        = B737_results.segments.climb.conditions.noise.total_SPL_dBA[3][0]
    B737_SPL_true   = 56.96076190763407
    print(B737_SPL) 
    B737_diff_SPL   = np.abs(B737_SPL - B737_SPL_true)
    print('SPL difference')
    print(B737_diff_SPL)
    assert np.abs((B737_SPL - B737_SPL_true)/B737_SPL_true) < 1e-3    
    return     
 
# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------  
def X57_full_setup():

    # vehicle data
    vehicle  = X57_vehicle_setup()
    
    # Set up configs
    configs  = X57_configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = X57_mission_setup(configs_analyses,vehicle)
    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses

    return configs, analyses
 

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------  
def B737_full_setup():

    # vehicle data
    vehicle  = B737_vehicle_setup()
    vehicle.wings.main_wing.control_surfaces.flap.configuration_type = 'triple_slotted'  
    vehicle.wings.main_wing.high_lift = True
    vehicle.wings.main_wing = wing_planform(vehicle.wings.main_wing)
    
    # Set up configs
    configs  = B737_configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = B737_mission_setup(configs_analyses)
    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses

    return configs, analyses 

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
    weights = SUAVE.Analyses.Weights.Weights_Transport()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero() 
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Noise Analysis
    noise = SUAVE.Analyses.Noise.Fidelity_One()   
    noise.geometry = vehicle 
    analyses.append(noise)

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
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------

def analyses_setup(configs):

    analyses = SUAVE.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config)
        analyses[tag] = analysis

    return analyses 


def simple_sizing(configs):

    base = configs.base
    base.pull_base()

    # wing areas
    for wing in base.wings:
        wing.areas.wetted   = 1.75 * wing.areas.reference
        wing.areas.exposed  = 0.8  * wing.areas.wetted
        wing.areas.affected = 0.6  * wing.areas.wetted


    # diff the new data
    base.store_diff()

    # done!
    return
 
# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------

def X57_mission_setup(analyses,vehicle):  
    
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
    base_segment                                             = Segments.Segment()
    ones_row                                                 = base_segment.state.ones_row
    base_segment.use_Jacobian                                = False  
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
    base_segment.state.numerics.number_control_points        = 4
    base_segment.process.iterate.unknowns.network            = vehicle.propulsors.battery_propeller.unpack_unknowns
    base_segment.process.iterate.residuals.network           = vehicle.propulsors.battery_propeller.residuals
    base_segment.state.unknowns.propeller_power_coefficient  = 0.005 * ones_row(1) 
    base_segment.state.unknowns.battery_voltage_under_load   = vehicle.propulsors.battery_propeller.battery.max_voltage * ones_row(1)  
    base_segment.state.residuals.network                     = 0. * ones_row(2)        
    
    # ------------------------------------------------------------------
    #   Initial Climb Area Segment Flight 1  
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment) 
    segment.tag = 'ICA' 
    segment.analyses.extend( analyses.base )  
    segment.battery_energy                                   = vehicle.propulsors.battery_propeller.battery.max_energy  
    segment.state.unknowns.throttle                          = 0.85  * ones_row(1)  
    segment.altitude_start                                   = 50.0 * Units.feet
    segment.altitude_end                                     = 500.0 * Units.feet
    segment.air_speed_start                                  = 45  * Units['m/s']   
    segment.air_speed_end                                    = 50 * Units['m/s']   
    segment.climb_rate                                       = 600 * Units['ft/min']    
    mission.append_segment(segment) 
              
    
    return mission

def B737_mission_setup(analyses): 

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
    base_segment.state.numerics.number_control_points = 4

    # ------------------------------------------------------------------
    #    Climb 1 : Constant Speed Constant Rate  
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb"

    segment.analyses.extend( analyses.takeoff )

    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 0.05  * Units.km
    segment.air_speed      = 150   * Units.knots
    segment.climb_rate     = 10.0  * Units['m/s']

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


def plot_results(results,filename):   
    
    # Plot noise level
    plot_noise_level(results,save_filename = filename)
    
    # Plot noise contour
    plot_flight_profile_noise_contour(results,save_filename = filename + 'contour') 
                        
    return  

if __name__ == '__main__': 
    main()    
    plt.show()