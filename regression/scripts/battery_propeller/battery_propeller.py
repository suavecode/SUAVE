# test_VTOL.py
# 
# Created: Feb 2020, M. Clarke
#          Sep 2020, M. Clarke 

""" setup file for electric aircraft regression """

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units 
import numpy as np
import pylab as plt 
import copy, time
from SUAVE.Plots.Mission_Plots import *
from SUAVE.Core import Data, Container
import sys

sys.path.append('../Vehicles')
# the analysis functions 
 
from X57_Maxwell    import vehicle_setup, configs_setup 

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
    
    # save results 
    # This should be always left uncommented to test the SUAVE's Input/Output archive functions
    save_results(results)
    
    # plot the results
    plot_results(results) 
      
    # load and plot old results  
    old_results = load_results()
    plot_results(old_results)  
    
    # RPM of rotor check during hover
    RPM        = results.segments.climb_1.conditions.propulsion.rpm[3][0]
    RPM_true   = 880.2614944605407
    print(RPM) 
    diff_RPM   = np.abs(RPM - RPM_true)
    print('RPM difference')
    print(diff_RPM)
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3  
    
    # lift Coefficient Check During Cruise
    lift_coefficient        = results.segments.cruise.conditions.aerodynamics.lift_coefficient[2][0]
    lift_coefficient_true   = 0.38376159297585954
    print(lift_coefficient)
    diff_CL                 = np.abs(lift_coefficient  - lift_coefficient_true) 
    print('CL difference')
    print(diff_CL)
    assert np.abs((lift_coefficient  - lift_coefficient_true)/lift_coefficient_true) < 1e-3   
        
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ---------------------------------------------------------------------- 

def full_setup():

    # vehicle data
    vehicle  = vehicle_setup()
    
    # Set up configs
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
    weights = SUAVE.Analyses.Weights.Weights_Transport()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.AERODAS() 
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
    base_segment.state.numerics.number_control_points        = 4
    base_segment.process.iterate.unknowns.network            = vehicle.propulsors.battery_propeller.unpack_unknowns
    base_segment.process.iterate.residuals.network           = vehicle.propulsors.battery_propeller.residuals
    base_segment.state.unknowns.propeller_power_coefficient  = 0.005 * ones_row(1) 
    base_segment.state.unknowns.battery_voltage_under_load   = vehicle.propulsors.battery_propeller.battery.max_voltage * ones_row(1)  
    base_segment.state.residuals.network                     = 0. * ones_row(2)        
    
    # ------------------------------------------------------------------
    #   Climb 1 : constant Speed, constant rate segment 
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"
    segment.analyses.extend( analyses.base )
    segment.battery_energy                   = vehicle.propulsors.battery_propeller.battery.max_energy * 0.89
    segment.altitude_start                   = 2500.0  * Units.feet
    segment.altitude_end                     = 8012    * Units.feet 
    segment.air_speed                        = 96.4260 * Units['mph'] 
    segment.climb_rate                       = 700.034 * Units['ft/min']  
    segment.state.unknowns.throttle          = 0.85 * ones_row(1)  

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Cruise Segment: constant Speed, constant altitude
    # ------------------------------------------------------------------ 
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise" 
    segment.analyses.extend(analyses.base) 
    segment.altitude                  = 8012   * Units.feet
    segment.air_speed                 = 140.91 * Units['mph'] 
    segment.distance                  =  20.   * Units.nautical_mile  
    segment.state.unknowns.throttle   = 0.9 *  ones_row(1)   

    # add to misison
    mission.append_segment(segment)    

    # ------------------------------------------------------------------
    #   Descent Segment: constant Speed, constant rate segment 
    # ------------------------------------------------------------------ 
    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "decent" 
    segment.analyses.extend( analyses.base ) 
    segment.altitude_start            = 8012  * Units.feet
    segment.altitude_end              = 2500  * Units.feet
    segment.air_speed                 = 140.91 * Units['mph']  
    segment.climb_rate                = - 500.401  * Units['ft/min']  
    segment.state.unknowns.throttle   = 0.9 * ones_row(1)  
    
    # add to misison
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


def plot_results(results,line_style = 'bo-'):  
    
    # Plot Flight Conditions 
    plot_flight_conditions(results, line_style) 
    
    # Plot Aerodynamic Coefficients
    plot_aerodynamic_coefficients(results, line_style)  
    
    # Plot Aircraft Flight Speed
    plot_aircraft_velocities(results, line_style)
    
    # Plot Aircraft Electronics
    plot_electronic_conditions(results, line_style)
    
    # Plot Propeller Conditions 
    plot_propeller_conditions(results, line_style) 
    
    # Plot Electric Motor and Propeller Efficiencies 
    plot_eMotor_Prop_efficiencies(results, line_style)
    
    # Plot propeller Disc and Power Loading
    plot_disc_power_loading(results, line_style)   
     
     
    return


def load_results():
    return SUAVE.Input_Output.SUAVE.load('electric_aircraft.res')

def save_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'electric_aircraft.res')
    
    return  
if __name__ == '__main__': 
    main()    
