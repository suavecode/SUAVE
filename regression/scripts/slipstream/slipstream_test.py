# Slipstream_Test.py
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

from SUAVE.Plots.Mission_Plots import *  
from SUAVE.Plots.Geometry_Plots.plot_vehicle import plot_vehicle  
from SUAVE.Plots.Geometry_Plots.plot_vehicle_vlm_panelization  import plot_vehicle_vlm_panelization
import sys
sys.path.append('../Vehicles') 
from X57_Maxwell import vehicle_setup, configs_setup 


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():

    configs, analyses = full_setup() 
    
    configs.finalize()
    analyses.finalize()  

    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()
     
    # lift coefficient  
    lift_coefficient              = results.segments.cruise.conditions.aerodynamics.lift_coefficient[1][0]
    lift_coefficient_true         = 0.41732243018101206

    print(lift_coefficient)
    diff_CL                       = np.abs(lift_coefficient  - lift_coefficient_true) 
    print('CL difference')
    print(diff_CL)
    assert np.abs(lift_coefficient  - lift_coefficient_true) < 1e-6
    
    # sectional lift coefficient check
    sectional_lift_coeff            = results.segments.cruise.conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional[0]
    sectional_lift_coeff_true       = np.array([4.06069910e-01, 4.03795913e-01, 3.84921041e-01, 3.33296062e-01,
                                                3.71319835e-01, 4.52766881e-01, 3.84962842e-01, 3.54150681e-01,
                                                3.29715932e-01, 3.01181799e-01, 2.67284913e-01, 2.27357864e-01,
                                                1.81572027e-01, 1.31744281e-01, 8.63610633e-02, 4.06069910e-01,
                                                4.03795913e-01, 3.84921041e-01, 3.33296062e-01, 3.71319835e-01,
                                                4.52766881e-01, 3.84962842e-01, 3.54150681e-01, 3.29715932e-01,
                                                3.01181799e-01, 2.67284913e-01, 2.27357864e-01, 1.81572027e-01,
                                                1.31744281e-01, 8.63610633e-02, 1.68367422e-02, 1.79634043e-02,
                                                1.94778080e-02, 2.13435071e-02, 2.34532179e-02, 2.56475960e-02,
                                                2.76986493e-02, 2.92945864e-02, 3.00847205e-02, 2.97471008e-02,
                                                2.80437082e-02, 2.48715894e-02, 2.03061215e-02, 1.46363439e-02,
                                                8.63976768e-03, 1.68367422e-02, 1.79634043e-02, 1.94778080e-02,
                                                2.13435071e-02, 2.34532179e-02, 2.56475960e-02, 2.76986493e-02,
                                                2.92945864e-02, 3.00847205e-02, 2.97471008e-02, 2.80437082e-02,
                                                2.48715894e-02, 2.03061215e-02, 1.46363439e-02, 8.63976768e-03,
                                                1.78736305e-33, 1.06916762e-34, 3.58394355e-36, 3.75859322e-36,
                                                3.53992892e-35, 7.34217058e-35, 8.31267757e-35, 9.81567209e-35,
                                                1.10462893e-34, 1.07979898e-34, 1.00336854e-34, 8.84723167e-35,
                                                6.66766719e-35, 3.89852169e-35, 1.36647862e-35])

    print(sectional_lift_coeff)
    diff_Cl                       = np.abs(sectional_lift_coeff - sectional_lift_coeff_true)
    print('Cl difference')
    print(diff_Cl)
    assert  np.max(np.abs(sectional_lift_coeff - sectional_lift_coeff_true)) < 1e-6

    # plot results 
    plot_mission(results,configs.base)  

    # Plot vehicle 
    plot_vehicle(configs.base, save_figure = False, plot_control_points = False)
    
    # Plot vortex distribution
    plot_vehicle_vlm_panelization(configs.base, save_figure=False, plot_control_points=True)
    
    return 

def plot_mission(results,vehicle): 
    
    # Plot surface pressure coefficient 
    plot_surface_pressure_contours(results,vehicle)
    
    # Plot lift distribution 
    plot_lift_distribution(results,vehicle)
    
    # Create Video Frames 
    create_video_frames(results,vehicle, save_figure = False)
    
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
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()     
    aerodynamics.settings.use_surrogate              = False
    aerodynamics.settings.propeller_wake_model       = True 
    aerodynamics.settings.number_spanwise_vortices   = 15
    aerodynamics.settings.number_chordwise_vortices  = 5   
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
    base_segment.process.iterate.unknowns.network            = vehicle.propulsors.battery_propeller.unpack_unknowns
    base_segment.process.iterate.residuals.network           = vehicle.propulsors.battery_propeller.residuals
    base_segment.state.unknowns.propeller_power_coefficient  = 0.2 * ones_row(1) 
    bat                                                      = vehicle.propulsors.battery_propeller.battery 
    base_segment.state.unknowns.battery_voltage_under_load   = bat.max_voltage * ones_row(1)  
    base_segment.state.residuals.network                     = 0. * ones_row(2) 
    base_segment.max_energy                                  = bat.max_energy 
    
    # ------------------------------------------------------------------
    #   Climb 1 : constant Speed, constant rate segment 
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"
    segment.analyses.extend( analyses.base )
    segment.battery_energy            =  bat.max_energy* 0.89
    segment.altitude_start            = 2500.0  * Units.feet
    segment.altitude_end              = 8012    * Units.feet 
    segment.air_speed                 = 96.4260 * Units['mph'] 
    segment.climb_rate                = 700.034 * Units['ft/min']  
    segment.state.unknowns.throttle   = 0.85 * ones_row(1)  

    # add to misison
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Cruise Segment: constant Speed, constant altitude
    # ------------------------------------------------------------------ 
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise" 
    segment.analyses.extend(analyses.base)  
    segment.air_speed                 = 135.   * Units['mph'] 
    segment.distance                  =  20.   * Units.nautical_mile  
    segment.state.unknowns.throttle   = 0.85  *  ones_row(1)   
    
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