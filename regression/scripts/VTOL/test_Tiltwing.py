# test_Tiltwing.py
# 
# Created: Feb 2020, M. Clarke
#          Sep 2020, M. Clarke 

""" setup file for a mission with Tiltwing eVTOL  
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units 
from SUAVE.Visualization.Performance.Aerodynamics.Vehicle import *  
from SUAVE.Visualization.Performance.Mission              import *  
from SUAVE.Visualization.Performance.Energy.Common        import *  
from SUAVE.Visualization.Performance.Energy.Battery       import *   
from SUAVE.Visualization.Performance.Noise                import *  
from SUAVE.Visualization.Geometry.Three_Dimensional.plot_3d_vehicle import plot_3d_vehicle 
import numpy as np  
import sys 

sys.path.append('../Vehicles')
# the analysis functions

from Tiltwing      import vehicle_setup, configs_setup  

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():   

    # ------------------------------------------------------------------------------------------------------------
    # Tiltwing CRM  
    # ------------------------------------------------------------------------------------------------------------
    # build the vehicle, configs, and analyses
    configs, analyses = full_setup() 
    configs.finalize()
    analyses.finalize()

    # Print weight properties of vehicle
    print(configs.base.weight_breakdown)
    print(configs.base.mass_properties.center_of_gravity)

    # Plot vehicle 
    plot_3d_vehicle(configs.cruise, save_figure = False, plot_wing_control_points = False)

    # evaluate mission    
    mission  = analyses.missions.base
    results  = mission.evaluate()

    # plot results
    plot_mission(results)   

    # save, load and plot old results 
    save_tiltwing_results(results)
    old_results = load_tiltwing_results() 
    plot_mission(old_results)

    # RPM check during hover
    RPM        = results.segments.departure.conditions.propulsion.propeller_rpm[0][0]
    RPM_true   = 1922.161595958687
    
    print(RPM) 
    diff_RPM   = np.abs(RPM - RPM_true)
    print('RPM difference')
    print(diff_RPM)
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3  

    # lift Coefficient Check During Cruise
    lift_coefficient        = results.segments.climb.conditions.aerodynamics.lift_coefficient[0][0] 
    lift_coefficient_true   = 1.0303444240760393
    print(lift_coefficient)
    diff_CL                 = np.abs(lift_coefficient  - lift_coefficient_true) 
    print('CL difference')
    print(diff_CL)
    assert np.abs((lift_coefficient  - lift_coefficient_true)/lift_coefficient_true) < 1e-3   
    return 

# ----------------------------------------------------------------------
#   Setup
# ----------------------------------------------------------------------
def full_setup():    

    # vehicle data
    vehicle  = vehicle_setup()
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission           = mission_setup(configs_analyses,vehicle)
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
    weights = SUAVE.Analyses.Weights.Weights_eVTOL()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry                = vehicle 
    aerodynamics.settings.model_fuselage = True     
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


def mission_setup(analyses,vehicle):
        
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission     = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'mission'

    # airport
    airport            = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment                                             = Segments.Segment()
    base_segment.state.numerics.number_control_points        = 5
    ones_row                                                 = base_segment.state.ones_row 
    base_segment.process.initialize.initialize_battery       = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery

  
    # ------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------ 
    segment                                            = Segments.Hover.Climb(base_segment)
    segment.tag                                        = "departure" 
    segment.analyses.extend( analyses.hover_climb ) 
    segment.altitude_start                             = 0.0  * Units.ft
    segment.altitude_end                               = 40.  * Units.ft
    segment.climb_rate                                 = 300. * Units['ft/min']
    segment.battery_energy                             = vehicle.networks.battery_rotor.battery.max_energy   
    segment.state.unknowns.throttle                    = 1.0 * ones_row(1) 
    segment.process.iterate.conditions.stability       = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability    = SUAVE.Methods.skip
    segment = vehicle.networks.battery_rotor.add_unknowns_and_residuals_to_segment(segment,\
                                                                                         initial_power_coefficient = 0.06)

    # add to misison
    mission.append_segment(segment) 
    
    # ------------------------------------------------------------------
    #   First Cruise Segment: Constant Acceleration, Constant Altitude
    # ------------------------------------------------------------------ 
    segment                                            = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    segment.tag                                        = "Climb" 
    segment.analyses.extend(analyses.cruise) 
    segment.climb_rate                                 = 300. * Units['ft/min']
    segment.air_speed_start                            = 85.   * Units['mph']
    segment.air_speed_end                              = 130.   * Units['mph']
    segment.altitude_start                             = 40.0 * Units.ft
    segment.altitude_end                               = 600.0 * Units.ft    
    segment.state.unknowns.throttle                    = 0.80 * ones_row(1)    
    segment.process.iterate.conditions.stability       = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability    = SUAVE.Methods.skip      
    segment = vehicle.networks.battery_rotor.add_unknowns_and_residuals_to_segment(segment,\
                                                                                         initial_power_coefficient = 0.03)

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


# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def plot_mission(results):  
    
    # Plot Flight Conditions 
    plot_flight_conditions(results) 
    
    # Plot Aerodynamic Coefficients
    plot_aerodynamic_coefficients(results)  
    
    # Plot Aircraft Flight Speed
    plot_aircraft_velocities(results)
    
    # Plot Aircraft Electronics
    plot_battery_pack_conditions(results)
    
    # Plot Propeller Conditions 
    plot_rotor_conditions(results) 
    
    # Plot Electric Motor and Propeller Efficiencies 
    plot_electric_motor_and_rotor_efficiencies(results)

    # Plot propeller Disc and Power Loading
    plot_disc_power_loading(results)  

    return


def load_tiltwing_results():
    return SUAVE.Input_Output.SUAVE.load('results_tiltwing.res')

def save_tiltwing_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'results_tiltwing.res')
    return

if __name__ == '__main__': 
    main()     
    plt.show(block=True)            
