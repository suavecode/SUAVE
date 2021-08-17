# test_VTOL.py
# 
# Created: Feb 2020, M. Clarke
#          Sep 2020, M. Clarke 
#          Jul 2021, R. Erhard

""" setup file for electric aircraft regression """

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units 
import numpy as np
from SUAVE.Plots.Mission_Plots import *
from SUAVE.Core import Data
from SUAVE.Methods.Weights.Buildups.eVTOL.empty import empty 
import sys

sys.path.append('../Vehicles')
from X57_Maxwell_Mod2    import vehicle_setup, configs_setup 

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():                
    configs, analyses = full_setup()

    simple_sizing(configs)

    configs.finalize()
    analyses.finalize()  
    
    # evaluate the weight of an electric general aviation aircraft using eVTOL weight build-up
    evtol_breakdown = empty(configs.base,contingency_factor=1.1)
    print(evtol_breakdown)    
    
    # check weights
    empty_r       = 1004.473514153519
    structural_r  = 293.21228559410815
    total_r       = 1285.4735141535189
    lift_rotors_r = 0.
    propellers_r  = 4.097659513422731
    prop_motors_r = 20.
    rot_motors_r  = 0.
    
    weights_error = Data()
    weights_error.empty       = abs(empty_r - evtol_breakdown.empty)/empty_r
    weights_error.structural  = abs(structural_r - evtol_breakdown.structural)/structural_r
    weights_error.total       = abs(total_r - evtol_breakdown.total)/total_r
    weights_error.lift_rotors = abs(lift_rotors_r - evtol_breakdown.lift_rotors)/lift_rotors_r if lift_rotors_r!=0 else 0 
    weights_error.propellers  = abs(propellers_r - evtol_breakdown.propellers)/propellers_r if propellers_r!=0 else 0 
    weights_error.prop_motors = abs(prop_motors_r - evtol_breakdown.propeller_motors)/prop_motors_r if prop_motors_r!=0 else 0 
    weights_error.rot_motors  = abs(rot_motors_r - evtol_breakdown.lift_rotor_motors)/rot_motors_r if rot_motors_r!=0 else 0
    
    for k, v in weights_error.items():
        assert (np.abs(v) < 1E-6)    
        
            
     
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
    RPM        = results.segments.climb_1.conditions.propulsion.propeller_rpm[3][0]
    RPM_true   = 887.1356296331286
    print(RPM) 
    diff_RPM   = np.abs(RPM - RPM_true)
    print('RPM difference')
    print(diff_RPM)
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3  
    
    # lift Coefficient Check During Cruise
    lift_coefficient        = results.segments.cruise.conditions.aerodynamics.lift_coefficient[2][0]
    lift_coefficient_true   =0.40171393102909314
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

    
    # ------------------------------------------------------------------
    #   Climb 1 : constant Speed, constant rate segment 
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"
    segment.analyses.extend( analyses.base )
    segment.battery_energy                   = vehicle.networks.battery_propeller.battery.max_energy * 0.89
    segment.altitude_start                   = 2500.0  * Units.feet
    segment.altitude_end                     = 8012    * Units.feet 
    segment.air_speed                        = 96.4260 * Units['mph'] 
    segment.climb_rate                       = 700.034 * Units['ft/min']  
    segment.state.unknowns.throttle          = 0.85 * ones_row(1)
    segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)    

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
    segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)   

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
    segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)   
    
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
