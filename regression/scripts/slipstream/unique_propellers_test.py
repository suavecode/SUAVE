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

from SUAVE.Plots.Mission_Plots import *  
from SUAVE.Plots.Geometry_Plots.plot_vehicle import plot_vehicle  
from SUAVE.Input_Output.VTK.save_vehicle_vtk import save_vehicle_vtks
from SUAVE.Methods.Weights.Buildups.eVTOL.empty import empty 

sys.path.append('../Vehicles') 
from X57_Mod4 import vehicle_setup, configs_setup 


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    # run test with helical fixed wake model
    save_vtks = False
    plot_vehicle = False
    helical_fixed_wake_analysis(save_vtks,plot_vehicle)
    return 

def helical_fixed_wake_analysis(save_vtks,plot_vehicle):
    # Evaluate wing in propeller wake (using helical fixed-wake model)
    fixed_helical_wake = True
    configs, analyses  = full_setup(fixed_helical_wake) 

    configs.finalize()
    analyses.finalize()  

    # mission analysis
    mission = analyses.missions.base
    print("\nEvaluating the mission...")
    results = mission.evaluate()

    # lift coefficient  
    lift_coefficient              = results.segments.cruise.conditions.aerodynamics.lift_coefficient[0][0]
    lift_coefficient_true         = 0.2277127893248017
    diff_CL                       = np.abs(lift_coefficient  - lift_coefficient_true)
    print(lift_coefficient)
    print('CL difference')
    print(diff_CL)

    # sectional lift coefficient check
    sectional_lift_coeff            = results.segments.cruise.conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional[0]
    sectional_lift_coeff_true       = np.array([ 5.36394747e-01,  7.08910766e-01,  4.90185369e-01, -1.01002420e-01,
                                                 -8.39700963e-03,  5.36394802e-01,  7.08910959e-01,  4.90185405e-01,
                                                 -1.01002449e-01, -8.39708270e-03,  4.17360381e-02,  4.26519321e-02,
                                                  3.67531636e-02,  2.56934392e-02,  1.50074699e-02,  4.17360203e-02,
                                                  4.26519099e-02,  3.67531364e-02,  2.56934255e-02,  1.50074953e-02,
                                                  1.93442102e-15,  1.08705900e-16, -2.92033700e-16, -3.92391436e-16,
                                                 -2.63521794e-16])

    diff_Cl                         = np.abs(sectional_lift_coeff - sectional_lift_coeff_true)
    
    print(sectional_lift_coeff)
    print('Cl difference')
    print(diff_Cl)
    
    assert np.abs(lift_coefficient  - lift_coefficient_true) < 1e-5
    assert np.max(np.abs(sectional_lift_coeff - sectional_lift_coeff_true)) < 1e-5
    
    # Check weights breakdown
    evtol_breakdown = empty(configs.base,contingency_factor=1.1)

    empty_r       = 1794.957643448814
    structural_r  = 829.4342213171035
    total_r       = 2075.957643448814
    lift_rotors_r = 0.
    propellers_r  = 10.224459471963282
    prop_motors_r = 140.0
    rot_motors_r  = 0.
    
    weights_error = Data()
    weights_error.empty       = abs(empty_r - evtol_breakdown.empty)/empty_r
    weights_error.structural  = abs(structural_r - evtol_breakdown.structural)/structural_r
    weights_error.total       = abs(total_r - evtol_breakdown.total)/total_r
    weights_error.lift_rotors = abs(lift_rotors_r - evtol_breakdown.lift_rotors)/lift_rotors_r if lift_rotors_r!=0 else abs(lift_rotors_r - evtol_breakdown.lift_rotors)
    weights_error.propellers  = abs(propellers_r - evtol_breakdown.propellers)/propellers_r if propellers_r!=0 else abs(propellers_r - evtol_breakdown.propellers)
    weights_error.prop_motors = abs(prop_motors_r - evtol_breakdown.propeller_motors)/prop_motors_r if prop_motors_r!=0 else abs(prop_motors_r - evtol_breakdown.propeller_motors)
    weights_error.rot_motors  = abs(rot_motors_r - evtol_breakdown.lift_rotor_motors)/rot_motors_r if rot_motors_r!=0 else abs(rot_motors_r - evtol_breakdown.lift_rotor_motors)
        
    for k, v in weights_error.items():
        assert (np.abs(v) < 1E-6)    
        
    if save_vtks:
        Results  = Data()
        Results['identical'] = False
        Results['all_prop_outputs'] = results.segments.cruise.conditions.noise.sources.propellers 
        save_vehicle_vtks(configs.base,Results,time_step=1)
        
    if plot_vehicle:
        plot_vehicle(configs.base, save_figure = False, plot_control_points = False)
        
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
    
    # ------------------------------------------------------------------
    #   Cruise Segment: constant Speed, constant altitude
    # ------------------------------------------------------------------ 
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise" 
    segment.analyses.extend(analyses.base)  
    segment.altitude                  = 8012.0  * Units.feet
    segment.air_speed                 = 135. * Units['mph'] 
    segment.distance                  = 20.  * Units.nautical_mile  
    segment.state.unknowns.throttle   = 0.85 *  ones_row(1)
    segment = vehicle.propulsors.battery_propeller.add_unknowns_and_residuals_to_segment(segment)
    
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
