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
import time
import sys

from SUAVE.Plots.Mission_Plots import *  
from SUAVE.Plots.Geometry_Plots.plot_vehicle import plot_vehicle  
from SUAVE.Input_Output.VTK.save_vehicle_vtk import save_vehicle_vtks
from SUAVE.Methods.Weights.Buildups.eVTOL.empty import empty 

sys.path.append('../Vehicles') 
from X57_Maxwell_Mod4 import vehicle_setup, configs_setup 

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
        
    ## Single cruise segment test cases (with and without wakes):
    #results_no_wake, configs = test_case_1(FHW=False, save_loc = "Case_1_No_Wakes/")
    #results_wake, configs    = test_case_1(FHW=True, save_loc = "Case_1_Wakes/")
    
    
    #plot_both_lift_distributions(results_no_wake,results_wake,configs.cruise, title="Cruise Mission Segment")
    #plt.show()
    
    ## Climb segment test cases (with and without wakes):
    #results_no_wake, configs = test_case_2(FHW=False, save_loc = "Case_2_No_Wakes/")
    #results_wake, configs    = test_case_2(FHW=True, save_loc = "Case_2_Wakes/")
    
    #plot_both_lift_distributions(results_no_wake,results_wake,configs.climb, title="Climb Mission Segment")
    #plt.show() 

    # Takeoff segment test cases (with and without wakes):
    results_no_wake, configs = test_case_3(FHW=False, save_loc = "Case_3_No_Wakes/")
    results_wake, configs    = test_case_3(FHW=True, save_loc = "Case_3_Wakes/") 
    
    plot_both_lift_distributions(results_no_wake,results_wake,configs.takeoff, title="Takeoff Mission Segment")
    plt.show() 
    
    return 

def test_case_1(FHW,save_loc):
    """
    Test Case 1: Runs the X57 Mod 4 for a single cruise segment, operating
    just the tip propellers with or without wake interaction.
    """
    if FHW:
        print("\nTest case 1: Cruise segment (with helical fixed-wakes).")
    else:
        print("\nTest case 1: Cruise segment (no wakes).")
    save_vtks    = True
    
    ti = time.time()
    results,configs = run_analysis(save_vtks,save_loc,FHW,mission_case=1)
    t_no_wake = (time.time() - ti)/60
    print("TIME:")
    print(t_no_wake)
    
    print_key_results(results)    
    
    
    return results, configs

def test_case_2(FHW,save_loc):
    """
    Test Case 2: Runs the X57 Mod 4 for a single climb segment, operating
    just the tip propellers with fixed helical wake interaction.
    """
    if FHW:
        print("\nTest case 2: Climb segment (with helical fixed-wakes).")
    else:
        print("\nTest case 2: Climb segment (no wakes).")
    save_vtks    = True
    
    ti = time.time()
    results,configs = run_analysis(save_vtks,save_loc,FHW,mission_case=2)
    t_no_wake = (time.time() - ti)/60
    print("TIME:")
    print(t_no_wake)
    
    print_key_results(results)    
         
    
    return results, configs


def test_case_3(FHW,save_loc):
    """
    Test Case 3: Runs the X57 Mod 4 for a single takeoff segment, operating
    tip props and distributed props for lift augmentation on takeoff/landing.
    """
    if FHW:
        print("\nTest case 3: Takeoff segment (with helical fixed-wakes).")
    else:
        print("\nTest case 3: Takeoff segment (no wakes).")
    save_vtks    = True
    
    ti = time.time()
    results,configs = run_analysis(save_vtks,save_loc,FHW,mission_case=3)
    t_no_wake = (time.time() - ti)/60
    print("TIME:")
    print(t_no_wake)
    
    print_key_results(results)    
         
    
    return results, configs


def print_key_results(results):
    # Extract mission segment results
    #rpms = results.segments.cruise.conditions.propulsion.propeller_rpm
    #print('propeller rpms')
    #print(rpms)
    for segment in results.segments.values():
        key = segment.tag
    
        throttle = results.segments[key].conditions.propulsion.throttle
        print('throttle')
        print(throttle)    
    
        # lift coefficient  
        lift_coefficient  = results.segments[key].conditions.aerodynamics.lift_coefficient[0][0]
        print('lift coefficient')
        print(lift_coefficient)
        
        # drag coefficient  
        drag_coefficient  = results.segments[key].conditions.aerodynamics.drag_coefficient[0][0]
        print('drag coefficient')
        print(drag_coefficient)    
        
        # angle of attack  
        aoa  = results.segments[key].conditions.aerodynamics.angle_of_attack[0][0]
        print('angle of attack  ')
        print(aoa/Units.deg)   
        
               
        
    

    return  


def run_analysis(save_vtks,save_loc,fixed_helical_wake,mission_case):
    # Evaluate vehicle during mission (option to use helical fixed-wake model)
    configs, analyses  = full_setup(fixed_helical_wake,mission_case) 

    configs.finalize()
    analyses.finalize()  

    # mission analysis
    mission = analyses.missions.base
    print("\nEvaluating the mission...")
    results = mission.evaluate()
    
    if save_vtks:
        # save segment vtks
        for segment in results.segments.values():
            save_vehicle_vtks(configs[segment.tag], results, time_step=0, save_loc=save_loc+segment.tag+"/")
        
    return results, configs

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup(fixed_helical_wake,mission_case):

    # vehicle data
    vehicle  = vehicle_setup() 
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs, fixed_helical_wake)

    # mission analyses
    if mission_case ==1:
        # Run single segment cruise mission
        mission  = cruise_mission_setup(configs_analyses,configs) 
    
    elif mission_case ==2:
        # Run single segment climb mission
        mission  = climb_mission_setup(configs_analyses,configs) 
    elif mission_case==3:
        # Run single segment takeoff mission
        mission = takeoff_mission_setup(configs_analyses,configs)
        
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
        aerodynamics.settings.propeller_wake_model       = True   
        #aerodynamics.settings.time_averaged_wake         = True
        aerodynamics.settings.use_bemt_wake_model        = False      
        
    aerodynamics.settings.use_surrogate              = False
        
    aerodynamics.settings.number_spanwise_vortices   = 25
    aerodynamics.settings.number_chordwise_vortices  = 5
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


# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------

def cruise_mission_setup(analyses,configs):
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'mission' 

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
    segment.analyses.extend(analyses.cruise) 
    segment.battery_energy                 = configs.cruise.networks.battery_propeller.battery.max_energy* 0.89
        
    segment.altitude                  = 8000.  * Units.feet
    segment.air_speed                 = 135. * Units['mph'] 
    segment.distance                  = 10.  * Units.nautical_mile  
    segment.state.unknowns.throttle   = 0.65 *  ones_row(1)
    segment = configs.cruise.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)
    
    # add to misison
    mission.append_segment(segment)        
    
    return mission



def climb_mission_setup(analyses,configs):
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'mission' 

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments 
    
    # base segment
    base_segment = Segments.Segment()
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
    base_segment.state.numerics.number_control_points        = 2
    
    # ------------------------------------------------------------------
    #   Climb Segment
    # ------------------------------------------------------------------ 
    
    segment                                = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag                            = "climb"
    segment.analyses.extend( analyses.climb )
    segment.battery_energy                 = configs.climb.networks.battery_propeller.battery.max_energy* 0.89
    
    segment.air_speed                      = 110 * Units.mph
    segment.altitude_start                 = 5000.0 * Units.ft
    segment.altitude_end                   = 8000. * Units.ft
    segment.climb_rate                     = 500. * Units['ft/min']
    segment = configs.climb.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)

    # add to misison
    mission.append_segment(segment)
    
   
    
    return mission



def takeoff_mission_setup(analyses,configs):
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'mission' 

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments 
    
    # base segment
    base_segment = Segments.Segment()
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
    base_segment.state.numerics.number_control_points        = 2
    
    # ------------------------------------------------------------------
    #   Takeoff/Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "takeoff"

    # connect vehicle configuration
    segment.analyses.extend( analyses.takeoff )
    segment.battery_energy = configs.takeoff.networks.battery_propeller.battery.max_energy* 0.89

    segment.altitude_start = 0.0  * Units.feet
    segment.altitude_end   = 1000 * Units.feet
    segment.air_speed      = 80.0 * Units['mph']
    segment.climb_rate     = 500. * Units['ft/min']

    segment = configs.takeoff.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)
    
    # add to misison
    mission.append_segment(segment)    
   
    
    return mission


def climb_plus_cruise_mission_setup(analyses,configs):
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'mission' 

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments 
    
    # base segment
    base_segment = Segments.Segment()
    ones_row     = base_segment.state.ones_row
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
    base_segment.state.numerics.number_control_points        = 2
    
    ## ------------------------------------------------------------------
    ##   Climb Segment
    ## ------------------------------------------------------------------ 
    
    #segment                                = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    #segment.tag                            = "climb"
    #segment.analyses.extend( analyses )
    #segment.battery_energy                 = vehicle.networks.battery_propeller.battery.max_energy* 0.89
    
    #segment.air_speed                      = 100
    #segment.altitude_start                 = 5000.0 * Units.ft
    #segment.altitude_end                   = 8000. * Units.ft
    #segment.climb_rate                     = 500. * Units['ft/min']
    #segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)

    ## add to misison
    #mission.append_segment(segment)
    

    # ------------------------------------------------------------------
    #   Cruise Segment: constant Speed, constant altitude
    # ------------------------------------------------------------------     
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise" 
    segment.analyses.extend(analyses.cruise) 
    segment.battery_energy                 = configs.cruise.networks.battery_propeller.battery.max_energy* 0.89
        
    segment.altitude                  = 8000.  * Units.feet
    segment.air_speed                 = 135. * Units['mph'] 
    segment.distance                  = 10.  * Units.nautical_mile  
    segment.state.unknowns.throttle   = 0.65 *  ones_row(1)
    segment = configs.cruise.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)
    
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


def plot_both_lift_distributions(results_1,results_2,vehicle, save_figure = False, save_filename = "Sectional_Lift", file_type = ".png", title=None):
    """This plots the sectional lift distrubtion at all control points
    on all lifting surfaces of the aircraft

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.aerodynamics.
        inviscid_wings_sectional_lift
    vehicle.vortex_distribution.
       n_sw
       n_w
       
    Outputs: 
    Plots

    Properties Used:
    N/A	
    """ 

    VD         = vehicle.vortex_distribution	 	
    n_w        = VD.n_w
    b_sw       = np.concatenate(([0],np.cumsum(VD.n_sw)))
    
    line_no_wakes = ['-b','-g']
    line_wakes    = ['-r','-k']
    
    axis_font  = {'size':'12'}  	
    img_idx    = 1
    seg_idx    = 1
    
    
    for segment in results_1.segments.values():   
        segment_wakes = results_2.segments[segment.tag]
        num_ctrl_pts = len(segment.conditions.frames.inertial.time)	
        
        for ti in range(num_ctrl_pts): 
            # Setup figure for this control point
            fig1  = plt.figure()
            fig1.set_size_inches(8,5)       
            axes1 = plt.subplot(1,1,1)  
            
            fig2  = plt.figure()
            fig2.set_size_inches(8,5)       
            axes2 = plt.subplot(1,1,1)             
            
            # Plot for no wake interaction case:
            cl_y = segment.conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional[ti] 
            cd_y = segment.conditions.aerodynamics.drag_breakdown.induced.wings_sectional[ti] 
            
            for i in range(n_w): 
                if i==0:
                    # only label first 
                    label = "No Wake Interactions"
                else:
                    label = None
                    
                y_pts = VD.Y_SW[b_sw[i]:b_sw[i+1]]
                z_pts = cl_y[b_sw[i]:b_sw[i+1]]
                axes1.plot(y_pts, z_pts, line_no_wakes[0],label=label) 
                
                y_pts = VD.Y_SW[b_sw[i]:b_sw[i+1]]
                z_pts = cd_y[b_sw[i]:b_sw[i+1]]
                axes2.plot(y_pts, z_pts, line_no_wakes[0],label=label)   
            
            # Repeat for no-wakes case:

            cl_y_wakes = segment_wakes.conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional[ti] 
            cd_y_wakes = segment_wakes.conditions.aerodynamics.drag_breakdown.induced.wings_sectional[ti] 
            
            for i in range(n_w): 
                if i==0:
                    # only label first 
                    label = "Wake Interactions"
                else:
                    label = None
                    
                y_pts = VD.Y_SW[b_sw[i]:b_sw[i+1]]
                z_pts = cl_y_wakes[b_sw[i]:b_sw[i+1]]
                axes1.plot(y_pts, z_pts, line_wakes[0],label=label) 
                
                y_pts = VD.Y_SW[b_sw[i]:b_sw[i+1]]
                z_pts = cd_y_wakes[b_sw[i]:b_sw[i+1]]
                axes2.plot(y_pts, z_pts, line_wakes[0],label=label)  
                
            axes1.set_xlabel("Spanwise Location (m)",axis_font)
            axes1.set_ylabel('$C_{l,y}$',axis_font)  
            axes1.set_title(title,axis_font)  
            axes2.set_xlabel("Spanwise Location (m)",axis_font)
            axes2.set_ylabel('$C_{d,y}$',axis_font)    
            axes2.set_title(title,axis_font)   
            axes1.legend()
            axes2.legend()
                    
            if save_figure: 
                plt.savefig( save_filename + '_' + str(img_idx) + file_type) 	
            img_idx += 1
        seg_idx +=1
        
    return   

def check_regression_with_wakes(results,vehicle):
    # Extract mission segment results
    rpms = results.segments.cruise.conditions.propulsion.propeller_rpm
    print('propeller rpms')
    print(rpms)
    
    throttle = results.segments.cruise.conditions.propulsion.throttle
    print('throttle')
    print(throttle)    

    # lift coefficient  
    lift_coefficient              = results.segments.cruise.conditions.aerodynamics.lift_coefficient[0][0]
    lift_coefficient_true         = 0.41450988979513925
    diff_CL                       = np.abs(lift_coefficient  - lift_coefficient_true)
    print(lift_coefficient)
    print('CL difference')
    print(diff_CL)

    # sectional lift coefficient check
    sectional_lift_coeff            = results.segments.cruise.conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional[0]
    sectional_lift_coeff_true       = np.array([8.56334013e-01, 7.94590750e-01, 7.01447643e-01, 5.14977671e-01,
                                                2.48710480e-01, 8.56334024e-01, 7.94590976e-01, 7.01447848e-01,
                                                5.14977732e-01, 2.48710035e-01, 2.74565991e-01, 2.94768343e-01,
                                                2.91913493e-01, 2.50056000e-01, 1.62220430e-01, 2.74565999e-01,
                                                2.94768403e-01, 2.91913642e-01, 2.50056310e-01, 1.62220954e-01,
                                                6.62977658e-16, 2.44262719e-15, 3.30007366e-15, 3.06533733e-15,
                                                2.06775326e-15])


    plot_lift_distribution(results,vehicle)
    diff_Cl                         = np.abs(sectional_lift_coeff - sectional_lift_coeff_true)
    
    
    
    print(sectional_lift_coeff)
    print('Cl difference')
    print(diff_Cl)
    
    #assert np.abs(lift_coefficient  - lift_coefficient_true) < 1e-5
    #assert np.max(np.abs(sectional_lift_coeff - sectional_lift_coeff_true)) < 1e-5

    return    


if __name__ == '__main__': 
    main()    
    plt.show()
