# engine_out.py
# 
# Created:  Feb 2021, T. MacDonald
# Modified: 


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
# Units allow any units to be specificied with SUAVE then automatically converting them the standard
from SUAVE.Core import Units
from SUAVE.Plots.Mission_Plots import * 

# Numpy is use extensively throughout SUAVE
import numpy as np
# Scipy is required here for integration functions used in post processing
import scipy as sp
from scipy import integrate

# Post processing plotting tools are imported here
import pylab as plt

# copy is used to copy variable that should not be linked
# time is used to measure run time if needed
import copy, time

# More basic SUAVE function
from SUAVE.Core import (
Data, Container,
)

import sys
sys.path.append('../Vehicles')
from Concorde import vehicle_setup, configs_setup

# This is a sizing function to fill turbojet parameters
from SUAVE.Methods.Propulsion.turbojet_sizing import turbojet_sizing
from SUAVE.Methods.Center_of_Gravity.compute_fuel_center_of_gravity_longitudinal_range \
     import compute_fuel_center_of_gravity_longitudinal_range
from SUAVE.Methods.Center_of_Gravity.compute_fuel_center_of_gravity_longitudinal_range \
     import plot_cg_map 

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    
    # First we construct the baseline aircraft
    configs, analyses = full_setup()
    
    # Here we finalize the configuration and analysis settings
    configs.finalize()
    analyses.finalize()

    # These functions analyze the mission
    mission = analyses.missions.base
    results = mission.evaluate()
    
    # load older results
    #save_results(results)
    old_results = load_results()   

    # plt the old results
    plot_mission(results)
    plot_mission(old_results,'k-')
    plt.show()

    # check the results
    check_results(results,old_results) 
    
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():
    
    # Vehicle data
    vehicle  = vehicle_setup()
    configs  = configs_setup(vehicle)
    
    # Vehicle analyses
    configs_analyses = analyses_setup(configs)
    
    # Mission analyses
    mission  = mission_setup(configs_analyses)
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
    for tag,config in list(configs.items()):
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
    aerodynamics = SUAVE.Analyses.Aerodynamics.Supersonic_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    aerodynamics.settings.span_efficiency = 0.95
    aerodynamics.settings.engine_out = True
    analyses.append(aerodynamics)
    
    
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
#   Plot Mission
# ----------------------------------------------------------------------

def plot_mission(results,line_style='bo-'):
    
    plot_altitude_sfc_weight(results, line_style) 
    
    plot_flight_conditions(results, line_style) 
    
    plot_aerodynamic_coefficients(results, line_style)  
    
    plot_aircraft_velocities(results, line_style)
    
    plot_drag_components(results, line_style)
    return

# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------
    
def mission_setup(analyses):
    
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
    
    # ------------------------------------------------------------------
    #  Single Point Segment 1: constant Speed, constant altitude
    # ------------------------------------------------------------------ 
    segment = Segments.Single_Point.Set_Speed_Set_Altitude(base_segment)
    segment.tag = "single_point_1" 
    segment.analyses.extend(analyses.base) 
    segment.altitude    =  10000. * Units.feet
    segment.air_speed   =  250.  * Units.kts

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
    
def check_results(new_results,old_results):

    # check segment values
    check_list = [
        'segments.cruise.conditions.aerodynamics.angle_of_attack',
        'segments.cruise.conditions.aerodynamics.drag_coefficient',
        'segments.cruise.conditions.aerodynamics.lift_coefficient',
        'segments.cruise.conditions.propulsion.throttle',
        'segments.cruise.conditions.weights.vehicle_mass_rate',
        'fuel_tank_test.masses',
        'fuel_tank_test.cg_mins',
        'fuel_tank_test.cg_maxes',
    ]

    # do the check
    for k in check_list:
        print(k)

        old_val = np.max( old_results.deep_get(k) )
        new_val = np.max( new_results.deep_get(k) )
        err = (new_val-old_val)/old_val
        print('Error at Max:' , err)
        assert np.abs(err) < 1e-6 , 'Max Check Failed : %s' % k

        old_val = np.min( old_results.deep_get(k) )
        new_val = np.min( new_results.deep_get(k) )
        err = (new_val-old_val)/old_val
        print('Error at Min:' , err)
        assert np.abs(err) < 1e-6 , 'Min Check Failed : %s' % k        

        print('')


    return


def load_results():
    return SUAVE.Input_Output.SUAVE.load('results_mission_concorde.res')

def save_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'results_mission_concorde.res')
    return    
        
if __name__ == '__main__': 
    main()    
    plt.show()