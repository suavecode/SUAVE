# test_solar_network.py
# 
# Created:  Aug 2014, Emilio Botero, 
#           Mar 2020, M. Clarke

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units
from SUAVE.Plots.Mission_Plots import *  
import matplotlib.pyplot as plt  
from SUAVE.Core import (
Data, Container,
)

import numpy as np
import copy, time

from SUAVE.Components.Energy.Networks.Solar import Solar
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_energy_and_power, initialize_from_mass

import sys

sys.path.append('../Vehicles')
# the analysis functions

from Solar_UAV import vehicle_setup, configs_setup

def main():

    # vehicle data
    vehicle  = vehicle_setup()
    configs  = configs_setup(vehicle) 
    configs_analyses = analyses_setup(configs)
    
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
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery       


    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    segment.analyses.extend( configs_analyses.cruise ) 
    
    # segment attributes     
    segment.state.numerics.number_control_points = 50
    segment.dynamic_pressure = 115.0 * Units.pascals
    segment.start_time       = time.strptime("Tue, Jun 21  11:00:00  2017", "%a, %b %d %H:%M:%S %Y",)
    segment.altitude         = 1000.0 * Units.feet
    segment.distance         = 1000.0 * Units.km
    segment.charge_ratio     = 1.0
    segment.latitude         = 37.4
    segment.longitude        = -122.15
    segment.state.conditions.frames.wind.body_rotations[:,2] = 125.* Units.degrees 
    

    # add to mission
    mission.append_segment(segment)


    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses
    
    configs.finalize()
    analyses.finalize()
 
    # mission analysis
    mission = analyses.missions.base    
    results = mission.evaluate()

    # load older results
    #save_results(results)
    old_results = load_results()   

    # plt the old results
    plot_mission(results)
    plot_mission(old_results,'k-')
    plt.show(block=True) 
    
    # Check Results 
    F       = 0 
    rpm     = 0 
    current = 0 
    energy  = 0  
    
    # Truth results
    truth_F   = [[360.43835053, 360.43835053]]
    truth_i   = [[ 249.31622624], [ 249.31622624]]
    truth_rpm = [[6668.4094191],[6668.4094191]]
    truth_bat = [[ 36000000.   ], [ 35987534.18868808]]
    
    error = Data()
    error.Thrust = np.max(np.abs(F[:,0]-truth_F))
    error.RPM = np.max(np.abs(rpm-truth_rpm))
    error.Current  = np.max(np.abs(current-truth_i))
    error.Battery = np.max(np.abs(energy-truth_bat))
    
    print(error)
    
    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)
 
    
    return


# ----------------------------------------------------------------------        
#   Setup Analyses
# ----------------------------------------------------------------------  

def analyses_setup(configs):
    
    analyses = SUAVE.Analyses.Analysis.Container()
    
    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config)
        analyses[tag] = analysis
    
    return analyses

# ----------------------------------------------------------------------        
#   Define Base Analysis
# ----------------------------------------------------------------------  

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
    # ------------------------------------------------------------------ 
    weights = SUAVE.Analyses.Weights.Weights_UAV()
    weights.settings.empty_weight_method = SUAVE.Methods.Weights.Correlations.UAV.empty
    weights.vehicle = vehicle
    analyses.append(weights)
    
    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    # ------------------------------------------------------------------ 
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    analyses.append(aerodynamics)
    
    # ------------------------------------------------------------------
    #  Energy
    # ------------------------------------------------------------------ 
    energy = SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.propulsors
    analyses.append(energy)
    
    # ------------------------------------------------------------------
    #  Planet Analysis
    # ------------------------------------------------------------------ 
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)
    
    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    # ------------------------------------------------------------------ 
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)   
    
    return analyses    


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
#   Plot Mission
# ----------------------------------------------------------------------

def plot_mission(results,line_style='bo-'):     
    
    # Plot Propeller Performance 
    plot_propeller_conditions(results,line_style)
    
    # Plot Power and Disc Loading
    plot_disc_power_loading(results,line_style)
    
    # Plot Solar Radiation Flux
    plot_solar_flux(results,line_style) 
    return


# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    plt.show()
