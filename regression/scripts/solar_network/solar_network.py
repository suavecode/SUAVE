# test_solar_network.py
# 
# Created:  Aug 2014, Emilio Botero, 
#           Mar 2020, M. Clarke
#           Apr 2020, M. Clarke

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units
from SUAVE.Plots.Performance.Mission_Plots import *  
import matplotlib.pyplot as plt  
from SUAVE.Core import (
Data, Container,
)

import numpy as np
import copy, time

from SUAVE.Components.Energy.Networks.Solar import Solar
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_energy_and_power, initialize_from_mass
from SUAVE.Plots.Geometry.plot_vehicle import plot_vehicle 
import sys

sys.path.append('../Vehicles')
# the analysis functions

from Solar_UAV import vehicle_setup, configs_setup

def main():
    
 
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
    
    configs.finalize()
    analyses.finalize()    
    
    # weight analysis
    weights = analyses.configs.base.weights    
    
    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()

    # load older results
    #save_results(results)
    old_results = load_results()   

    ## plt the old results
    plot_mission(results)
    plot_mission(old_results,'k-') 
    
    # Check Results 
    F       = results.segments.cruise1.conditions.frames.body.thrust_force_vector[1,0]
    rpm     = results.segments.cruise1.conditions.propulsion.propeller_rpm[1,0]
    current = results.segments.cruise1.conditions.propulsion.battery_current[1,0]
    energy  = results.segments.cruise1.conditions.propulsion.battery_energy[8,0]  
    
    # Truth results

    truth_F   = 105.36115293829462
    truth_rpm = 218.18739964349612
    truth_i   = 130.17994767726535
    truth_bat = 136584698.345862
    
    print('battery energy')
    print(energy)
    print('\n')
    
    error = Data()
    error.Thrust   = np.max(np.abs((F-truth_F)/truth_F))
    error.RPM      = np.max(np.abs((rpm-truth_rpm)/truth_rpm))
    error.Current  = np.max(np.abs((current-truth_i)/truth_i))
    error.Battery  = np.max(np.abs((energy-truth_bat)/truth_bat))
    
    print(error)
    
    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)
 
    # Plot vehicle 
    plot_vehicle(configs.cruise, save_figure = False, plot_control_points = True)
    
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

def base_analysis(vehicle): # ------------------------------------------------------------------
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
    weights = SUAVE.Analyses.Weights.Weights_UAV()
    weights.settings.empty_weight_method = \
        SUAVE.Methods.Weights.Correlations.Human_Powered.empty
    weights.vehicle = vehicle
    analyses.append(weights)
    
    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero() 
    aerodynamics.geometry                            = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    aerodynamics.settings.span_efficiency = 0.98
    analyses.append(aerodynamics)
    
    # ------------------------------------------------------------------
    #  Energy
    energy = SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.networks #what is called throughout the mission (at every time step))
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


# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------
def mission_setup(analyses,vehicle):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'The Test Mission'

    mission.atmosphere  = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    mission.planet      = SUAVE.Attributes.Planets.Earth()
    
    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments
    
    # base segment
    base_segment = Segments.Segment()
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Analyses.Mission.Segments.Cruise.Constant_Mach_Constant_Altitude(base_segment)
    segment.tag = "cruise1"
    
    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise)
    
    # segment attributes     
    segment.start_time     = time.strptime("Tue, Jun 21 11:30:00  2020", "%a, %b %d %H:%M:%S %Y",)
    segment.altitude       = 15.0  * Units.km 
    segment.mach           = 0.12
    segment.distance       = 3050.0 * Units.km
    segment.battery_energy = vehicle.networks.solar.battery.max_energy*0.3 #Charge the battery to start
    segment.latitude       = 37.4300   # this defaults to degrees (do not use Units.degrees)
    segment.longitude      = -122.1700 # this defaults to degrees
    segment = vehicle.networks.solar.add_unknowns_and_residuals_to_segment(segment,initial_power_coefficient = 0.05)    
    
    
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



def load_results():
    return SUAVE.Input_Output.SUAVE.load('solar_uav_mission.res')

def save_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'solar_uav_mission.res')
    return


# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    plt.show()
