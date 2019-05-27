# Cessna_172.py
# 
# Created:  Mar 2019, M. Clarke

""" setup file for a mission with a twin prop modified C172 SP NAV III with 
propeller interaction 
"""
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt

from SUAVE.Core import Data , Container
from SUAVE.Methods.Propulsion import propeller_design
import sys
sys.path.append('../Vehicles')
from Cessna_172 import configs_setup ,vehicle_setup
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
     
    # lift coefficient  
    lift_coefficient              = results.segments['cruise'].conditions.aerodynamics.lift_coefficient[2][0]
    lift_coefficient_true         = 0.41648757412447335 
    print(lift_coefficient)
    diff_CL                       = np.abs(lift_coefficient  - lift_coefficient_true) 
    print('CL difference')
    print(diff_CL)
    assert np.abs((lift_coefficient  - lift_coefficient_true)/lift_coefficient_true) < 1e-6
    
    # sectional lift coefficient check
    sectional_lift_coeff            = results.segments['cruise'].conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional_lift['main_wing'][2][:]
    sectional_lift_coeff_true       = np.array([0.39688074, 0.39789599, 0.39759175, 0.39598577, 0.39302907,
                                                0.38861399, 0.38256543, 0.37461901, 0.36437972, 0.3512425 ,
                                                0.33422483, 0.31154751, 0.2791921 , 0.21920585, 0.16904757,
                                                0.13540296, 0.12179052, 0.12765071, 0.15139584, 0.19181159,
                                                0.25356333, 0.31326464, 0.37308351, 0.43511796, 0.49967181,
                                                0.56782375, 0.64134112, 0.70135903, 0.74678748, 0.77579888,
                                                0.78678922, 0.77687711, 0.73886236, 0.64294577, 0.59171574,
                                                0.55466008, 0.52468674, 0.49886926, 0.47560758, 0.4538663 ,
                                                0.43288281, 0.41202613, 0.39071187, 0.36833567, 0.34420292,
                                                0.31743009, 0.2867711 , 0.25024397, 0.20411721, 0.13892142])
    print(sectional_lift_coeff)
    diff_Cl                       = np.abs(sectional_lift_coeff - sectional_lift_coeff_true)
    print('Cl difference')
    print(diff_Cl)
    assert  max(np.abs((sectional_lift_coeff - sectional_lift_coeff_true)/sectional_lift_coeff_true)) < 1e-6 
    
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():

    # vehicle data
    vehicle  = vehicle_setup()
    #write(vehicle, 'Cessna_172') 
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
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

    configs, analyses = full_setup()

    simple_sizing(configs)

    configs.finalize()
    analyses.finalize()

    # weight analysis
    weights = analyses.configs.base.weights
    breakdown = weights.evaluate()      

    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()
    
    # Single point analysis
    
    
    # save results 
    #save_results(results, vec_configs)
    
    # plt the old results
    plot_mission(results)


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
    weights = SUAVE.Analyses.Weights.Weights_Tube_Wing()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.process.compute.lift.inviscid_wings.settings.use_surrogate = False
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Stability Analysis
    stability = SUAVE.Analyses.Stability.Fidelity_Zero()    
    #stability.settings.spanwise_vortex_density                  = 3
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
#   Define the Configurations
# ---------------------------------------------------------------------

def configs_setup(vehicle):

    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------

    configs = SUAVE.Components.Configs.Config.Container()

    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)
 


    # done!
    return configs

# ----------------------------------------------------------------------
#   Sizing for the Vehicle Configs
# ----------------------------------------------------------------------
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
    #   Climb 1 : constant Speed, constant rate segment 
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"

    segment.analyses.extend( analyses.base )
    segment.altitude_start = 0.0 * Units.feet
    segment.altitude_end   = 8500. * Units.feet
    segment.air_speed      = 105.  * Units['mph']  
    segment.climb_rate     = 500.  * Units['ft/min']
    
    # add to misison
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Cruise Segment: constant Speed, constant altitude
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    segment.analyses.extend(analyses.base)

    segment.altitude  = 8500. * Units.feet
    segment.air_speed = 132.   *Units['mph']  
    segment.distance  = 50.   * Units.nautical_mile
    
    # add to misison
    mission.append_segment(segment)    
    
    # ------------------------------------------------------------------
    #   Descent Segment: constant Speed, constant rate segment 
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "decent"

    segment.analyses.extend( analyses.base )

    segment.altitude_start = 8500. * Units.feet
    segment.altitude_end   = 0.      * Units.feet
    segment.air_speed      = 80.    * Units['mph'] 
    
    segment.climb_rate     = -300.  * Units['ft/min']
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


if __name__ == '__main__': 
    main()    
    plt.show()