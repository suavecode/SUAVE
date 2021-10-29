# mission_B737.py
# 
# Created:  Aug 2014, SUAVE Team
# Modified: Jun 2016, T. MacDonald
#           May 2019, T. MacDonald 
#           Mar 2020, M. Clarke
#           Apr 2020, M. Clarke
#           Apr 2020, E. Botero
#           Aug 2021, R. Erhard

""" setup file for a mission with a 737
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units
from SUAVE.Plots.Performance.Mission_Plots import *
from SUAVE.Plots.Geometry import * 
import matplotlib.pyplot as plt  
import numpy as np 

from SUAVE.Methods.Center_of_Gravity.compute_component_centers_of_gravity import compute_component_centers_of_gravity

import sys

sys.path.append('../Vehicles')
# the analysis functions

from Boeing_737 import vehicle_setup, configs_setup



from SUAVE.Input_Output.Results import  print_parasite_drag,  \
     print_compress_drag, \
     print_engine_data,   \
     print_mission_breakdown, \
     print_weight_breakdown
# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():

    configs, analyses = full_setup()

    simple_sizing(configs, analyses)
    configs.finalize()
    analyses.finalize() 
 
    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()

    # load older results
    #save_results(results)
    old_results = load_results()   
    
    # check the results
    check_results(results,old_results) 
    
    # plt the old results
    plot_mission(results)
    plot_mission(old_results,'k-')
    #plt.show(block=True)    
    
    # print weights breakdown
    print_weight_breakdown(configs.cruise)
    
    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------
    
    # plot vehicle 
    plot_vehicle(configs.base,plot_control_points = True)      
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():

    # vehicle data
    vehicle  = vehicle_setup()
    print_wing_segs(vehicle)
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

    analyses = SUAVE.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in list(configs.items()):
        analysis = base_analysis(config)
        analyses[tag] = analysis

    # adjust analyses for configs

    # takeoff_analysis
    analyses.takeoff.aerodynamics.settings.drag_coefficient_increment = 0.0000

    # landing analysis
    aerodynamics = analyses.landing.aerodynamics
    # do something here eventually

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
    aerodynamics.geometry                            = vehicle
    aerodynamics.settings.use_surrogate              = True
    aerodynamics.settings.number_spanwise_vortices   = 5
    aerodynamics.settings.number_chordwise_vortices  = 2    
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Stability Analysis
    stability = SUAVE.Analyses.Stability.Fidelity_Zero()
    stability.geometry = vehicle
    analyses.append(stability)

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

    # done!
    return analyses    

# ----------------------------------------------------------------------
#   Plot Mission
# ----------------------------------------------------------------------

def plot_mission(results,line_style='bo-'):
    
    # Plot Flight Conditions 
    plot_flight_conditions(results, line_style)
    
    # Plot Aerodynamic Forces 
    plot_aerodynamic_forces(results, line_style)
    
    # Plot Aerodynamic Coefficients 
    plot_aerodynamic_coefficients(results, line_style)
    
    # Plot Static Stability Coefficients 
    plot_stability_coefficients(results, line_style)    
    
    # Drag Components
    plot_drag_components(results, line_style)
    
    # Plot Altitude, sfc, vehicle weight 
    plot_altitude_sfc_weight(results, line_style)
    
    # Plot Velocities 
    plot_aircraft_velocities(results, line_style)  

    return

def simple_sizing(configs, analyses):

    base = configs.base
    base.pull_base()

    # zero fuel weight
    base.mass_properties.max_zero_fuel = 0.9 * base.mass_properties.max_takeoff 

    # wing areas
    for wing in base.wings:
        wing.areas.wetted   = 2.0 * wing.areas.reference
        wing.areas.exposed  = 0.8 * wing.areas.wetted
        wing.areas.affected = 0.6 * wing.areas.wetted

    # fuselage seats
    base.fuselages['fuselage'].number_coach_seats = base.passengers
    
    # weight analysis
    #need to put here, otherwise it won't be updated
    weights = analyses.configs.base.weights
    breakdown = weights.evaluate(method='New SUAVE')    
    
    #compute centers of gravity
    #need to put here, otherwise, results won't be stored
    compute_component_centers_of_gravity(base)
    base.center_of_gravity()
    
    # diff the new data
    base.store_diff()

    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------
    landing = configs.landing

    # make sure base data is current
    landing.pull_base()

    # landing weight
    landing.mass_properties.landing = 0.85 * base.mass_properties.takeoff

    # diff the new data
    landing.store_diff()

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
    #   First Climb Segment: Constant Speed Constant Rate  
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"

    segment.analyses.extend( analyses.takeoff )

    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 3.0   * Units.km
    segment.air_speed      = 125.0 * Units['m/s']
    segment.climb_rate     = 6.0   * Units['m/s']

    # add to misison
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Second Climb Segment: Constant Speed Constant Rate  
    # ------------------------------------------------------------------    

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_2"

    segment.analyses.extend( analyses.cruise )

    segment.altitude_end   = 8.0   * Units.km
    segment.air_speed      = 190.0 * Units['m/s']
    segment.climb_rate     = 6.0   * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Third Climb Segment: Constant Speed Constant Rate  
    # ------------------------------------------------------------------    

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_3"

    segment.analyses.extend( analyses.cruise )

    segment.altitude_end = 10.5   * Units.km
    segment.air_speed    = 226.0  * Units['m/s']
    segment.climb_rate   = 3.0    * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------    
    #   Cruise Segment: Constant Speed Constant Altitude
    # ------------------------------------------------------------------    

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    segment.analyses.extend( analyses.cruise )

    segment.altitude  = 10.668 * Units.km # small jump to test altitude updating
    segment.air_speed = 230.412 * Units['m/s']
    segment.distance  = (3933.65 + 770 - 92.6) * Units.km
    
    segment.state.numerics.number_control_points = 10
    
    # post-process aerodynamic derivatives in cruise
    segment.process.finalize.post_process.aero_derivatives = SUAVE.Methods.Flight_Dynamics.Static_Stability.compute_aero_derivatives

    # add to mission
    mission.append_segment(segment)


# ------------------------------------------------------------------
#   First Descent Segment: Constant Speed Constant Rate  
# ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_1"

    segment.analyses.extend( analyses.cruise )

    segment.altitude_start = 10.5 * Units.km # small jump to test altitude updating
    segment.altitude_end   = 8.0   * Units.km
    segment.air_speed      = 220.0 * Units['m/s']
    segment.descent_rate   = 4.5   * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Second Descent Segment: Constant Speed Constant Rate  
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_2"

    segment.analyses.extend( analyses.landing )

    analyses.landing.aerodynamics.settings.spoiler_drag_increment = 0.00

    segment.altitude_end = 6.0   * Units.km
    segment.air_speed    = 195.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Third Descent Segment: Constant Speed Constant Rate  
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_3"

    segment.analyses.extend( analyses.landing )

    analyses.landing.aerodynamics.settings.spoiler_drag_increment = 0.00

    segment.altitude_end = 4.0   * Units.km
    segment.air_speed    = 170.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Fourth Descent Segment: Constant Speed Constant Rate  
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_4"

    segment.analyses.extend( analyses.landing )

    analyses.landing.aerodynamics.settings.spoiler_drag_increment = 0.00

    segment.altitude_end = 2.0   * Units.km
    segment.air_speed    = 150.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']


    # add to mission
    mission.append_segment(segment)



    # ------------------------------------------------------------------
    #   Fifth Descent Segment:Constant Speed Constant Rate  
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_5"

    segment.analyses.extend( analyses.landing )
    analyses.landing.aerodynamics.settings.spoiler_drag_increment = 0.00


    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 145.0 * Units['m/s']
    segment.descent_rate = 3.0   * Units['m/s']


    # append to mission
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

    # ------------------------------------------------------------------
    #   Mission for Constrained Fuel
    # ------------------------------------------------------------------    
    fuel_mission         = SUAVE.Analyses.Mission.Mission() #Fuel_Constrained()
    fuel_mission.tag     = 'fuel'
    fuel_mission.range   = 1277. * Units.nautical_mile
    fuel_mission.payload = 19000.
    missions.append(fuel_mission)    

    # ------------------------------------------------------------------
    #   Mission for Constrained Short Field
    # ------------------------------------------------------------------    
    short_field = SUAVE.Analyses.Mission.Mission(base_mission) #Short_Field_Constrained()
    short_field.tag = 'short_field'    

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    airport.available_tofl = 1500.
    short_field.airport = airport    
    missions.append(short_field)

    # ------------------------------------------------------------------
    #   Mission for Fixed Payload
    # ------------------------------------------------------------------    
    payload = SUAVE.Analyses.Mission.Mission() #Payload_Constrained()
    payload.tag     = 'payload'
    payload.range   = 2316. * Units.nautical_mile
    payload.payload = 19000.
    missions.append(payload)

    # done!
    return missions  

def check_results(new_results,old_results):

    # check segment values
    check_list = [
        'segments.cruise.conditions.aerodynamics.angle_of_attack',
        'segments.cruise.conditions.aerodynamics.drag_coefficient',
        'segments.cruise.conditions.aerodynamics.lift_coefficient',
        'segments.cruise.conditions.stability.static.Cm_alpha',
        'segments.cruise.conditions.stability.static.Cn_beta',
        'segments.cruise.conditions.propulsion.throttle',
        'segments.cruise.conditions.weights.vehicle_mass_rate',
        'segments.cruise.conditions.aero_derivatives.dCL_dAlpha',
        'segments.cruise.conditions.aero_derivatives.dCn_dBeta'
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

    ## check high level outputs
    #def check_vals(a,b):
        #if isinstance(a,Data):
            #for k in a.keys():
                #err = check_vals(a[k],b[k])
                #if err is None: continue
                #print 'outputs' , k
                #print 'Error:' , err
                #print ''
                #assert np.abs(err) < 1e-6 , 'Outputs Check Failed : %s' % k  
        #else:
            #return (a-b)/a

    ## do the check
    #check_vals(old_results.output,new_results.output)


    return



def print_wing_segs(vehicle):
    
    """This prints the wing segments to provide coverage for dataordered printing"""
    
    print(vehicle.wings.main_wing.Segments)
    
    return


def load_results():
    return SUAVE.Input_Output.SUAVE.load('results_mission_B737.res')

def save_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'results_mission_B737.res')
    return

if __name__ == '__main__': 
    main()    
    plt.show()
