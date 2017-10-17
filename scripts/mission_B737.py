# mission_B737.py
# 
# Created:  Oct 2017, SUAVE Team

# Use the variables in the "Analysis Controls" block in the main function
# to run an analyses with different years and technology settings.

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np
import pylab as plt
import copy
import sys

import SUAVE
from SUAVE.Core import (
Units, Data, Container,
)
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing
from SUAVE.Methods.Center_of_Gravity.compute_component_centers_of_gravity import compute_component_centers_of_gravity
from SUAVE.Methods.Center_of_Gravity.compute_aircraft_center_of_gravity import compute_aircraft_center_of_gravity

from Boeing_737 import vehicle_setup, configs_setup

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    
    # -----------------------------------------------------------
    # Analysis Controls
    # -----------------------------------------------------------

    # 2027 or 2037 (or 2017 for no change)
    year = 2037
    # NLF or HLF (laminar flow type)
    LF_type = 'NLF'
    # metal or composite
    fuselage_material = 'composite'
    
    # -----------------------------------------------------------
    #
    # -----------------------------------------------------------
    
    LD_factor    = find_LD_factor(year,LF_type)
    wt_factors   = weight_factors(year,fuselage_material)
    prop_factors = propulsion_factors(year)

    configs, analyses = full_setup(LD_factor,wt_factors,prop_factors)

    simple_sizing(configs, analyses)

    configs.finalize()
    analyses.finalize()

    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()
    
    # Post process for emissions
    results = post_process_emissions(configs, results)

    # load older results
    #save_results(results)
    old_results = load_results()   

    # plt the old results
    plot_mission(results)
    plot_mission(old_results,'k-')
    plt.show(block=True)
    # check the results
    check_results(results,old_results)
    
   

    return

def find_LD_factor(year,LF_type):
    
    if year == 2017:
        return 0.
    
    LD_factors = dict()
    # values are single aisle (SA), [1,.25,40,60] corresponds to 1% +- .25% with 40% chance in 2027 and 60% in 2037
    LD_factors['Riblets']               = [1,.25,40,60]
    LD_factors['Excrescence Reduction'] = [.4,.4,60,90]
    if LF_type == 'NLF': # Natural Laminar Flow
        LD_factors['LF - Nacelle']         = [.5,.25,60,90]
        LD_factors['LF - Wing']            = [3,1.5,40,60]
        LD_factors['LF - HTail']           = [.25,.25,10,40]
        LD_factors['LF - VTail']           = [.25,.25,10,40]
    elif LF_type == 'HLF': # Hybrid Laminar Flow
        LD_factors['LF - Nacelle']         = [.6,.25,10,40]
        LD_factors['LF - Wing']            = [3,1.5,10,40]
        LD_factors['LF - HTail']           = [.5,.25,10,40]
        LD_factors['LF - VTail']           = [.5,.25,40,60]
    else:
        raise ValueError('Use NLF or HLF for laminar flow type setting')
    LD_factors['Advanced Winglets']     = [1,1,90,90]
    LD_factors['TE Variable Camber']    = [.5,.5,40,90]
    LD_factors['Active CG Control']     = [.5,.5,20,30]
    LD_factors['Active Flow Control']   = [.2,.2,20,50]
    
    LD_increase = 0
    for tech,vals in LD_factors.iteritems():
        if year == 2027:
            tech_factor = vals[0]/100.*vals[2]/100.
        elif year == 2037:
            tech_factor = vals[0]/100.*vals[3]/100.
        else:
            raise ValueError('Year must be 2027 or 2037')
        LD_increase += tech_factor
        
    return LD_increase

def weight_factors(year,fuselage_material):
    
    wt_factors = Data()
    wt_factors.main_wing = 0
    wt_factors.fuselage  = 0
    wt_factors.empennage = 0    
    
    if year == 2017:
        return wt_factors
    
    wing_tech      = dict()
    fuselage_tech  = dict()
    empennage_tech = dict()
    
    wing_tech['Advanced Composites'] = [10,2,80,100]
    wing_tech['Opt Local Design']    = [2,1,40,70]
    wing_tech['Multifunc Materials'] = [2,1,40,70]
    wing_tech['Load Alleviation']    = [1.5,1,50,80]
    
    if fuselage_material == 'metal':
        fuselage_tech['Advanced Metals'] = [4,2,60,90]
    elif fuselage_material == 'composite':
        fuselage_tech['Advanced Composites'] = [8,3,80,100]
    else:
        raise ValueError('Fuselage material must be metal or composite')
    fuselage_tech['Opt Local Design']    = [2,1,40,70]
    fuselage_tech['Multifunc Materials'] = [3,1,40,70]
    fuselage_tech['Load Alleviation']    = [.5,.5,30,60]
    
    empennage_tech['Advanced Composites'] = [8,2,80,100]
    empennage_tech['Opt Local Design']    = [2,1,40,70]
    empennage_tech['Multifunc Materials'] = [2,1,40,70]
    empennage_tech['Load Alleviation']    = [.5,.5,50,80]    
    
    weight_techs = dict()
    weight_techs['main_wing'] = wing_tech
    weight_techs['fuselage']  = fuselage_tech
    weight_techs['empennage'] = empennage_tech
    
    for component,comp_dict in weight_techs.iteritems():
        for tech,vals in comp_dict.iteritems():
            if year == 2027:
                tech_factor = vals[0]/100.*vals[2]/100.
            elif year == 2037:
                tech_factor = vals[0]/100.*vals[3]/100.
            else:
                raise ValueError('Year must be 2027 or 2037')
            wt_factors[component] += tech_factor    
            
    return wt_factors


def propulsion_factors(year):
    # values are single aisle (SA), [1,.25,40,60] corresponds to 1% +- .25% with 40% chance in 2027 and 60% in 2037
    
    # How to incorporate CO2 changes, these are stoichiometric from the slides?
    
    if   year == 2017:
        prop_factors = 0.16 * 1.0
    elif year == 2027:
        prop_factors = 0.16 + 0.04 * 0.60
    elif year == 2037:
        prop_factors = 0.16 + 0.04 * 0.60 + ((1+0.6/100.)**10.-1.) * 0.60 
    else:
        raise ValueError('Year must be 2017, 2027, or 2037')    
        
    return prop_factors


def post_process_emissions(configs,results):
    
    # Unpack
    turbofan_emmissions  = SUAVE.Methods.Propulsion.turbofan_emission_index
    turbofan = configs.base.propulsors.turbofan
    
    CO2_total = 0. 
    SO2_total = 0.
    NOx_total = 0.
    H2O_total = 0.
    for segment in results.segments.values():
        
        # Find the emissions indices
        emissions = turbofan_emmissions(turbofan,segment)
        segment.conditions.emissions = emissions
        
        # Add up the total emissions from all segments
        CO2_total = CO2_total + emissions.total.CO2[-1]
        SO2_total = SO2_total + emissions.total.SO2[-1]
        NOx_total = NOx_total + emissions.total.NOx[-1]
        H2O_total = H2O_total + emissions.total.H2O[-1]     
        
    results.emissions = Data()
    results.emissions.total = Data()
    results.emissions.total.CO2 = CO2_total
    results.emissions.total.SO2 = SO2_total
    results.emissions.total.NOx = NOx_total
    results.emissions.total.H2O = H2O_total
    
    return results

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup(LD_factor,wt_factors,prop_factors):

    # vehicle data
    vehicle  = vehicle_setup()
    
    # Change the propulsion factor
    vehicle.propulsors.turbofan.thrust.SFC_adjustment = prop_factors
    
    # Setup multiple configs
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs,LD_factor,wt_factors)

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

def analyses_setup(configs,LD_factor,wt_factors):

    analyses = SUAVE.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config,LD_factor,wt_factors)
        analyses[tag] = analysis

    # adjust analyses for configs

    # takeoff_analysis
    analyses.takeoff.aerodynamics.settings.drag_coefficient_increment = 0.0000

    # landing analysis
    aerodynamics = analyses.landing.aerodynamics
    # do something here eventually

    return analyses

def base_analysis(vehicle,LD_factor,wt_factors):

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
    weights.settings.weight_reduction_factors.main_wing = wt_factors.main_wing
    weights.settings.weight_reduction_factors.fuselage  = wt_factors.fuselage
    weights.settings.weight_reduction_factors.empennage = wt_factors.empennage
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle

    aerodynamics.settings.drag_coefficient_increment = 0.0000
    aerodynamics.settings.lift_to_drag_adjustment    = LD_factor
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Stability Analysis
    stability = SUAVE.Analyses.Stability.Fidelity_Zero()
    stability.geometry = vehicle
    analyses.append(stability)

    # ------------------------------------------------------------------
    #  Energy
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.propulsors #what is called throughout the mission (at every time step))
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

    axis_font = {'fontname':'Arial', 'size':'14'}    

    # ------------------------------------------------------------------
    #   Aerodynamics
    # ------------------------------------------------------------------


    fig = plt.figure("Aerodynamic Forces",figsize=(8,6))
    for segment in results.segments.values():

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Lift   = -segment.conditions.frames.wind.lift_force_vector[:,2]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]*0.224808943
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]*0.224808943
        eta    = segment.conditions.propulsion.throttle[:,0]
        mdot   = segment.conditions.weights.vehicle_mass_rate[:,0]
        thrust =  segment.conditions.frames.body.thrust_force_vector[:,0]
        sfc    = 3600. * mdot / 0.1019715 / thrust	


        axes = fig.add_subplot(2,1,1)
        axes.plot( time , Thrust , line_style )
        #axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Thrust (lbf)',axis_font)
        axes.grid(True)

        axes = fig.add_subplot(2,1,2)
        axes.plot( time , eta , line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('eta (lb/lbf-hr)',axis_font)
        axes.grid(True)	

        #plt.savefig("B737_engine.pdf")
        #plt.savefig("B737_engine.png")


    # ------------------------------------------------------------------
    #   Aerodynamics 2
    # ------------------------------------------------------------------
    fig = plt.figure("Aerodynamic Coefficients",figsize=(8,10))
    for segment in results.segments.values():

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        CLift  = segment.conditions.aerodynamics.lift_coefficient[:,0]
        CDrag  = segment.conditions.aerodynamics.drag_coefficient[:,0]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]
        aoa = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        l_d = CLift/CDrag


        axes = fig.add_subplot(3,1,1)
        axes.plot( time , CLift , line_style )
        #axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Lift Coefficient',axis_font)
        axes.grid(True)

        axes = fig.add_subplot(3,1,2)
        axes.plot( time , l_d , line_style )
        #axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('L/D',axis_font)
        axes.grid(True)

        axes = fig.add_subplot(3,1,3)
        axes.plot( time , aoa , 'ro-' )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('AOA (deg)',axis_font)
        axes.grid(True)

        #plt.savefig("B737_aero.pdf")
        #plt.savefig("B737_aero.png")

    # ------------------------------------------------------------------
    #   Aerodynamics 2
    # ------------------------------------------------------------------
    fig = plt.figure("Drag Components",figsize=(8,10))
    axes = plt.gca()
    for i, segment in enumerate(results.segments.values()):

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        drag_breakdown = segment.conditions.aerodynamics.drag_breakdown
        cdp = drag_breakdown.parasite.total[:,0]
        cdi = drag_breakdown.induced.total[:,0]
        cdc = drag_breakdown.compressible.total[:,0]
        cdm = drag_breakdown.miscellaneous.total[:,0]
        cd  = drag_breakdown.total[:,0]

        if line_style == 'bo-':
            axes.plot( time , cdp , 'ko-', label='CD parasite' )
            axes.plot( time , cdi , 'bo-', label='CD induced' )
            axes.plot( time , cdc , 'go-', label='CD compressibility' )
            axes.plot( time , cdm , 'yo-', label='CD miscellaneous' )
            axes.plot( time , cd  , 'ro-', label='CD total'   )
            if i == 0:
                axes.legend(loc='upper center')            
        else:
            axes.plot( time , cdp , line_style )
            axes.plot( time , cdi , line_style )
            axes.plot( time , cdc , line_style )
            axes.plot( time , cdm , line_style )
            axes.plot( time , cd  , line_style )            

    axes.set_xlabel('Time (min)')
    axes.set_ylabel('CD')
    axes.grid(True)
    #plt.savefig("B737_drag.pdf")
    #plt.savefig("B737_drag.png")

    # ------------------------------------------------------------------
    #   Altitude,sfc,vehiclde weight
    # ------------------------------------------------------------------

    fig = plt.figure("Altitude_sfc_weight",figsize=(8,10))
    for segment in results.segments.values():

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        CLift  = segment.conditions.aerodynamics.lift_coefficient[:,0]
        CDrag  = segment.conditions.aerodynamics.drag_coefficient[:,0]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]
        aoa    = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        l_d    = CLift/CDrag
        mass   = segment.conditions.weights.total_mass[:,0]*2.20462
        altitude = segment.conditions.freestream.altitude[:,0] / Units.km *3.28084 *1000
        mdot   = segment.conditions.weights.vehicle_mass_rate[:,0]
        thrust =  segment.conditions.frames.body.thrust_force_vector[:,0]
        sfc    = 3600. * mdot / 0.1019715 / thrust	


        axes = fig.add_subplot(3,1,1)
        axes.plot( time , altitude , line_style )
        #axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Altitude (ft)',axis_font)
        axes.grid(True)

        axes = fig.add_subplot(3,1,3)
        axes.plot( time , sfc , line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('sfc (lb/lbf-hr)',axis_font)
        axes.grid(True)

        axes = fig.add_subplot(3,1,2)
        axes.plot( time , mass , 'ro-' )
        #axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Weight (lb)',axis_font)
        axes.grid(True)


        #plt.savefig("B737_mission.pdf")
        #plt.savefig("B737_mission.png")

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
    
    weights = analyses.configs.base.weights
    improvements = copy.deepcopy(weights.settings.weight_reduction_factors) #(copy to resey)
    
    weights.settings.weight_reduction_factors.main_wing = 0.
    weights.settings.weight_reduction_factors.fuselage  = 0.
    weights.settings.weight_reduction_factors.empennage = 0.
    
    initial_breakdown = weights.evaluate()    
    
    # Reset weight analysis
    weights.settings.weight_reduction_factors = improvements
    
    # weight analysis
    #need to put here, otherwise it won't be updated
    improved_breakdown = weights.evaluate()   
    
    #compute centers of gravity
    #need to put here, otherwise, results won't be stored
    compute_component_centers_of_gravity(base,compute_propulsor_origin=True)
    compute_aircraft_center_of_gravity(base)
    
    weight_diff = improved_breakdown.empty - initial_breakdown.empty
    
    base.mass_properties.takeoff = base.mass_properties.takeoff + weight_diff    
    
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
    #   First Climb Segment: constant Mach, constant segment angle 
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
    #   Second Climb Segment: constant Speed, constant segment angle 
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
    #   Third Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_3"

    segment.analyses.extend( analyses.cruise )

    segment.altitude_end = 10.668 * Units.km
    segment.air_speed    = 226.0  * Units['m/s']
    segment.climb_rate   = 3.0    * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    segment.analyses.extend( analyses.cruise )

    segment.air_speed  = 230.412 * Units['m/s']
    segment.distance   = (3933.65 + 770 - 92.6) * Units.km
    
    segment.state.numerics.number_control_points = 10

    # add to mission
    mission.append_segment(segment)


# ------------------------------------------------------------------
#   First Descent Segment: consant speed, constant segment rate
# ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_1"

    segment.analyses.extend( analyses.cruise )

    segment.altitude_end = 8.0   * Units.km
    segment.air_speed    = 220.0 * Units['m/s']
    segment.descent_rate = 4.5   * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Second Descent Segment: consant speed, constant segment rate
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
    #   Third Descent Segment: consant speed, constant segment rate
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
    #   Fourth Descent Segment: consant speed, constant segment rate
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
    #   Fifth Descent Segment: consant speed, constant segment rate
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
    fuel_mission = SUAVE.Analyses.Mission.Mission() #Fuel_Constrained()
    fuel_mission.tag = 'fuel'
    fuel_mission.range   = 1277. * Units.nautical_mile
    fuel_mission.payload   = 19000.
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
    payload.tag = 'payload'
    payload.range   = 2316. * Units.nautical_mile
    payload.payload   = 19000.
    missions.append(payload)


    # done!
    return missions  

def check_results(new_results,old_results):

    # check segment values
    check_list = [
        'segments.cruise.conditions.aerodynamics.angle_of_attack',
        'segments.cruise.conditions.aerodynamics.drag_coefficient',
        'segments.cruise.conditions.aerodynamics.lift_coefficient',
        'segments.cruise.conditions.stability.static.cm_alpha',
        'segments.cruise.conditions.stability.static.cn_beta',
        'segments.cruise.conditions.propulsion.throttle',
        'segments.cruise.conditions.weights.vehicle_mass_rate',
    ]

    # do the check
    for k in check_list:
        print k

        old_val = np.max( old_results.deep_get(k) )
        new_val = np.max( new_results.deep_get(k) )
        err = (new_val-old_val)/old_val
        print 'Error at Max:' , err
        assert np.abs(err) < 1e-6 , 'Max Check Failed : %s' % k

        old_val = np.min( old_results.deep_get(k) )
        new_val = np.min( new_results.deep_get(k) )        
        err = (new_val-old_val)/old_val
        print 'Error at Min:' , err
        assert np.abs(err) < 1e-6 , 'Min Check Failed : %s' % k        

        print ''

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


def load_results():
    return SUAVE.Input_Output.SUAVE.load('results_mission_B737.res')

def save_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'results_mission_B737.res')
    return

if __name__ == '__main__': 
    main()    
    #plt.show()

