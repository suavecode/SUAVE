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
from collections import OrderedDict

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
    # This determines which performance plots are shown
    # Plots showing overall comparisons will still be shown for both 2027 and 2037
    plotting_year = 2037
    # NLF or HLF (laminar flow type)
    LF_type = 'NLF'
    # metal or composite
    fuselage_material = 'composite'
    
    # -----------------------------------------------------------
    #
    # -----------------------------------------------------------

    years = [2010,2017,2027,2037]
    
    results_dict = dict()
    emissions    = dict()
    
    for year in years:

        print 'Computing results for ' + str(year)
        
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
        
        results_dict[year] = results
        emissions[year]    = post_process_emissions(configs, results).emissions.total 

    # plt the old results
    plot_mission(results_dict[2010],'k-')
    if plotting_year == 2017:
        plot_mission(results_dict[plotting_year],'r-')    
    elif plotting_year == 2027:
        plot_mission(results_dict[plotting_year],'g-')
    elif plotting_year == 2037:
        plot_mission(results_dict[plotting_year],'b-')
    else:
        plot_mission(results_dict[plotting_year],'y-')
    plot_general_results(results_dict,emissions,years)
    plt.show(block=True)

    return

def find_LD_factor(year,LF_type):
    
    if year == 2017 or year==2010:
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
    
    if year == 2017 or year==2010:
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
    # values are single aisle (SA)
    
    if  year == 2010:
        prop_factors = 0.
    elif  year == 2017:
        prop_factors = 0.16 * 1.0
    elif year == 2027:
        prop_factors = 0.16 + 0.04 * 0.60
    elif year == 2037:
        prop_factors = 0.16 + 0.04 * 0.60 + ((1+0.6/100.)**10.-1.) * 0.60 
    else:
        raise ValueError('Year must be 2010, 2017, 2027, or 2037')    
        
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

def plot_mission(results,line_style='b-'):

    axis_font = {'fontname':'Arial', 'size':'14'}    

    # ------------------------------------------------------------------
    #   Aerodynamics
    # ------------------------------------------------------------------


    fig = plt.figure("Aerodynamic Forces",figsize=(8,6))
    for segment in results.segments.values():

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0] / Units.lb
        eta    = segment.conditions.propulsion.throttle[:,0]

        axes = fig.add_subplot(2,1,1)
        axes.plot( time , Thrust , line_style )
        axes.set_ylabel('Thrust (lbf)',axis_font)
        axes.grid(True)

        axes = fig.add_subplot(2,1,2)
        axes.plot( time , eta , line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Throttle Setting',axis_font)
        plt.ylim((0,1))
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
        aoa = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        l_d = CLift/CDrag


        axes = fig.add_subplot(3,1,1)
        axes.plot( time , CLift , line_style )
        axes.set_ylabel('Lift Coefficient',axis_font)
        axes.grid(True)

        axes = fig.add_subplot(3,1,2)
        axes.plot( time , l_d , line_style )
        axes.set_ylabel('L/D',axis_font)
        axes.grid(True)

        axes = fig.add_subplot(3,1,3)
        axes.plot( time , aoa , line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('AOA (deg)',axis_font)
        axes.grid(True)

        #plt.savefig("B737_aero.pdf")
        #plt.savefig("B737_aero.png")

    # ------------------------------------------------------------------
    #   Altitude, sfc, weight
    # ------------------------------------------------------------------

    fig = plt.figure("Altitude_sfc_weight",figsize=(8,10))
    for segment in results.segments.values():

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        mass   = segment.conditions.weights.total_mass[:,0] / Units.kg
        altitude = segment.conditions.freestream.altitude[:,0] / Units.feet
        mdot   = segment.conditions.weights.vehicle_mass_rate[:,0]
        thrust =  segment.conditions.frames.body.thrust_force_vector[:,0]
        sfc    = 9.81 * mdot / thrust * Units.hour 
        
        axes = fig.add_subplot(3,1,1)
        axes.plot( time , altitude , line_style )
        axes.set_ylabel('Altitude (ft)',axis_font)
        axes.grid(True)

        axes = fig.add_subplot(3,1,3)
        axes.plot( time , sfc , line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('sfc (lb/lbf-hr)',axis_font)
        axes.grid(True)

        axes = fig.add_subplot(3,1,2)
        axes.plot( time , mass , line_style )
        axes.set_ylabel('Weight (lb)',axis_font)
        axes.grid(True)


        #plt.savefig("B737_mission.pdf")
        #plt.savefig("B737_mission.png")

    return

# ----------------------------------------------------------------------
#   Plot General Results
# ----------------------------------------------------------------------

def plot_general_results(results_dict,emissions,years):
    
    # Fuel Burn Calculation
    
    fuel_burn      = dict()
    for year in years:
        takeoff_mass    = results_dict[year].segments[0].conditions.weights.total_mass[0,0]
        final_mass      = results_dict[year].segments[-1].conditions.weights.total_mass[-1,0]
        fuel_burn[year] = (takeoff_mass-final_mass) / Units.lb
        
    fuel_burn_vals = np.array([fuel_burn[2010],fuel_burn[2017],fuel_burn[2027],fuel_burn[2037]])

    # Plot Fuel Burn
    
    ind = np.arange(1,5)
    fig, ax = plt.subplots()
    y17,y27,y37,y47 = plt.bar(ind,fuel_burn_vals,align='center')
    y17.set_facecolor('k')
    y27.set_facecolor('r')
    y37.set_facecolor('y')    
    y47.set_facecolor('g')   
    
    perc27 = (1.-fuel_burn[2017]/fuel_burn[2010])*100.
    perc37 = (1.-fuel_burn[2027]/fuel_burn[2010])*100.   
    perc47 = (1.-fuel_burn[2037]/fuel_burn[2010])*100.  
    
    legend_27 = "{:.1f}".format(perc27) + '% Reduction'
    legend_37 = "{:.1f}".format(perc37) + '% Reduction'
    legend_47 = "{:.1f}".format(perc47) + '% Reduction'
    
    plt.legend([y17,y27,y37,y47],['Baseline',legend_27,legend_37,legend_47])
    
    ax.set_xticks(ind)
    ax.set_xticklabels(['2010','2017','2027','2037'])
    ax.set_ylim([0,fuel_burn[2010]*1.4])
    ax.set_ylabel('Fuel Burn (lb)')
    ax.set_title('Fuel Burn by Year')    

    # Extract Emissions Values
    H2O_vals = np.array([emissions[2010].H2O[0],emissions[2017].H2O[0],emissions[2027].H2O[0],emissions[2037].H2O[0]]) / Units.lb
    CO2_vals = np.array([emissions[2010].CO2[0],emissions[2017].CO2[0],emissions[2027].CO2[0],emissions[2037].CO2[0]]) / Units.lb
    NOx_vals = np.array([emissions[2010].NOx[0],emissions[2017].NOx[0],emissions[2027].NOx[0],emissions[2037].NOx[0]]) / Units.lb
    SO2_vals = np.array([emissions[2010].SO2[0],emissions[2017].SO2[0],emissions[2027].SO2[0],emissions[2037].SO2[0]]) / Units.lb
    
    # 4 Plot
    emissions_types = ['H2O','CO2','NOx','SO2']
    emissions_vals  = [H2O_vals,CO2_vals,NOx_vals,SO2_vals]
    
    fig, ax = plt.subplots(2,2,figsize=(13, 7))
    for ii in range(2):
        for jj in range(2):
        
            em_ind = ii*2+jj
            y17,y27,y37,y47 = ax[ii,jj].bar(ind,emissions_vals[em_ind],align='center')
            
            y17.set_facecolor('k')
            y27.set_facecolor('r')
            y37.set_facecolor('y')    
            y47.set_facecolor('g') 
            
            perc27 = (1.-emissions_vals[em_ind][1]/emissions_vals[em_ind][0])*100.
            perc37 = (1.-emissions_vals[em_ind][2]/emissions_vals[em_ind][0])*100.
            perc47 = (1.-emissions_vals[em_ind][3]/emissions_vals[em_ind][0])*100.
            
            legend_27 = "{:.1f}".format(perc27) + '% Reduction'
            legend_37 = "{:.1f}".format(perc37) + '% Reduction'
            legend_47 = "{:.1f}".format(perc47) + '% Reduction'
            
            ax[ii,jj].legend([y17,y27,y37,y47],['Baseline',legend_27,legend_37,legend_47])
            
            ax[ii,jj].set_xticks(ind)
            ax[ii,jj].set_xticklabels(['2010','2017','2027','2037'])
            ax[ii,jj].set_ylim([0,emissions_vals[em_ind][0]*1.4])
            ax[ii,jj].set_ylabel(emissions_types[em_ind] + ' (lb)')
            ax[ii,jj].set_title(emissions_types[em_ind] + ' by Year')
      

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
    base_segment.state.numerics.number_control_points = 8

    # ------------------------------------------------------------------
    #   First Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"

    segment.analyses.extend( analyses.takeoff )

    segment.altitude_start = 0.0   
    segment.altitude_end   = 10000. * Units.feet
    segment.air_speed      = 125.0  * Units['m/s']
    segment.climb_rate     = 1000. * Units['feet/minute']

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Climb Segment: constant Speed, constant segment angle 
    # ------------------------------------------------------------------    

    segment = Segments.Climb.Constant_Mach_Constant_Rate(base_segment)
    segment.tag = "climb_2"

    segment.analyses.extend( analyses.cruise )

    segment.altitude_end   = 25000. * Units.feet
    segment.mach           = 0.74
    segment.climb_rate     = 800. * Units['feet/minute']

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Third Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    

    segment = Segments.Climb.Constant_Mach_Constant_Rate(base_segment)
    segment.tag = "climb_3"

    segment.analyses.extend( analyses.cruise )

    segment.altitude_end = 35000. * Units.feet
    segment.mach         = 0.68
    segment.climb_rate   = 275. * Units['feet/minute']

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    

    segment = Segments.Cruise.Constant_Mach_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    segment.analyses.extend( analyses.cruise )

    segment.mach     = 0.78
    segment.distance = 2275. * Units.nautical_mile   
    
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

    
    #------------------------------------------------------------------
    #         Reserve mission
    #------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Throttle
    # ------------------------------------------------------------------
 
    segment = Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "reserve_climb"
 
    # connect vehicle configuration
    segment.analyses.extend( analyses.base )

    segment.altitude_start = 0.0    * Units.km
    segment.altitude_end   = 15000. * Units.ft
    segment.air_speed      = 138.0  * Units['m/s']
    segment.climb_rate     = 2000.  * Units['ft/min']
 
    # add to misison
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------
    
    segment = Segments.Cruise.Constant_Mach_Constant_Altitude(base_segment)
    segment.tag = "reserve_cruise"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.mach      = 0.5
    segment.distance  = 140.0 * Units.nautical_mile    
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Loiter Segment: constant mach, constant time
    # ------------------------------------------------------------------
    
    segment = Segments.Cruise.Constant_Mach_Constant_Altitude_Loiter(base_segment)
    segment.tag = "reserve_loiter"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.mach = 0.5
    segment.time = 30.0 * Units.minutes
    
    mission.append_segment(segment)    
    
    
    # ------------------------------------------------------------------
    #   Fifth Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------
    
    segment = Segments.Descent.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "reserve_descent_1"
    
    segment.analyses.extend( analyses.landing )
    analyses.landing.aerodynamics.settings.spoiler_drag_increment = 0.00
    
    
    segment.altitude_end = 0.0   * Units.km
    segment.descent_rate = 3.0   * Units['m/s']
    segment.mach_end     = 0.24
    segment.mach_start   = 0.3
    
    # append to mission
    mission.append_segment(segment)
    
    #------------------------------------------------------------------
    #          Reserve mission completed
    #------------------------------------------------------------------    

    return mission

def missions_setup(base_mission):

    # the mission container
    missions = SUAVE.Analyses.Mission.Mission.Container()
    missions.base = base_mission

    return missions  

def load_results():
    return SUAVE.Input_Output.SUAVE.load('results_mission_B737.res')

def save_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'results_mission_B737.res')
    return

if __name__ == '__main__': 
    main()    