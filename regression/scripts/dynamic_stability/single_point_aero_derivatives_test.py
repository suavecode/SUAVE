# Aerodynamic_Finite_Differencing.py
# 
# Created:   Jul 2021, R. Erhard
# Modified: 

""" 
Setup file for evaluating gradients for the NASA X-57 Maxwell in a trimmed cruise
condition. Gradients of the vehicle forces and moments are evaluated with respect
to the wing angle of attack and each propeller throttle setting.
"""
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units
from SUAVE.Plots.Mission_Plots import *  
from SUAVE.Methods.Flight_Dynamics.Dynamic_Stability.single_point_aero_derivatives import single_point_aero_derivatives

import pylab as plt
import sys

sys.path.append('../Vehicles') 
from X57_Maxwell_Mod2 import vehicle_setup, configs_setup


# -------------------------------------------------------------------------------
#   Main
# -------------------------------------------------------------------------------
def main():
    #----------------------------------------------------------------------------
    # Analyze without prop-wing interaction:
    #----------------------------------------------------------------------------
    aero_derivatives = run_test(fixed_helical_wake=False)
    
    # verify results    
    dCL_dAlpha_t     = np.array([6.484447102988838])
    dCT_dAlpha0_t    = np.array([0.012724685069449848])
    
    assert(aero_derivatives.dCL_dAlpha-dCL_dAlpha_t < 1e-5)
    assert(aero_derivatives.dCT_dAlpha[0]-dCT_dAlpha0_t < 1e-5)
    
    # print table of aerodynamic derivatives
    display_derivative_table(aero_derivatives)
    
    return 

def run_test(fixed_helical_wake):
    #----------------------------------------------------------------------------
    # setup the vehicle and mission segment
    #----------------------------------------------------------------------------    
    Vcruise            = 135 * Units.mph
    Alt                = 8000 * Units.feet
    configs, analyses  = full_setup(fixed_helical_wake,Vcruise=Vcruise,Alt=Alt) 
    configs.finalize()
    analyses.finalize()
    
    mission       = analyses.missions.base
    vehicle       = configs.base
    
    results = mission.evaluate()
    cruise_state   = results.segments.cruise.state
    
    #----------------------------------------------------------------------------
    # compute aerodynamic derivatives for this single-point mission segment
    #----------------------------------------------------------------------------  
    aero_derivatives = single_point_aero_derivatives(analyses.configs,vehicle, cruise_state)
    

    return aero_derivatives

def display_derivative_table(aero_derivatives):
    
    # plot table of aero derivatives 
    column_labels=["$d\\alpha$", "dThrottle", "$dV_\\infty$"]
    row_labels=["$dC_L$", "$dC_{D}$"]
    
    plt.rcParams["figure.figsize"] = [7.00, 4.00]
    plt.rcParams["figure.autolayout"] = True    
    fig, ax =plt.subplots(1,1)
    
    data=np.array([[aero_derivatives.dCL_dAlpha,aero_derivatives.dCL_dThrottle,aero_derivatives.dCL_dV],
          [aero_derivatives.dCD_dAlpha,aero_derivatives.dCD_dThrottle,aero_derivatives.dCD_dV],])
    
    for i in range(len(aero_derivatives.dCT_dAlpha)):
        data = np.append(data,[[aero_derivatives.dCT_dAlpha[i],aero_derivatives.dCT_dThrottle[i],aero_derivatives.dCT_dV[i]]],axis=0)
        row_labels = np.append(row_labels,f"$dC_T({i})$")
        data = np.append(data,[[aero_derivatives.dMyaw_dAlpha[i],aero_derivatives.dMyaw_dThrottle[i],aero_derivatives.dMyaw_dV[i]]],axis=0)
        row_labels = np.append(row_labels,f"$dM_y({i})$" )
    
    
    round_data = np.round(data,3)
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=round_data,rowLabels=row_labels,colLabels=column_labels,loc="center")  
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    plt.show()
    
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup(fixed_helical_wake,Vcruise,Alt):

    # vehicle data
    vehicle  = vehicle_setup() 
    configs  = configs_setup(vehicle)
    
    # enable 2d rotor analysis
    vehicle.networks.battery_propeller.propellers.propeller.use_2d_analysis  = True

    # vehicle analyses
    configs_analyses = analyses_setup(configs, fixed_helical_wake)

    # mission analyses
    mission  = cruise_mission(configs_analyses,vehicle,Vcruise=Vcruise,Alt=Alt) 
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
#   Define the Initial Mission
# ----------------------------------------------------------------------

def cruise_mission(analyses,vehicle,Vcruise=135.*Units.mph, Alt=8000.*Units.feet):
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
    #   Cruise Segment: constant Speed, constant altitude
    # ------------------------------------------------------------------ 
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise" 
    segment.analyses.extend(analyses.base)  
    segment.battery_energy            = vehicle.networks.battery_propeller.battery.max_energy* 0.8
    segment.altitude                  = Alt 
    segment.air_speed                 = Vcruise
    segment.distance                  = 20.   * Units.nautical_mile  
    segment.state.unknowns.throttle   = 0.85  * ones_row(1)
    segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)
    
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

    return missions  


if __name__ == '__main__': 
    main()    
    plt.show()
