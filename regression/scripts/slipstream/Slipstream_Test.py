# Cessna_172.py
# 
# Created:  Mar 2019, M. Clarke

""" setup file for a cruise segment of the NASA X-57 Maxwell (Twin Engine Variant) Electric Aircraft 
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
from X57_Maxwell import vehicle_setup, configs_setup 

import copy

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():

    configs, analyses = full_setup() 
    
    configs.finalize()
    analyses.finalize()  

    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()
     
    # lift coefficient  
    lift_coefficient              = results.segments.cruise.conditions.aerodynamics.lift_coefficient[2][0]
    lift_coefficient_true         = 0.38334976763569456
    print(lift_coefficient)
    diff_CL                       = np.abs(lift_coefficient  - lift_coefficient_true) 
    print('CL difference')
    print(diff_CL)
    assert np.abs((lift_coefficient  - lift_coefficient_true)/lift_coefficient_true) < 1e-6
    
    # sectional lift coefficient check
    sectional_lift_coeff            = results.segments.cruise.conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional_lift[0]
    sectional_lift_coeff_true       = np.array([ 6.58384044e-02,  6.58040160e-02,  6.51219648e-02,  6.39198461e-02,
                                                 6.22727840e-02,  6.02383367e-02,  5.78644070e-02,  5.51925277e-02,
                                                 5.22596905e-02,  4.90995483e-02,  4.57433655e-02,  4.22209521e-02,
                                                 3.85617637e-02,  3.47962993e-02,  3.09578472e-02,  2.70844205e-02,
                                                 2.32203126e-02,  1.94161501e-02,  1.57264275e-02,  1.22061312e-02,
                                                 8.91378952e-03,  5.92974527e-03,  3.38156996e-03,  1.44245855e-03,
                                                 2.80627201e-04,  6.58527202e-02,  6.58456267e-02,  6.51869030e-02,
                                                 6.40026152e-02,  6.23673731e-02,  6.03390050e-02,  5.79661925e-02,
                                                 5.52914906e-02,  5.23529502e-02,  4.91851880e-02,  4.58202711e-02,
                                                 4.22886332e-02,  3.86201868e-02,  3.48457490e-02,  3.09988193e-02,
                                                 2.71175481e-02,  2.32463209e-02,  1.94358337e-02,  1.57406406e-02,
                                                 1.22157771e-02,  8.91979956e-03,  5.93304733e-03,  3.38305095e-03,
                                                 1.44291171e-03,  2.80679619e-04, -1.53946964e-04, -1.50276583e-04,
                                                -1.38933835e-04, -1.20431171e-04, -9.58611563e-05, -6.66806634e-05,
                                                -3.45505036e-05, -1.20118198e-06,  3.16768305e-05,  6.25171515e-05,
                                                 8.99395684e-05,  1.12796387e-04,  1.30206378e-04,  1.41577193e-04,
                                                 1.46617328e-04,  1.45339516e-04,  1.38059243e-04,  1.25395696e-04,
                                                 1.08283677e-04,  8.79966050e-05,  6.61579250e-05,  4.46902184e-05,
                                                 2.56519613e-05,  1.09753067e-05,  2.14822950e-06, -1.53049312e-04,
                                                -1.49806404e-04, -1.38618139e-04, -1.20211714e-04, -9.57151550e-05,
                                                -6.65899536e-05, -3.44985757e-05, -1.17361608e-06,  3.16917277e-05,
                                                 6.25280178e-05,  8.99520273e-05,  1.12813384e-04,  1.30228704e-04,
                                                 1.41604079e-04,  1.46647040e-04,  1.45369888e-04,  1.38088118e-04,
                                                 1.25421251e-04,  1.08304638e-04,  8.80123545e-05,  6.61685280e-05,
                                                 4.46963596e-05,  2.56547889e-05,  1.09761718e-05,  2.14832890e-06,
                                                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                                 0.00000000e+00])
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
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

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
    aerodynamics.settings.use_surrogate = False
    aerodynamics.geometry = vehicle
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
    base_segment.state.numerics.number_control_points        = 4
    base_segment.process.iterate.unknowns.network            = vehicle.propulsors.propulsor.unpack_unknowns
    base_segment.process.iterate.residuals.network           = vehicle.propulsors.propulsor.residuals
    base_segment.state.unknowns.propeller_power_coefficient  = 0.005 * ones_row(1) 
    base_segment.state.unknowns.battery_voltage_under_load   = vehicle.propulsors.propulsor.battery.max_voltage * ones_row(1)  
    base_segment.state.residuals.network                     = 0. * ones_row(2)        
    
    # ------------------------------------------------------------------
    #   Climb 1 : constant Speed, constant rate segment 
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"
    segment.analyses.extend( analyses.base )
    segment.battery_energy                   = vehicle.propulsors.propulsor.battery.max_energy * 0.89
    segment.altitude_start                   = 2500.0  * Units.feet
    segment.altitude_end                     = 8012    * Units.feet 
    segment.air_speed                        = 96.4260 * Units['mph'] 
    segment.climb_rate                       = 700.034 * Units['ft/min']  
    segment.state.unknowns.throttle          = 0.85 * ones_row(1)  

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Cruise Segment: constant Speed, constant altitude
    # ------------------------------------------------------------------ 
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise" 
    segment.analyses.extend(analyses.base) 
    segment.altitude                  = 8012   * Units.feet
    segment.air_speed                 = 140.91 * Units['mph'] 
    segment.distance                  =  20.   * Units.nautical_mile  
    segment.state.unknowns.throttle   = 0.9 *  ones_row(1)   

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