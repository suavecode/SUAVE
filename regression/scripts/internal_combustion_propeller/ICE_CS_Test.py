# ICE_Test.py
# 
# Created: Feb 2020, M. Clarke 
 
""" setup file for a mission with a Cessna 172 with an internal combustion engine network
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import MARC
from MARC.Core import Units 
import numpy as np 
 

from MARC.Core import (
Data, Container,
)

import sys

sys.path.append('../Vehicles')
# the analysis functions 
 
from Cessna_172      import vehicle_setup  
from MARC.Methods.Propulsion import propeller_design

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():   
     
    # Define internal combustion engine from Cessna Regression Aircraft 
    vehicle    = vehicle_setup()
    
    # Setup the modified constant speed version of the network
    vehicle = ICE_CS(vehicle)
    
    # Setup analyses and mission
    analyses = base_analysis(vehicle)
    analyses.finalize()
    mission  = mission_setup(analyses)
    
    # evaluate
    results = mission.evaluate()
    
    P_truth     = 53595.133857796805
    mdot_truth  = 0.004708991159029469
    
    P    = results.segments.cruise.state.conditions.propulsion.propulsor_group_0.power[-1,0]
    mdot = results.segments.cruise.state.conditions.weights.vehicle_mass_rate[-1,0]     

    # Check the errors
    error = Data()
    error.P      = np.max(np.abs((P     - P_truth)/P_truth))
    error.mdot   = np.max(np.abs((mdot - mdot_truth)/mdot_truth))


    print('Errors:')
    print(error)

    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)

    return


def ICE_CS(vehicle):
    
    # Replace the C172 engine and propeller with a constant speed propeller
    # Let's assume its an STC or 172RG 
    
    # build network
    net                                         = MARC.Components.Energy.Networks.Internal_Combustion_Propeller_Constant_Speed()
    net.tag                                     = 'internal_combustion'
    net.number_of_engines                       = 1.
    net.rated_speed                             = 2700. * Units.rpm
    net.rated_power                             = 180.  * Units.hp
    
    # Component 1 the engine                    
    engine                                  = MARC.Components.Energy.Converters.Internal_Combustion_Engine()
    engine.sea_level_power                  = 180. * Units.horsepower
    engine.flat_rate_altitude               = 0.0
    engine.rated_speed                      = 2700. * Units.rpm
    engine.power_specific_fuel_consumption  = 0.52
    
    net.engines.append(engine)
    
    # 
    prop                                   = MARC.Components.Energy.Converters.Propeller()
    prop.number_of_blades                  = 2.0
    prop.tip_radius                        = 76./2. * Units.inches
    prop.hub_radius                        = 8.     * Units.inches
    prop.cruise.design_freestream_velocity = 119.   * Units.knots
    prop.cruise.design_angular_velocity    = 2650.  * Units.rpm
    prop.cruise.design_Cl                  = 0.8
    prop.cruise.design_altitude            = 12000. * Units.feet
    prop.cruise.design_power               = .64 * 180. * Units.horsepower 
    airfoil                                = MARC.Components.Airfoils.Airfoil()   
    airfoil.coordinate_file                = '../Vehicles/Airfoils/NACA_4412.txt'
    airfoil.polar_files                    = ['../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_50000.txt' ,
                                           '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_100000.txt' ,
                                           '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_200000.txt' ,
                                           '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_500000.txt' ,
                                           '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_1000000.txt' ] 
    prop.append_airfoil(airfoil)  
    prop.airfoil_polar_stations            = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    prop                                   = propeller_design(prop)    
    
    net.propellers.append(prop)
    
    # Replace the network
    vehicle.networks.internal_combustion = net
    
    
    return vehicle


# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------

def mission_setup(analyses):

    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = MARC.Analyses.Mission.Sequential_Segments()
    mission.tag = 'the_mission'

    #airport
    airport = MARC.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = MARC.Attributes.Atmospheres.Earth.US_Standard_1976()
    mission.airport    = airport    

    # unpack Segments module
    Segments = MARC.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment()
    ones_row     = base_segment.state.ones_row
    base_segment.state.numerics.number_control_points    = 3


    # ------------------------------------------------------------------    
    #   Cruise Segment: Constant Speed Constant Altitude
    # ------------------------------------------------------------------    

    segment        = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag    = "cruise"
    segment.analyses.extend( analyses )

    segment.altitude                                = 12000. * Units.feet
    segment.air_speed                               = 119.   * Units.knots
    segment.distance                                = 10 * Units.nautical_mile
    segment.state.conditions.propulsion.rpm         = 2650.  * Units.rpm *  ones_row(1) 
    segment.state.unknowns.throttle                 = 0.5 *  ones_row(1)

    # add to mission
    mission.append_segment(segment)


    return mission



def base_analysis(vehicle):

    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------     
    analyses = MARC.Analyses.Vehicle()

    # ------------------------------------------------------------------
    #  Basic Geometry Relations
    sizing = MARC.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)

    # ------------------------------------------------------------------
    #  Weights
    weights = MARC.Analyses.Weights.Weights_Transport()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = MARC.Analyses.Aerodynamics.Fidelity_Zero() 
    aerodynamics.geometry                            = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Energy
    energy= MARC.Analyses.Energy.Energy()
    energy.network = vehicle.networks #what is called throughout the mission (at every time step))
    analyses.append(energy)

    # ------------------------------------------------------------------
    #  Planet Analysis
    planet = MARC.Analyses.Planets.Planet()
    analyses.append(planet)

    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = MARC.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)   

    # done!
    return analyses
    
    
    
    

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    