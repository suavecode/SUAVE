# ICE_Test.py
# 
# Created: Feb 2020, M. Clarke 
 
""" setup file for a mission with a Cessna 172 with an internal combustion engine network
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units 
import numpy as np 
 

from SUAVE.Core import (
Data, Container,
)

import sys

sys.path.append('../Vehicles')
# the analysis functions 
 
from Cessna_172  import vehicle_setup  
# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():   
     
    # Define internal combustion engine from Cessna Regression Aircraft 
    vehicle    = vehicle_setup()

    # Setup analyses and mission
    analyses = base_analysis(vehicle)
    analyses.finalize()
    mission  = mission_setup(analyses,vehicle)
    
    # evaluate
    results = mission.evaluate()
    
    P_truth     = 113312.8154744112
    mdot_truth  = 0.008754147707289401
    
    P    = results.segments.cruise.state.conditions.propulsion.power[-1,0]
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


# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------

def mission_setup(analyses,vehicle):

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
    #   Cruise Segment: Constant Speed Constant Altitude
    # ------------------------------------------------------------------    

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    segment.analyses.extend( analyses )

    segment.altitude  = 12000. * Units.feet
    segment.air_speed = 119.   * Units.knots
    segment.distance  = 10 * Units.nautical_mile
    
    ones_row                                        = segment.state.ones_row   
    segment.state.numerics.number_control_points    = 4
    segment.state.unknowns.throttle                 = 1.0 * ones_row(1)
    segment = vehicle.networks.internal_combustion.add_unknowns_and_residuals_to_segment(segment,rpm=2650)
    
    
    segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability = SUAVE.Methods.skip    

    # add to mission
    mission.append_segment(segment)


    return mission


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
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    