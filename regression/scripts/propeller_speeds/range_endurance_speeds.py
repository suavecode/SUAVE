

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------


import SUAVE
from SUAVE.Core import Units

import scipy as sp
import numpy as np


#SUAVE.Analyses.Process.verbose = True
import sys
sys.path.append('../Vehicles')
from Stopped_Rotor import vehicle_setup as vehicle_setup_SR 

from SUAVE.Methods.Performance.propeller_range_endurance_speeds import propeller_range_endurance_speeds

# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():

    vehicle_SR, analyses_SR = full_setup_SR()
    analyses_SR.finalize()
    
    altitude = 100.
    CL_max = 5.
    up_bnd = 250. * Units['mph']
    delta_isa = 0.
    
    results = propeller_range_endurance_speeds(analyses_SR,altitude,CL_max,up_bnd,delta_isa)
    print(results.L_D_max.air_speed)
    print(results.CL32.air_speed)
    
    saved_results_L_D  = 43.29245867
    saved_results_CL32 = 35.83097395

    error_L_D = float(abs(results.L_D_max.air_speed - saved_results_L_D)/saved_results_L_D)
    error_32  = float(abs(results.CL32.air_speed    - saved_results_CL32)/saved_results_CL32)
    
    print('Error in L/D Max Airspeed' , error_L_D)
    assert error_L_D < 1e-6    
    
    print('Error in CL**(3/2) / CD Max Airspeed' , error_32)
    assert error_32 < 1e-6        
  

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------
def full_setup_SR():
    
    # vehicle data
    vehicle  = vehicle_setup_SR() 

    # vehicle analyses
    analyses = base_analysis_SR(vehicle)
    
    return  vehicle, analyses


def base_analysis_SR(vehicle):

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
    weights = SUAVE.Analyses.Weights.Weights_eVTOL()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.4*vehicle.excrescence_area_spin / vehicle.reference_area
    analyses.append(aerodynamics)

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

    return analyses    
    

if __name__ == '__main__':
    main()

