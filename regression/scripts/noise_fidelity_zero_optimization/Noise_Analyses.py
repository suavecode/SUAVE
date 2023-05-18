# Noise_Analyses.py
# 
# Created:  Nov 2015, Carlos / Tarik
# Modified: Jul, 2017 M. Clarke

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import MARC

# ----------------------------------------------------------------------        
#   Setup Analyses
# ----------------------------------------------------------------------  

def setup(configs): 
    
    analyses = MARC.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in list(configs.items()):
        analysis = base(config)
        analyses[tag] = analysis

    return analyses

# ----------------------------------------------------------------------        
#   Define Base Analysis
# ----------------------------------------------------------------------  

def base(vehicle):

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
    aerodynamics.settings.number_spanwise_vortices   = 5
    aerodynamics.settings.number_chordwise_vortices  = 2
    aerodynamics.geometry                    = vehicle

    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Stability Analysis
    stability = MARC.Analyses.Stability.Fidelity_Zero()
    stability.geometry = vehicle
    analyses.append(stability)


    # ------------------------------------------------------------------
    #  Noise Analysis
    noise = MARC.Analyses.Noise.Fidelity_Zero()
    noise.geometry = vehicle
    analyses.append(noise)

    # ------------------------------------------------------------------
    #  Energy
    energy= MARC.Analyses.Energy.Energy()
    energy.network = vehicle.networks 
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