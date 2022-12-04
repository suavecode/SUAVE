## @ingroup Methods-Weights-Buildups-Common

# fuselage.py
#
# Created: Jun, 2017, J. Smart
# Modified: Apr 2018, J. Smart
#           Mar 2020, M. Clarke
#           Mar 2020, J. Smart
#           Dec 2022, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Data, Container, Units
from SUAVE.Attributes.Solids import (
    Bidirectional_Carbon_Fiber as BiCRFP,
    Unidirectional_Carbon_Fiber as UniCRFP,
    Carbon_Fiber_Honeycomb as CFHoneycomb,
    Paint,
    Acrylic,
    Steel)

from SUAVE.Methods.Weights.Buildups.Common import elliptical_shell
from SUAVE.Methods.Weights.Buildups.Common.stack_mass import stack_mass

import numpy as np


#-------------------------------------------------------------------------------
# Fuselage
#-------------------------------------------------------------------------------

## @ingroup Methods-Weights-Buildups-Common
def fuselage(config,
             maximum_g_load = 3.8,
             landing_impact_factor = 3.5,
             safety_factor = 1.5):
    """ Calculates the structural mass of a fuselage for an eVTOL vehicle,
        assuming a structural keel taking bending an torsional loads.
        
        Assumptions:
        Assumes an elliptical fuselage. Intended for use with the following
        SUAVE vehicle types, but may be used elsewhere:

            Electric Multicopter
            Electric Vectored_Thrust
            Electric Stopped Rotor

        Originally written as part of an AA 290 project intended for trade study
        of the above vehicle types.

        If vehicle model does not have material properties assigned, appropriate
        assumptions are made based on SUAVE's Solids Attributes library.
        
        Sources:
        Project Vahana Conceptual Trade Study

        Inputs:

            config                      SUAVE Vehicle Configuration
            max_g_load                  Max Accelerative Load During Flight [Unitless]
            landing_impact_factor       Maximum Load Multiplier on Landing  [Unitless]

        Outputs:

            weight:                 Estimated Fuselage Mass             [kg]
        
        Properties Used:
        Material Properties of Imported SUAVE Solids
    """

    #-------------------------------------------------------------------------------
    # Unpack Inputs
    #-------------------------------------------------------------------------------
 
    fuse    = config.fuselages.fuselage 
    l = fuse.lengths.total
    w  = fuse.width
    h = fuse.heights.maximum
    b = config.wings["main_wing"].spans.projected
    MTOW    = config.mass_properties.max_takeoff
    G_max   = maximum_g_load
    LIF     = landing_impact_factor
    SF      = safety_factor

    #---------------------------------------------------------------------------
    # Unpack Material Properties
    #---------------------------------------------------------------------------

    # Bending Carrier

    try:
        mats = fuse.materials
    except AttributeError:
        fuse.materials              = Data()
        fuse.materials.keel         = Container()
        fuse.materials.skin         = Container()
        fuse.materials.canopy       = Container()
        fuse.materials.landing_gear = Container()

    try:
        rbmMat = fuse.materials.keel.bending_carrier
    except AttributeError:
        fuse.materials.keel.bending_carrier = UniCRFP()
        rbmMat = fuse.materials.keel.bending_carrier

    rbmDen = rbmMat.density
    rbmUTS = rbmMat.ultimate_tensile_strength

    # Shear Carrier

    try:
        shearMat = fuse.materials.keel.shear_carrier
    except AttributeError:
        fuse.materials.keel.shear_carrier = BiCRFP()
        shearMat = fuse.materials.keel.shear_carrier

    shearDen = shearMat.density
    shearUSS = shearMat.ultimate_shear_strength

    # Landing Gear Bearing Material

    try:
        bearingMat = fuse.materials.landing_gear.bearing_carrier
    except AttributeError:
        fuse.materials.landing_gear.bearing_carrier = BiCRFP()
        bearingMat = fuse.materials.landing_gear.bearing_carrier
    bearingDen = bearingMat.density
    bearingUBS = bearingMat.ultimate_bearing_strength

    # Landing Gear Shear Bolt Material

    try:
        boltMat = fuse.materials.landing_gear.bolt
    except AttributeError:
        fuse.materials.landing_gear.bolt = Steel()
        boltMat = fuse.materials.landing_gear.bolt

    boltUSS = boltMat.ultimate_shear_strength

    # Skin Materials


    skinMats = fuse.materials.skin

    if len(skinMats)==0:
        fuse.materials.skin.base    = BiCRFP()
        fuse.materials.skin.core    = CFHoneycomb()
        fuse.materials.skin.cover   = Paint()
        skinMats = fuse.materials.skin

    # Canopy Materials

    canopyMats = fuse.materials.canopy

    if len(canopyMats)==0:
        fuse.materials.canopy.base = Acrylic()

        canopyMats = fuse.materials.canopy

    # --------------------------------------------------------------------------
    # Unloaded Components
    # --------------------------------------------------------------------------

    # Calculate Skin & Canopy Weight Assuming 1/8 of the Wetted Area is Canopy

    skinMass    = 0.875 * elliptical_shell(fuse)
    canopyMass  = 0.125 * elliptical_shell(fuse, skin_materials=canopyMats)

    # Calculate the mass of a structural bulkhead

    bulkheadMass = 3 * np.pi * h * w/4 * stack_mass(skinMats)

    # --------------------------------------------------------------------------
    # Loaded Components
    # --------------------------------------------------------------------------

    # Calculate keel mass needed to carry lifting moment

    L_max       = G_max * MTOW * 9.8 * SF   # Max Lifting Load
    M_lift      = L_max * l/2.              # Max Moment Due to Lift
    beamWidth   = w/3.                      # Allowable Keel Width
    beamHeight  = h/10.                     # Allowable Keel Height

    beamArea    = M_lift * beamHeight/(4*rbmUTS*(beamHeight/2)**2)
    keelMass    = beamArea * l * rbmDen

    # Calculate keel mass needed to carry wing bending moment shear

    M_bend      = L_max/2 * b/2                         # Max Bending Moment
    beamArea    = beamHeight * beamWidth                # Enclosed Beam Area
    beamThk     = 0.5 * M_bend/(shearUSS * beamArea)    # Beam Thickness
    keelMass   += 2*(beamHeight + beamWidth)*beamThk*shearDen

    # Calculate keel mass needed to carry landing impact load assuming 40 deg.

    shear_comp  = np.sin(np.deg2rad(40))                # Shear Force Component
    F_landing   = SF * MTOW * 9.8 * LIF * shear_comp    # Side Landing Force
    boltArea    = F_landing/boltUSS                     # Required Bolt Area
    boltDiam    = 2 * np.sqrt(boltArea/np.pi)           # Bolt Diameter
    lamThk      = F_landing/(boltDiam*bearingUBS)       # Laminate Thickness
    lamVol      = (np.pi*(20*lamThk)**2)*(lamThk/3)     # Laminate Pad volume
    keelMass   += 4*lamVol*bearingDen                   # Mass of 4 Pads

    # Calculate total mass as the sum of skin mass, bulkhead mass, canopy pass,
    # and keel mass. Called weight by SUAVE convention

    # --------------------------------------------------------------------------
    # Pack Results
    # --------------------------------------------------------------------------

    results             = Data()
    results.total       = (skinMass+canopyMass+bulkheadMass+keelMass) * Units.kg
    results.skin        = skinMass                                    * Units.kg
    results.canopy      = canopyMass                                  * Units.kg
    results.bulkheads   = bulkheadMass                                * Units.kg
    results.keel        = keelMass                                    * Units.kg

    return results