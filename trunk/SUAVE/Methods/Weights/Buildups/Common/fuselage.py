## @ingroup Methods-Weights-Buildups-Common

# fuselage.py
#
# Created: Jun, 2017, J. Smart
# Modified: Apr, 2018, J. Smart
#           Mar 2020, M. Clarke

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Units
from SUAVE.Attributes.Solids import (
    Bidirectional_Carbon_Fiber, Carbon_Fiber_Honeycomb, Paint, Unidirectional_Carbon_Fiber, Acrylic, Steel)
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

    fLength = config.fuselages.fuselage.lengths.total
    fWidth  = config.fuselages.fuselage.width
    fHeight = config.fuselages.fuselage.heights.maximum
    maxSpan = config.wings['main_wing'].spans.projected
    MTOW    = config.mass_properties.max_takeoff
    G_max   = maximum_g_load
    LIF     = landing_impact_factor
    SF      = safety_factor
    
#-------------------------------------------------------------------------------
# Unpack Material Properties
#-------------------------------------------------------------------------------

    BiCF = Bidirectional_Carbon_Fiber()
    BiCF_MGT = BiCF.minimum_gage_thickness
    BiCF_DEN = BiCF.density
    BiCF_UTS = BiCF.ultimate_tensile_strength
    BiCF_USS = BiCF.ultimate_shear_strength
    BiCF_UBS = BiCF.ultimate_bearing_strength
    
    UniCF = Unidirectional_Carbon_Fiber()
    UniCF_MGT = UniCF.minimum_gage_thickness
    UniCF_DEN = UniCF.density
    UniCF_UTS = UniCF.ultimate_tensile_strength
    UniCF_USS = UniCF.ultimate_shear_strength
    
    HCMB = Carbon_Fiber_Honeycomb()
    HCMB_MGT = HCMB.minimum_gage_thickness
    HCMB_DEN = HCMB.density
    
    PAINT = Paint()
    PAINT_MGT = PAINT.minimum_gage_thickness
    PAINT_DEN = PAINT.density
    
    ACRYL = Acrylic()
    ACRYL_MGT = ACRYL.minimum_gage_thickness
    ACRYL_DEN = ACRYL.density
    
    STEEL = Steel()
    STEEL_USS = STEEL.ultimate_shear_strength

    # Calculate Skin Weight Per Unit Area (arealWeight) based on material
    # properties. In this instance we assume the use of
    # Bi-directional Carbon Fiber, a Honeycomb Core, and Paint:

    arealWeight =(
          BiCF_MGT*BiCF_DEN
        + HCMB_MGT * HCMB_DEN
        + PAINT_MGT * PAINT_DEN
        )

    # Calculate fuselage area (using assumption of ellipsoid), and weight:

    S_wet = 4 * np.pi * (((fLength * fWidth/4)**1.6
        + (fLength * fHeight/4)**1.6
        + (fWidth * fHeight/4)**1.6)/3)**(1/1.6)
    skinMass = S_wet * arealWeight

    # Calculate the mass of a structural bulkhead

    bulkheadMass = 3 * np.pi * fHeight * fWidth/4 * arealWeight

    # Calculate the mass of a canopy, assuming Acrylic:

    canopyMass = S_wet/8 * ACRYL_MGT * ACRYL_DEN

    # Calculate keel mass needed to carry lifting moment, assuming
    # Uni-directional Carbon Fiber used to carry load

    L_max       = G_max * MTOW * 9.8 * SF  # Max Lifting Load
    M_lift      = L_max * fLength/2.       # Max Moment Due to Lift
    beamWidth   = fWidth/3.                # Allowable Keel Width
    beamHeight  = fHeight/10.              # Allowable Keel Height

    beamArea    = M_lift * beamHeight/(4*UniCF_UTS*(beamHeight/2)**2)
    massKeel    = beamArea * fLength * UniCF_DEN

    # Calculate keel mass needed to carry wing bending moment, assuming
    # thin walled Bi-directional Carbon Fiber used to carry load

    M_bend      = L_max/2 * maxSpan/2                           # Max Bending Moment
    beamArea    = beamHeight * beamWidth                        # Enclosed Beam Area
    beamThk     = 0.5 * M_bend/(BiCF_USS * beamArea)            # Beam Thickness
    massKeel   += 2*(beamHeight + beamWidth)*beamThk*BiCF_DEN

    # Calculate keel mass needed to carry landing impact load assuming
    # Steel bolts, and Bi-directional Carbon Fiber laminate pads used to carry
    # loads in a side landing

    F_landing   = SF * MTOW * 9.8 * LIF * 0.6403        # Side Landing Force
    boltArea    = F_landing/STEEL_USS                   # Required Bolt Area
    boltDiam    = 2 * np.sqrt(boltArea/np.pi)           # Bolt Diameter
    lamThk      = F_landing/(boltDiam*BiCF_UBS)         # Laminate Thickness
    lamVol      = (np.pi*(20*lamThk)**2)*(lamThk/3)     # Laminate Pad volume
    massKeel   += 4*lamVol*BiCF_DEN                     # Mass of 4 Pads

    # Calculate total mass as the sum of skin mass, bulkhead mass, canopy pass,
    # and keel mass. Called weight by SUAVE convention

    weight = skinMass + bulkheadMass + canopyMass + massKeel

    return weight