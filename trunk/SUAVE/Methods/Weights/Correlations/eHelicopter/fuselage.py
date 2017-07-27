# fuselage.py
#
# Created: Jun 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Units
from SUAVE.Attributes import Solids                     
import numpy as np


#-------------------------------------------------------------------------------
# Fuselage
#-------------------------------------------------------------------------------

def fuselage(fLength, fWidth, fHeight, wingspan, MTOW):
    """ weight = SUAVE.Methods.Weights.Correlations.eHelicopter.fuselage(
            fLength,
            fWidth,
            fHeight,
            wingspan,
            MTOW
        )

        Calculates the structural mass of a fuselage for an eVTOL vehicle,
        assuming a structural keel taking bending an torsional loads.

        Assumes an elliptical fuselage. Intended for use with the following
        SUAVE vehicle types, but may be used elsewhere:

            eHelicopter
            eTiltwing
            eTiltrotor
            eStopped_Rotor

        Originally written as part of an AA 290 project intended for trade study
        of the above vehicle types.

        Inputs:

            fLength:    Fuselage Length [m]
            fWidth:     Fuselage Width  [m]
            fHeight:    Fuselage Height [m]
            wingspan:   Wingspan        [m]
            MTOW:       Max TO weight   [kg]

        Outputs:

            weight:     Fuselage Mass   [kg]
    """

    G_max   = 3.8    # Maximum G's Experienced During Climb
    LIF     = 3.5    # Landing Load Impact Factor
    SF      = 1.5    # Factor of Safety

    # Calculate Skin Weight Per Unit Area (arealWeight) based on material
    # properties. In this instance we assume the use of
    # Bi-directional Carbon Fiber, a Honeycomb Core, and Paint:

    arealWeight =(
          mats.BiCF.minThk      * mats.BiCF.density
        + mats.Honeycomb.minThk * mats.Honeycomb.density
        + mats.Paint.minThk     * mats.Paint.density
        )

    # Calculate fuselage area (using assumption of ellipsoid), and weight:

    S_wet = 4 * np.pi * (((fLength * fWidth/4)**1.6
        + (fLength * fHeight/4)**1.6
        + (fWidth * fHeight/4)**1.6)/3)**(1/1.6)
    skinMass = S_wet * arealWeight

    # Calculate the mass of a structural bulkhead

    bulkheadMass = 3 * np.pi * fHeight * fWidth/4 * arealWeight

    # Calculate the mass of a canopy, assuming Acrylic:

    canopyMass = S_wet/8 * mats.Acrylic.minThk * mats.Acrylic.density

    # Calculate keel mass needed to carry lifting moment, assuming
    # Uni-directional Carbon Fiber used to carry load

    L_max       = G_max * MTOW * SF       # Max Lifting Load
    M_lift      = L_max * fLength/2       # Max Moment Due to Lift
    beamWidth   = fWidth/3                # Allowable Keel Width
    beamHeight  = fHeight/10              # Allowable Keel Height

    beamArea    = M_lift * beamHeight/(4*mats.UniCF.UTS*(beamHeight/2)**2)
    massKeel    = beamArea * fLength * mats.UniCF.density

    # Calculate keel mass needed to carry wing bending moment, assuming
    # thin walled Bi-directional Carbon Fiber used to carry load

    M_bend      = 0.25 * L_max * wingspan/2                 # Max Bending Moment
    beamArea    = beamHeight * beamWidth                    # Enclosed Beam Area
    beamThk     = 0.5 * M_bend/(mats.BiCF.USS*beamArea)     # Beam Thickness
    massKeel   += 2*(beamHeight + beamWidth)*beamThk*mats.BiCF.density

    # Calculate keel mass needed to carry landing impact load assuming
    # Steel bolts, and Bi-directional Carbon Fiber laminate pads used to carry
    # loads in a side landing

    F_landing   = SF * MTOW * LIF * 0.6403              # Side Landing Force
    boltArea    = F_landing/mats.Steel.USS              # Required Bolt Area
    boltDiam    = 2 * np.sqrt(boltArea/np.pi)           # Bolt Diameter
    lamThk      = F_landing/(boltDiam*mats.BiCF.UBS)    # Laminate Thickness
    lamVol      = (np.pi*(20*lamThk)**2)*(lamThk/3)     # Laminate Pad volume
    massKeel   += 4*lamVol*mats.BiCF.density            # Mass of 4 Pads

    # Calculate total mass as the sum of skin mass, bulkhead mass, canopy pass,
    # and keel mass. Called weight by SUAVE convention

    weight = skinMass + bulkheadMass + canopyMass + massKeel

    return weight
