## @ingroup Methods-Weights-Buildups-Common
# rotor_boom.py
# 
# Created:    Dec 2022, J. Smart
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

from SUAVE.Core import Data, Container, Units
from SUAVE.Components.Energy.Networks import Battery_Propeller, Lift_Cruise
from SUAVE.Attributes.Solids import (
    Bidirectional_Carbon_Fiber as BiCRFP,
    Unidirectional_Carbon_Fiber as UniCRFP,
    Carbon_Fiber_Honeycomb as CFHoneycomb,
    Paint,
    Acrylic,
    Steel)

from SUAVE.Methods.Weights.Buildups.Common import elliptical_shell
from SUAVE.Methods.Weights.Buildups.Common import stack_mass

import numpy as np

## @ingroup Methods-Weights-Buildups-Common
def rotor_boom(boom,
               config,
               safety_factor = 1.5,
               *args, **kwargs):
    """ Calculates the structural mass of a rotor for an eVTOL vehicle,
        assuming a structural keel taking bending loads.

        Assumptions:
        Assumes n-2 engine out scenario for maximum rotor thrust and two rotors
        per boom

        Sources:


        Inputs:

            rotor_boom                                  [SUAVE.Rotor_Boom]
                .number_of_rotors                       [int]
            config                                      [SUAVE.config]
                .networks.number_of_lift_rotor_engines  [int]   If Lift-Cruise
                .networks.

            safety_factor   [Unitless]  Loading Condition Factor of Safety

        Outputs:

            weight:                 Estimated Fuselage Mass             [kg]

        Properties Used:
        Material Properties of Imported SUAVE Solids
    """

    # --------------------------------------------------------------------------
    # Unpack Materials
    # --------------------------------------------------------------------------

    try:
        mats = boom.materials
    except AttributeError:
        boom.materials              = Data()
        boom.materials.keel         = Container()
        boom.materials.skin         = Container()
        boom.materials.canopy       = Container()
        boom.materials.landing_gear = Container()

    try:
        rbmMat = boom.materials.keel.bending_carrier
    except AttributeError:
        boom.materials.keel.bending_carrier = UniCRFP()
        rbmMat = boom.materials.keel.bending_carrier

    rbmDen = rbmMat.density
    rbmUTS = rbmMat.ultimate_tensile_strength

    # Shear Carrier

    try:
        shearMat = boom.materials.keel.shear_carrier
    except AttributeError:
        boom.materials.keel.shear_carrier = BiCRFP()
        shearMat = boom.materials.keel.shear_carrier

    shearDen = shearMat.density
    shearUSS = shearMat.ultimate_shear_strength

    l = boom.length
    d = boom.diameter
    r = d/2
    h = r * np.sqrt(2)

    MTOW = config.mass_properties.max_takeoff
    n_rotors = 0

    for network in config.networks:
        if isinstance(network, Battery_Propeller):
            n_rotors += network.number_of_propeller_engines
        elif isinstance(network, Lift_Cruise):
            n_rotors += network.number_of_lift_rotor_engines
        else:
            continue

    rotor_thrust = MTOW/(n_rotors - 2) * 9.8    # Thrust Per Rotor w/ 2EO
    arm_length = l/boom.number_of_rotors        # Moment Arm

    # Bending Mass

    M_max = rotor_thrust * arm_length           # Maximum Moment
    A_bend = M_max * h/(4*rbmUTS*(h/2)**2)      # Required Bending Cross-Section

    keelMass = A_bend * l * rbmDen

    # Shear Mass

    A_shear = h**2                              # Beam Enclosed Area
    t       = 0.5 * M_max/(shearUSS*A_shear)    # Beam Thickness

    keelMass += (4*h)*t*shearDen

    # --------------------------------------------------------------------------
    # Skin Mass
    # --------------------------------------------------------------------------

    skinMass = elliptical_shell(boom)

    # --------------------------------------------------------------------------
    # Pack Results
    # --------------------------------------------------------------------------

    results         = Data()
    results.total   = (keelMass + skinMass) * Units.kg
    results.keel    = keelMass              * Units.kg
    results.skin    = skinMass              * Units.kg

    return results
