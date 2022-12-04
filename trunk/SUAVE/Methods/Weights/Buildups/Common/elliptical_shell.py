## @ingroup Methods-Weights-Buildups-Common
# elliptical_shell.py
# 
# Created:    Nov 2022, J. Smart
# Modified:   Dec 2022, J. Smart

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

import warnings

from SUAVE.Core import Data, Container
from SUAVE.Attributes.Solids import (
    Bidirectional_Carbon_Fiber as BiCRFP,
    Carbon_Fiber_Honeycomb as CFHoneycomb,
    Paint
)

from SUAVE.Methods.Weights.Buildups.Common.stack_mass import stack_mass

import numpy as np

## @ingroup Methods-Weights-Buildups-Common
def elliptical_shell(component,
                     skin_materials=None,
                     *args, **kwargs):
    """Calculates the structural mass associated with a non-loadbearing
    elliptical shell.

    Assumptions:
    Intended for use with fuselage and nacelle mass estimation for eVTOLs.
    If skin materials are not specified, a stack of composite materials is
    assumed and assigned.

    If the component does not have a specified wetted area, the wetted area
    will be approximated based on the length, width, height, and/or diameter
    of the component. If no wetted area or sufficient dimensions are specified,
    the component will be assigned zero mass.


    Source:
    None

    Inputs:
        components      [Lofted Body]

    Outputs: 
        mass            [kg]

    Properties Used:
    N/A	
    """

    # Determine the component's skin area

    try:
        shell_area = component.areas.wetted_area
    except:
        shell_area = area_backup(component)

    if shell_area == 0:
        shell_area = area_backup(component)

    # Determine mass per unit area

    if skin_materials is not None: # Function call overrides component materials

        shell_areal_mass = stack_mass(skin_materials)

    else:

        try:
            shell_areal_mass = stack_mass(component.skin_materials)

        except AttributeError: # If no skin materials is specified
            component.materials             = Data()
            component.materials.skin        = Container()
            component.materials.skin.base   = BiCRFP()
            component.materials.skin.core   = CFHoneycomb()
            component.materials.skin.cover  = Paint()

            shell_areal_mass = stack_mass(component.materials.skin)

    shell_mass = shell_area * shell_areal_mass

    return shell_mass

def area_backup(component):

    try:  # Fuselage Style Geometry
        l = component.lengths.total
        h = component.heights.maximum
        w = component.width

        # Thomsen elliptical approximation

        shell_area = 4 * np.pi * (((l * w / 4) ** 1.6
                                   + (l * h / 4) ** 1.6
                                   + (w * h / 4) ** 1.6) / 3) ** (1 / 1.6)
    except:  # Nacelle Style Geometry
        try:
            l = component.length
            h = component.diameter
            w = component.diameter

            # Thomsen elliptical approximation

            shell_area = 4 * np.pi * (((l * w / 4) ** 1.6
                                       + (l * h / 4) ** 1.6
                                       + (w * h / 4) ** 1.6) / 3) ** (1 / 1.6)
        except:
            warnings.warn(f"{component.tag} has insufficient geometry for " +
                          "weight estimation. Assigning zero mass.",
                          stacklevel=1)
            shell_area = 0.

    return shell_area