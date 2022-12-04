## @ingroup Methods-Weights-Buildups-Common
# elliptical_shell.py
# 
# Created:    Nov 2022, J. Smart
# Modified:   Dec 2022, J. Smart

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

import warnings

from SUAVE.Core import Units
from SUAVE.Attributes.Solids import (
    Bidirectional_Carbon_Fiber,
    Unidirectional_Carbon_Fiber,
    Carbon_Fiber_Honeycomb,
    Paint,
    Acrylic
)

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
    assumed.

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
        try:    # Fuselage Style Geometry
            l   = component.lengths.total
            h   = component.heights.maximum
            w   = component.width

            # Thomsen elliptical approximation

            shell_area = 4 * np.pi * (((l * w/4)**1.6
                                    + (l * h/4)**1.6
                                    + (w * h/4)**1.6)/3)**(1/1.6)
        except: # Nacelle Style Geometry
            try:
                l = component.length
                h = component.diameter
                w - component.diameter

                # Thomsen elliptical approximation

                shell_area = 4 * np.pi * (((l * w/4)**1.6
                                        + (l * h/4)**1.6
                                        + (w * h/4)**1.6)/3)**(1/1.6)
            except:
                warnings.warn(f"{component.tag} has insufficient geometry for "+
                              "weight estimation. Assigning zero mass.",
                              stacklevel=1)

    # Determine mass per unit area

    if skin_materials is not None:

        # Allow override of component assigned skin materials (e.g. for canopy)

        shell_areal_mass = np.sum([(mat.minimum_gage_thickness * mat.density)
                                  for mat in skin_materials])

    else:
        try:
            shell_areal_mass = np.sum([(mat.minimum_gage_thickness * mat.density)
                                      for mat in component.skin_materials])
        except AttributeError:
            shell_areal_mass = 1.2995 # Stack of BiCFRP, Honeycomb, Paint

    shell_mass = shell_area * shell_areal_mass

    return shell_mass
