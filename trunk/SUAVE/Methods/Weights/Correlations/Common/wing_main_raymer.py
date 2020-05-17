## @ingroup Methods-Weights-Correlations-Common
# wing_main.py
#
# Created:  Jan 2014, A. Wendorff
# Modified: Feb 2014, A. Wendorff
#           Feb 2016, E. Botero
#           Jul 2017, M. Clarke
#           Mar 2019, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units
import numpy as np


# ----------------------------------------------------------------------
#   Wing Main
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Common
def wing_main_raymer(vehicle, wing):
    """ Calculate the wing weight of the aircraft based on the fully-stressed
    bending weight of the wing box

    Assumptions:
        calculated total wing weight based on a bending index and actual data
        from 15 transport aircraft

    Source:
        http://aerodesign.stanford.edu/aircraftdesign/AircraftDesign.html
        search for: Derivation of the Wing Weight Index

    Inputs:
        wing
             .span.projected                         [meters**2]
             .taper                                  [dimensionless]
             .sweeps.quarter_chord                   [radians]
             .thickness_to_chord                     [dimensionless]
             .Segments
                 .root_chord_percent                 [dimensionless]
                 .thickness_to_chord                 [dimensionless]
                 .percent_span_location              [dimensionless]
        Nult - ultimate load factor of the aircraft  [dimensionless]
        TOW - maximum takeoff weight of the aircraft [kilograms]
        wt_zf - zero fuel weight of the aircraft     [kilograms]

    Outputs:
        weight - weight of the wing                  [kilograms]

    Properties Used:
        N/A
    """

    # unpack inputs
    span = wing.spans.projected
    taper = wing.taper
    sweep = wing.sweeps.quarter_chord
    area = wing.areas.reference
    t_c_w = wing.thickness_to_chord

    Wdg = vehicle.mass_properties.max_takeoff / Units.lb
    Nz = vehicle.envelope.ultimate_load
    Sw = area / Units.ft ** 2
    A = wing.aspect_ratio
    tc_root = t_c_w
    taper = taper
    sweep = sweep
    Scsw = Sw * .1

    if vehicle.systems.accessories == 'sst':
        sweep = 0
    Wwing = 0.0051 * (Wdg * Nz) ** .557 * Sw ** .649 * A ** .5 * tc_root ** -.4 * (1 + taper) ** .1 * np.cos(
        sweep) ** -1. * Scsw ** .1
    weight = Wwing * Units.lb

    return weight
