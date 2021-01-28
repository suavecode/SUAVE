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
def wing_main(vehicle, wing, rho, sigma, computation_type = 'segmented'):
    """ Calculate the wing weight of the aircraft based on the fully-stressed
    bending weight of the wing box

    Assumptions:
        calculated total wing weight based on a bending index and actual data
        from 15 transport aircraft

    Source:
        http://aerodesign.stanford.edu/aircraftdesign/AircraftDesign.html
        search for: Derivation of the Wing Weight Index

    Inputs:
        vehicle - data dictionary with vehicle properties                   [dimensionless]
            -.mass_properties.max_takeoff: MTOW                             [kilograms]
            -.mass_properties.max_zero_fuel: zero fuel weight aircraft      [kilograms]
            -.envelope.ultimate_load: ultimate load factor
        wing    - data dictionary with specific wing properties             [dimensionless]
            -.areas.reference: wing reference surface area                  [m^2]
            -.sweeps.quarter_chord: quarter chord sweep angle               [deg]
            -.spans.projected: wing span                                    [m]
            -.thickness_to_chord: thickness to chord of wing
            -.taper: taper ratio of wing
            -.chords.root: root chord                                       [m]

    Outputs:
        weight - weight of the wing                  [kilograms]

    Properties Used:
        N/A
    """

    # unpack inputs
    span        = wing.spans.projected
    taper       = wing.taper
    sweep       = wing.sweeps.quarter_chord
    area        = wing.areas.reference
    t_c_w       = wing.thickness_to_chord
    RC          = wing.chords.root
    frac        = wing.areas.reference / vehicle.reference_area
    rho_sigma   = rho * 9.81 / sigma
    Nult        = vehicle.envelope.ultimate_load
    TOW         = vehicle.mass_properties.max_takeoff
    wt_zf       = vehicle.mass_properties.max_zero_fuel

    # Start the calculations
    l_tot   = Nult * np.sqrt(TOW * wt_zf) * 9.81
    L0      = frac * 2 * l_tot / (span * np.pi)

    if len(wing.Segments) > 0 and computation_type == 'segmented':

        # Prime some numbers
        run_sum = 0
        b       = span

        for i in range(1, len(wing.Segments)):

            # Unpack segment level info
            Y1 = wing.Segments[i - 1].percent_span_location
            Y2 = wing.Segments[i].percent_span_location

            if wing.Segments[i - 1].root_chord_percent == wing.Segments[i].root_chord_percent and \
                    wing.Segments[i - 1].thickness_to_chord == wing.Segments[i].thickness_to_chord:
                C   = wing.Segments[i].root_chord_percent * RC
                G   = wing.Segments[i].thickness_to_chord
                SW  = wing.Segments[i - 1].sweeps.quarter_chord

                WB = (1 / (G * C * np.cos(SW) ** 2)) * 1 / 3 * (
                        1 / 8 * (-Y1 * (5 - 2 * Y1 ** 2) * np.sqrt(1 - Y1 ** 2) -
                                 3 * np.arcsin(Y1)) + 1 / 8 * (
                                Y2 * (5 - 2 * Y2 ** 2) * np.sqrt(1 - Y2 ** 2) + 3 * np.arcsin(Y2)))

            else:
                # A is the root thickness
                A = RC * wing.Segments[i - 1].root_chord_percent * wing.Segments[i - 1].thickness_to_chord
                # B is the slope of the thickness
                B = (A - RC * wing.Segments[i].root_chord_percent * wing.Segments[i].thickness_to_chord) / (
                        wing.Segments[i].percent_span_location - wing.Segments[i - 1].percent_span_location)
                # C is the offset
                C = wing.Segments[i - 1].percent_span_location
                SW = wing.Segments[i - 1].sweeps.quarter_chord

                WB1 = big_integral(Y1, A, B, C)
                WB2 = big_integral(Y2, A, B, C)

                WB = (WB2 - WB1) / (np.cos(SW) ** 2)

            run_sum += np.real(WB)

        weight_factor = rho_sigma * (b ** 2) * L0 * run_sum / 2

        weight = 4.22 * area / Units.feet ** 2 + (weight_factor / Units.lb)

    else:

        area    = wing.areas.reference / Units.ft ** 2
        span    = wing.spans.projected / Units.ft
        mtow    = TOW / Units.lb  # Convert kg to lbs
        zfw     = wt_zf / Units.lb  # Convert kg to lbs

        # Calculate weight of wing for traditional aircraft wing
        weight  = 4.22 * area + 1.642 * 10. ** -6. * Nult * (span) ** 3. * (mtow * zfw) ** 0.5 \
                 * (1. + 2. * taper) / (t_c_w * (np.cos(sweep)) ** 2. * area * (1. + taper))


    weight = weight * Units.lb  # Convert lb to kg

    return weight


def big_integral(x, A, B, C):
    """ Integrate the wing bending moment over a section

    Assumptions:
        Linearly tapering thickness

    Source:
        Botero 2019

    Inputs:
        x - span wise station      [dimensionless]
        A - origin thickness       [meters]
        B - taper ratio of section [dimensionless]
        C - origin offset          [dimensionless]

    Outputs:
        result - evaluate one side of an indefinite integral [meters^-1]

    Properties Used:
        N/A
    """

    results = (1/(4*B**3))*(2*A**2*(np.pi)*x+4*A*B*C*(np.pi)*x+ \
            B**2*(-1+2*C**2)*(np.pi)*x- \
            2*(2*A**2+4*A*B*C+B**2*(-1+2*C**2))*np.sqrt(1-x**2)+ \
            2/3*B*np.sqrt(1-x**2)*(3*A*x+B*(-1+3*C*x+x**2))+ \
            2*B*(A+B*C)*np.arcsin(x)- \
            2*(2*A**2+4*A*B*C+B**2*(-1+2*C**2))*x*np.arcsin(x)-( \
            4*(A+B*C)**2*np.sqrt(0j+-A**2-2*A*B*C-B**2*(-1+C**2))* \
            np.log(A+B*C-B*x))/B-( \
            4*(A**3+3*A**2*B*C+B**3*C*(-1+C**2)+A*B**2*(-1+3*C**2))*x*np.log( \
            A+B*C-B*x))/np.sqrt(0j+-A**2-2*A*B*C-B**2*(-1+C**2))+( \
            8*(A+B*C)**2*np.sqrt(0j-A**2-2*A*B*C-B**2*(-1+C**2))* \
            np.log(0j-A-B*C+B*x))/ \
            B+(4*(A**3+3*A**2*B*C+B**3*C*(-1+C**2)+ \
            A*B**2*(-1+3*C**2))*x*np.log(0j-B+A*x+B*C*x- \
            np.sqrt(0j+-A**2+B**2-2*A*B*C-B**2*C**2)*np.sqrt( \
            1-x**2)))/(np.sqrt(0j-A**2-2*A*B*C-B**2*(-1+C**2)))-(1/B)* \
            4*(A+B*C)**2*np.sqrt(0j-A**2-2*A*B*C-B**2*(-1+C**2))* \
            np.log(-B+A*x+B*C*x+ \
            np.sqrt(0j-A**2+B**2-2*A*B*C-B**2*C**2)*np.sqrt(1-x**2))+(1/B)* 4*(A+B*C)*np.sqrt(0j-(A**2+2*A*B*C+B**2*(-1+C**2))**2)*np.log(A**2*x+2*A*B*C*x+B**2*(-1+C**2)*x+ \
            np.sqrt(0j-(A**2+2*A*B*C+B**2*(-1+C**2))**2)*np.sqrt(1-x**2)))

    return results
