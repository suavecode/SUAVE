import SUAVE
from SUAVE.Core import Units, Data
import numpy as np
import matplotlib.pyplot as plt


def wing_weight_FLOPS(vehicle, WPOD, complexity):
    SW = vehicle.reference_area / (Units.ft ** 2)  # Reference wing area, ft^2
    GLOV = 0  # Gloved area, assumed 0
    SX = SW - GLOV  # Wing trapezoidal area
    wing = vehicle.wings['main_wing']
    SPAN = wing.spans.projected / Units.ft  # Wing span, ft
    SEMISPAN = SPAN / 2
    AR = SPAN ** 2 / SX  # Aspect ratio
    TR = vehicle.wings['main_wing'].taper  # Taper
    if AR <= 5:
        CAYA = 0
    else:
        CAYA = AR - 5
    FAERT = 0  # Aeroelastic tailoring factor [0 no aeroelastic tailoring, 1 maximum aeroelastic tailoring]
    FSTRT = 0  # Wing strut bracing factor [0 for no struts, 1 for struts]
    propulsor_name = list(vehicle.propulsors.keys())[0]
    propulsors = vehicle.propulsors[propulsor_name]
    NEW = sum(propulsors.wing_mounted)
    DG = vehicle.mass_properties.max_takeoff / Units.lbs  # Design gross weight in lb

    if complexity == 'Simple':
        EMS = 1 - 0.25 * FSTRT  # Wing strut bracing factor
        TLAM = np.tan(wing.sweeps.quarter_chord) \
               - 2 * (1 - TR) / (AR * (1 + TR))  # Tangent of the 3/4 chord sweep angle
        SLAM = TLAM / np.sqrt(1 + TLAM ** 2)  # sine of 3/4 chord wing sweep angle
        C6 = 0.5 * FAERT - 0.16 * FSTRT
        C4 = 1 - 0.5 * FAERT
        CAYL = (1.0 - SLAM ** 2) * \
               (1.0 + C6 * SLAM ** 2 + 0.03 * CAYA * C4 * SLAM)  # Wing sweep factor due to aeroelastic tailoring
        TCA = wing.thickness_to_chord
        BT = 0.215 * (0.37 + 0.7 * TR) * (SPAN ** 2 / SW) ** EMS / (CAYL * TCA)  # Bending factor
        CAYE = 1 - 0.03 * NEW

    else:
        NSD = 500
        N2 = int(sum(propulsors.wing_mounted) / 2)
        ETA, C, T, SWP = generate_wing_stations(vehicle)
        NS, Y = generate_int_stations(NSD, ETA)
        EETA = get_spanwise_engine(propulsors, SEMISPAN)
        P0 = calculate_load(ETA[-1], SEMISPAN)
        ASW = 0
        EM = 0
        EL = 0
        C0 = C[-1]
        W = 0
        S = 0
        A0 = 0
        EEL = 0
        NE = 0
        EEM = 0
        EA0 = 0
        EW = 0

        for i in range(NS - 1, 1, -1):
            Y1 = Y[i]
            DY = Y[i + 1] - Y1
            P1 = calculate_load(Y1, SEMISPAN)
            C1 = np.interp(Y1, ETA, C)
            T1 = np.interp(Y1, ETA, T)
            SWP1 = find_sweep(Y1, ETA, SWP)
            ASW = ASW + (DY + 2 * Y1) * DY * SWP1
            DELP = DY / 6 * (C0 * (2 * P0 + P1) + C1 * (2 * P1 + P0))
            DELM = DY ** 2 * (C0 * (3.0 * P0 + P1) + C1 * (P1 + P0)) / 12.
            EM = EM + (DELM + DY * EL) * 1 / np.cos(SWP1 * np.pi / 180)
            A1 = EM * 1 / np.cos(SWP1 * np.pi / 180) * 1 / (C1 * T1)
            W = W + (A0 + A1) * DY / 2.
            S = S + (C0 + C1) * DY / 2.
            EL = EL + DELP
            A0 = A1
            C0 = C1
            P0 = P1
            if N2 > 0:
                DELM = DY * EEL
                if NE < N2:
                    if Y1 <= EETA[N2 - NE - 1]:
                        DELM = DELM - Y1 + EETA[N2 - NE - 1]
                        EEL = EEL + 1
                        NE = NE + 1
                EEM = EEM + DELM * 1 / np.cos(SWP1 * np.pi / 180)
                EA1 = EEM * 1 / np.cos(SWP1 * np.pi / 180) * 1 / (C1 * T1)
                EW = EW + (EA0 + EA1) * DY / 2
                EA0 = EA1
        EM = EM / EL
        W = 4. * W / EL
        EW = 8. * EW
        SA = np.sin(ASW * np.pi / 180)
        AR = 2 / S
        if AR <= 5:
            CAYA = 0
        else:
            CAYA = AR - 5
        DEN = AR ** (.25 * FSTRT) * (1.0 + (.50 * FAERT - .160 * FSTRT) * SA ** 2 /
                                     + .03 * CAYA * (1.0 - .50 * FAERT) * SA)
        BT = W / DEN
        BTE = EW
        CAYE = 1
        if NEW > 0:
            CAYE = 1 - BTE / BT * WPOD / DG

    A = wing_weight_constants_FLOPS(vehicle.systems.accessories)  # Wing weight constants
    FCOMP = 0.5  # Composite utilization factor [0 no composite, 1 full composite]
    ULF = 3.75  # vehicle.envelope.ultimate_load #vehicle.envelope.ultimate_load  # Structural ultimate load factor, 3.75 default
    CAYF = 1  # Multiple fuselage factor [1 one fuselage, 0.5 multiple fuselages]
    VFACT = 1  # Variable sweep factor (if wings can rotate)
    PCTL = 1  # Fraction of load carried by this wing
    W1NIR = A[0] * BT * (1 + np.sqrt(A[1] / SPAN)) * ULF * SPAN * (1 - 0.4 * FCOMP) * (
            1 - 0.1 * FAERT) * CAYF * VFACT * PCTL / 10.0 ** 6  # Wing bending material weight lb
    SFLAP = vehicle.flap_ratio * SX

    W2 = A[2] * (1 - 0.17 * FCOMP) * SFLAP ** (A[3]) * DG ** (A[4])
    W3 = A[5] * (1 - 0.3 * FCOMP) * SW ** (A[6])
    W1 = (DG * CAYE * W1NIR + W2 + W3) / (1 + W1NIR) - W2 - W3
    WWING = W1 + W2 + W3  # Total wing weight

    return WWING * Units.lbs


def generate_wing_stations(vehicle):
    wing = vehicle.wings['main_wing']
    SPAN = wing.spans.projected / Units.ft  # Wing span, ft
    SEMISPAN = SPAN / 2
    root_chord = wing.chords.root / Units.ft
    num_seg = len(wing.Segments.keys())

    if num_seg == 0:
        segment = SUAVE.Components.Wings.Segment()
        segment.tag = 'root'
        segment.percent_span_location = 0.
        segment.twist = wing.twists.root
        segment.root_chord_percent = 1
        segment.dihedral_outboard = 0.
        segment.sweeps.quarter_chord = wing.sweeps.quarter_chord
        segment.thickness_to_chord = wing.thickness_to_chord
        wing.Segments.append(segment)

        segment = SUAVE.Components.Wings.Segment()
        segment.tag = 'tip'
        segment.percent_span_location = 1.
        segment.twist = wing.twists.tip
        segment.root_chord_percent = wing.chords.tip / wing.chords.root
        segment.dihedral_outboard = 0.
        segment.sweeps.quarter_chord = wing.sweeps.quarter_chord
        segment.thickness_to_chord = wing.thickness_to_chord
        wing.Segments.append(segment)
        num_seg = len(wing.Segments.keys())
    ETA = np.zeros(num_seg + 1)
    C = np.zeros(num_seg + 1)
    T = np.zeros(num_seg + 1)
    SWP = np.zeros(num_seg + 1)
    ETA[0] = wing.Segments[0].percent_span_location
    C[0] = root_chord * wing.Segments[0].root_chord_percent * 1 / SEMISPAN
    SWP[0] = 0
    if hasattr(wing.Segments[0], 'thickness_to_chord'):
        T[0] = wing.Segments[0].thickness_to_chord
    else:
        T[0] = wing.thickness_to_chord
    ETA[1] = vehicle.fuselages.fuselage.width / 2 * 1 / Units.ft * 1 / SEMISPAN
    C[1] = determine_fuselage_chord(vehicle) * 1/SEMISPAN

    if hasattr(wing.Segments[0], 'thickness_to_chord'):
        T[1] = wing.Segments[0].thickness_to_chord
    else:
        T[1] = wing.thickness_to_chord
    for i in range(1, num_seg):
        ETA[i + 1] = wing.Segments[i].percent_span_location
        C[i + 1] = root_chord * wing.Segments[i].root_chord_percent * 1 / SEMISPAN
        if hasattr(wing.Segments[i], 'thickness_to_chord'):
            T[i + 1] = wing.Segments[i].thickness_to_chord
        else:
            T[i + 1] = wing.thickness_to_chord
        SWP[i] = np.arctan(np.tan(wing.Segments[i-1].sweeps.quarter_chord) - (C[i-1] - C[i]))
    SWP[-1] = np.arctan(np.tan(wing.Segments[-2].sweeps.quarter_chord) - (C[-2] - C[-1]))
    return ETA, C, T, SWP


def generate_int_stations(NSD, ETA):
    Y = [ETA[1]]
    desired_int = (ETA[-1] - ETA[1]) / NSD
    NS = 0
    for i in range(2, len(ETA)):
        NP = int((ETA[i] - ETA[i - 1]) / desired_int + 0.5)
        if NP < 1:
            NP = 1
        AINT = (ETA[i] - ETA[i - 1]) / NP
        for j in range(NP):
            NS = NS + 1
            Y.append(Y[-1] + AINT)
    return NS, Y


def calculate_load(ETA, semispan):
    PS = np.sqrt(1. - (ETA) ** 2)
    return PS


def find_sweep(y, lst_y, swp):
    diff = lst_y - y
    for i in range(len(diff)):
        if diff[i] > 0:
            return swp[i - 1]
        elif diff[i] == 0:
            return swp[i]
    return swp[-1]


def get_spanwise_engine(propulsors, SEMISPAN):
    N2 = int(sum(propulsors.wing_mounted) / 2)
    EETA = np.zeros(N2)
    idx = 0
    for i in range(int(propulsors.number_of_engines)):
        if propulsors.wing_mounted[i] and propulsors.origin[i][1] > 0:
            EETA[idx] = (propulsors.origin[i][1] / Units.ft) * 1 / SEMISPAN
            idx += 1
    return EETA


def wing_weight_constants_FLOPS(ac_type):
    if ac_type == "short_range" or ac_type == "business" or \
            ac_type == "commuter":
        A = [30.0, 0., 0.25, 0.5, 0.5, 0.16, 1.2]
    else:
        A = [8.8, 6.25, 0.68, 0.34, 0.6, 0.035, 1.5]
    return A


def determine_fuselage_chord(vehicle):
    wing = vehicle.wings['main_wing']
    root_chord = wing.chords.root / Units.ft
    SPAN = wing.spans.projected / Units.ft  # Wing span, ft
    SEMISPAN = SPAN / 2
    c1 = root_chord * wing.Segments[0].root_chord_percent
    c2 = root_chord * wing.Segments[1].root_chord_percent
    y1 = wing.Segments[0].percent_span_location
    y2 = wing.Segments[1].percent_span_location
    b = (y2 - y1) * SEMISPAN
    taper = c2 / c1
    y = vehicle.fuselages.fuselage.width / 2 * 1 / Units.ft
    chord = c1 * (1 - (1-taper) * 2 * y/b)
    return chord
