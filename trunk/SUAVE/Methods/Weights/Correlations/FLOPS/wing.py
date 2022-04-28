## @ingroup Methods-Weights-Correlations-FLOPS
# wing.py
#
# Created:  May 2020, W. Van Gijseghem
# Modified: Aug 2021, J. Mukhopadhaya

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
import numpy as np
import copy

## @ingroup Methods-Weights-Correlations-FLOPS
def wing_weight_FLOPS(vehicle, wing, WPOD, complexity, settings, num_main_wings):
    """ Calculate the wing weight based on the flops method. The wing weight consists of:
        - Total Wing Shear Material and Control Surface Weight
        - Total Wing Miscellaneous Items Weight
        - Total Wing Bending Material Weight

        Assumptions:
            Wing is elliptically loaded
            Gloved wing area is 0
            Load between multiple main wings is distributed equally
            Wing sweep is fixed

        Source:
            The Flight Optimization System Weight Estimation Method

       Inputs:
            vehicle - data dictionary with vehicle properties                   [dimensionless]
                -.reference_area: wing surface area                             [m^2]
                -.mass_properties.max_takeoff: MTOW                             [kilograms]
                -.envelope.ultimate_load: ultimate load factor (default: 3.75)
                -.systems.accessories: type of aircraft (short-range, commuter
                                                        medium-range, long-range,
                                                        sst, cargo)
                -.fuselages.fuselage.width: width of the fuselage               [m]
             -wing: data dictionary with wing properties
                    -.taper: taper ratio
                    -.sweeps.quarter_chord: quarter chord sweep angle           [deg]
                    -.thickness_to_chord: thickness to chord
                    -.spans.projected: wing span                                [m]
                    -.chords.root: root chord                                   [m]
                    -.tip.root: tip chord                                       [m]
                    -.twists.root: twist of wing at root                        [deg]
                    -.twists.tip: twist of wing at tip                          [deg]
                    -.flap_ratio: flap surface area over wing surface area
                 -.networks: data dictionary containing all propulsion properties
                    -.number_of_engines: number of engines
                    -.sealevel_static_thrust: thrust at sea level               [N]
            WPOD - weight of engine pod including the nacelle                   [kilograms]
            complexity - "simple" or "complex" depending on the wing weight method chosen

       Outputs:
            WWING - wing weight                                          [kilograms]

        Properties Used:
            N/A
    """
    SW          = wing.areas.reference / (Units.ft ** 2)  # Reference wing area, ft^2
    GLOV        = 0 
    SX          = SW - GLOV  # Wing trapezoidal area
    SPAN        = wing.spans.projected / Units.ft  # Wing span, ft
    SEMISPAN    = SPAN / 2
    AR          = SPAN ** 2 / SX  # Aspect ratio
    TR          = wing.taper  # Taper
    
    aeroelastic_tailoring_factor = settings.FLOPS.aeroelastic_tailoring_factor
    strut_braced_wing_factor     = settings.FLOPS.strut_braced_wing_factor
    composite_utilization_factor = settings.FLOPS.composite_utilization_factor
    
    if AR <= 5:
        CAYA = 0
    else:
        CAYA = AR - 5
    # Aeroelastic tailoring factor [0 no aeroelastic tailoring, 1 maximum aeroelastic tailoring]
    FAERT           = aeroelastic_tailoring_factor  
    # Wing strut bracing factor [0 for no struts, 1 for struts]
    FSTRT           = strut_braced_wing_factor
    network_name    = list(vehicle.networks.keys())[0]
    networks        = vehicle.networks[network_name]
    NEW             = sum(networks.wing_mounted)
    DG              = vehicle.mass_properties.max_takeoff / Units.lbs  # Design gross weight in lb

    if complexity == 'Simple':
        EMS  = 1 - 0.25 * FSTRT  # Wing strut bracing factor
        TLAM = np.tan(wing.sweeps.quarter_chord) \
               - 2 * (1 - TR) / (AR * (1 + TR))  # Tangent of the 3/4 chord sweep angle
        SLAM = TLAM / np.sqrt(1 + TLAM ** 2)  # sine of 3/4 chord wing sweep angle
        C6   = 0.5 * FAERT - 0.16 * FSTRT
        C4   = 1 - 0.5 * FAERT
        CAYL = (1.0 - SLAM ** 2) * \
               (1.0 + C6 * SLAM ** 2 + 0.03 * CAYA * C4 * SLAM)  # Wing sweep factor due to aeroelastic tailoring
        TCA  = wing.thickness_to_chord
        BT   = 0.215 * (0.37 + 0.7 * TR) * (SPAN ** 2 / SW) ** EMS / (CAYL * TCA)  # Bending factor
        CAYE = 1 - 0.03 * NEW

    else:
        NSD             = 500
        N2              = int(sum(networks.wing_mounted) / 2)
        ETA, C, T, SWP  = generate_wing_stations(vehicle.fuselages['fuselage'].width, copy.deepcopy(wing))
        NS, Y           = generate_int_stations(NSD, ETA)
        EETA            = get_spanwise_engine(networks, SEMISPAN)
        P0              = calculate_load(ETA[-1])
        ASW             = 0
        EM              = 0
        EL              = 0
        C0              = C[-1]
        S               = 0
        EEL             = 0
        EEM             = 0
        EA0             = 0
        EW              = 0
        
        
        # Replaced FOR LOOP
        # Reverse Order
        Y  = np.flip(Y)
        
        # DY distance
        DY = np.diff(Y)
        
        # Trim the vectors away from the tip and center
        Y  = Y[1:-2]
        DY = -DY[0:-2]
        
        # Get normalized pressure loading across the wing
        P1     = calculate_load(Y)
        P0     = np.zeros_like(P1)
        P0[1:] = P1[0:-1]
        
        # Get local chord length
        C1     = np.interp(Y, ETA, C)
        C0     = np.zeros_like(C1)
        C0[0]  = C[-1]
        C0[1:] = C1[0:-1]
        
        # Calculate local pressure load and moments (DELP and DELM)
        T1   = np.interp(Y, ETA, T)
        SWP1 = find_sweep(Y,ETA,SWP)
        DELP = DY / 6 * (C0 * (2 * P0 + P1) + C1 * (2 * P1 + P0))
        DELM = DY ** 2 * (C0 * (3.0 * P0 + P1) + C1 * (P1 + P0)) / 12.
        
        # Sum loads
        EL     = np.zeros_like(DELP) 
        EL[1:] = np.cumsum(DELP[0:-1])
        
        # Sum moments
        EM     = np.cumsum((DELM + DY * EL) * 1 / np.cos(SWP1))
        
        # Calculate required bending material area
        BMA1     = EM * 1 / np.cos(SWP1) * 1 / (C1 * T1)
        
        BMA0     = np.zeros_like(BMA1)
        BMA0[1:] = BMA1[0:-1]
        
        # Compute segment values
        ASW  = np.cumsum((DY + 2 * Y) * DY * SWP1)
        PM   = np.cumsum((BMA0 + BMA1) * DY / 2.)
        S    = np.cumsum((C0 + C1) * DY / 2.)


        # Adjust for engine loads
        if N2>0: # If there are engines
            EEL   = np.zeros_like(Y)
            DELM2 = np.zeros_like(Y)
            
            # Do a for loop over engine stations
            for ii in range(len(EETA)):
                # Find the station closest to the engine but inboard
                distances = EETA[ii]-Y
                distances[distances<0] = np.inf
                distance = np.min(distances)
                loc      = np.argmin(distances)
                DELM2[loc] = DELM2[loc] + distance
                EEL[loc+1:] = EEL[loc+1:] + 1

            DELM2 = DELM2 + EEL*DY

            EEM = np.cumsum(DELM2/np.cos(SWP1))
            EA1 = EEM * 1 / np.cos(SWP1) * 1 / (C1 * T1)
            
            EA0 = np.zeros_like(Y)
            EA0[1:] = EA1[0:-1]
            
            EW  = np.sum((EA0 + EA1) * DY / 2)
            
        # Finalize properties
        EL = EL[-1] + DELP[-1]    
        EM = EM[-1] / EL
        PM = 4. * PM[-1] / EL
        EW = 8. * EW
        SA = np.sin(ASW[-1])
        AR = 2 / S[-1]       
                
        if AR <= 5:
            CAYA = 0
        else:
            CAYA = AR - 5
        DEN = AR ** (.25 * FSTRT) * (1.0 + (.50 * FAERT - .160 * FSTRT) * SA ** 2 /
                                     + .03 * CAYA * (1.0 - .50 * FAERT) * SA)
        BT = PM / DEN
        BTE = EW
        CAYE = 1
        if NEW > 0:
            CAYE = 1 - BTE / BT * WPOD / DG

    A       = wing_weight_constants_FLOPS(vehicle.systems.accessories)  # Wing weight constants
    # Composite utilization factor [0 no composite, 1 full composite]
    FCOMP   = composite_utilization_factor  
    ULF     = vehicle.envelope.ultimate_load
    if len(vehicle.fuselages) == 1:
        CAYF    = 1  # Multiple fuselage factor [1 one fuselage, 0.5 multiple fuselages]
    elif len(vehicle.fuselage) > 1:
        CAYF    = 0.5
    else:
        raise NotImplementedError
    VFACT   = 1  # Variable sweep factor, TODO: add equation to allow variable sweep penalty
    PCTL    = 1/num_main_wings  # Fraction of load carried by this wing
    W1NIR   = A[0] * BT * (1 + np.sqrt(A[1] / SPAN)) * ULF * SPAN * (1 - 0.4 * FCOMP) * (
                1 - 0.1 * FAERT) * CAYF * VFACT * PCTL / 10.0 ** 6  # Wing bending material weight lb
    SFLAP   = wing.flap_ratio * SX

    W2 = A[2] * (1 - 0.17 * FCOMP) * SFLAP ** (A[3]) * DG ** (A[4])  # shear material weight
    W3 = A[5] * (1 - 0.3 * FCOMP) * SW ** (A[6])  # miscellaneous items weight
    W1 = (DG * CAYE * W1NIR + W2 + W3) / (1 + W1NIR) - W2 - W3  # bending material weight
    WWING = W1 + W2 + W3  # Total wing weight

    return WWING * Units.lbs

## @ingroup Methods-Weights-Correlations-FLOPS
def generate_wing_stations(fuselage_width, wing):
    """ Divides half the wing in sections, using the defined sections
        and adding a section at the intersection of wing and fuselage

        Assumptions:

        Source:
            The Flight Optimization System Weight Estimation Method

        Inputs:
            fuselage_width: fuselage width                                      [m]
            wing: data dictionary with wing properties
                    -.taper: taper ration wing
                    -.sweeps.quarter_chord: quarter chord sweep angle           [deg]
                    -.thickness_to_chord: thickness to chord
                    -.spans.projected: wing span                                [m]
                    -.chords.root: root chord                                   [m]
                    -.tip.root: tip chord                                       [m]
                    -.twists.root: twist of wing at root                        [deg]
                    -.twists.tip: twist of wing at tip                          [deg]
                    -.Segments: trapezoidal segments of the wing

       Outputs:
           ETA: spanwise location of the sections normalized by half span
           C: chord lengths at every spanwise location in ETA normalized by half span
           T: thickness to chord ratio at every span wise location in ETA
           SWP: quarter chord sweep angle at every span wise location in ETA

        Properties Used:
            N/A
    """
    SPAN        = wing.spans.projected / Units.ft  # Wing span, ft
    SEMISPAN    = SPAN / 2
    root_chord  = wing.chords.root / Units.ft
    num_seg     = len(wing.Segments.keys())

    if num_seg == 0:
        segment                         = SUAVE.Components.Wings.Segment()
        segment.tag                     = 'root'
        segment.percent_span_location   = 0.
        segment.twist                   = wing.twists.root
        segment.root_chord_percent      = 1
        segment.dihedral_outboard       = 0.
        segment.sweeps.quarter_chord    = wing.sweeps.quarter_chord
        segment.thickness_to_chord      = wing.thickness_to_chord
        wing.Segments.append(segment)

        segment                         = SUAVE.Components.Wings.Segment()
        segment.tag                     = 'tip'
        segment.percent_span_location   = 1.
        segment.twist                   = wing.twists.tip
        segment.root_chord_percent      = wing.chords.tip / wing.chords.root
        segment.dihedral_outboard       = 0.
        segment.sweeps.quarter_chord    = wing.sweeps.quarter_chord
        segment.thickness_to_chord      = wing.thickness_to_chord
        wing.Segments.append(segment)
        num_seg = len(wing.Segments.keys())
        
    ETA    = np.zeros(num_seg + 1)
    C      = np.zeros(num_seg + 1)
    T      = np.zeros(num_seg + 1)
    SWP    = np.zeros(num_seg + 1)
    ETA[0] = wing.Segments[0].percent_span_location
    C[0]   = root_chord * wing.Segments[0].root_chord_percent * 1 / SEMISPAN
    SWP[0] = 0
    
    if hasattr(wing.Segments[0], 'thickness_to_chord'):
        T[0] = wing.Segments[0].thickness_to_chord
    else:
        T[0] = wing.thickness_to_chord
    ETA[1] = fuselage_width / 2 * 1 / Units.ft * 1 / SEMISPAN
    C[1] = determine_fuselage_chord(fuselage_width, wing) * 1 / SEMISPAN

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
        SWP[i] = np.arctan(np.tan(wing.Segments[i - 1].sweeps.quarter_chord) - (C[i - 1] - C[i]))
    SWP[-1] = np.arctan(np.tan(wing.Segments[-2].sweeps.quarter_chord) - (C[-2] - C[-1]))
    return ETA, C, T, SWP

## @ingroup Methods-Weights-Correlations-FLOPS
def generate_int_stations(NSD, ETA):
    """ Divides half of the wing in integration stations

        Assumptions:

        Source:
            The Flight Optimization System Weight Estimation Method

        Inputs:
            NSD: number of integration stations requested
            ETA: list of spanwise locations of all sections of the wing

       Outputs:
           NS: actual number of integration stations
           Y: spanwise locations of the integrations stations normalized by half span

        Properties Used:
            N/A
    """
    Y           = [ETA[1]]
    desired_int = (ETA[-1] - ETA[1]) / NSD
    NS          = 0
    for i in range(2, len(ETA)):
        NP = int((ETA[i] - ETA[i - 1]) / desired_int + 0.5)
        if NP < 1:
            NP = 1
        AINT = (ETA[i] - ETA[i - 1]) / NP
        for j in range(NP):
            NS = NS + 1
            Y.append(Y[-1] + AINT)
    return NS, Y

## @ingroup Methods-Weights-Correlations-FLOPS
def calculate_load(ETA):
    """ Returns load factor assuming elliptical load distribution

        Assumptions:

        Source:
            The Flight Optimization System Weight Estimation Method

        Inputs:
            ETA: list of spanwise locations of all sections of the wing

       Outputs:
           PS: load factor at every location in ETA assuming elliptical load distribution

        Properties Used:
            N/A
    """
    PS = np.sqrt(1. - ETA ** 2)
    return PS

## @ingroup Methods-Weights-Correlations-FLOPS
def find_sweep(y, lst_y, swp):
    """ Finds sweep angle for a certain y-location along the wing

        Assumptions:

        Source:
            The Flight Optimization System Weight Estimation Method

        Inputs:
            y: spanwise location
            lst_y: list of spanwise stations where sweep is known (eg sections)
            swp: list of quarter chord sweep angles at the locations listed in lst_y

       Outputs:
           swps: sweep angle at y

        Properties Used:
            N/A
    """
    
    # All initial sweeps are the root chord sweep
    swps = np.ones_like(y)*swp[0]
    
    for i in range(len(lst_y)-1):
        e       = lst_y[i]
        swps[y>=e] = swp[i]
        

    return swps

## @ingroup Methods-Weights-Correlations-FLOPS
def get_spanwise_engine(networks, SEMISPAN):
    """ Returns EETA for the engine locations along the wing

        Assumptions:

        Source:
            The Flight Optimization System Weight Estimation Method

        Inputs:
            networks: data dictionary with all the engine properties
                -.wing_mounted: list of boolean if engine is mounted to wing
                -.number_of_engines: number of engines
                -.origin: origin of the engine
            SEMISPAN: half span                                 [m]
       Outputs:
           EETA: span wise locations of the engines mounted to the wing normalized by the half span

        Properties Used:
            N/A
    """
    N2      = int(sum(networks.wing_mounted) / 2)
    EETA    = np.zeros(N2)
    idx     = 0
    for i in range(int(networks.number_of_engines)):
        if networks.wing_mounted[i] and networks.origin[i][1] > 0:
            EETA[idx] = (networks.origin[i][1] / Units.ft) * 1 / SEMISPAN
            idx += 1
    return EETA

## @ingroup Methods-Weights-Correlations-FLOPS
def wing_weight_constants_FLOPS(ac_type):
    """Defines wing weight constants as defined by FLOPS
        Inputs: ac_type - determines type of instruments, electronics, and operating items based on types:
                "short-range", "medium-range", "long-range", "business", "cargo", "commuter", "sst"
        Outputs: list of coefficients used in weight estimations

    """
    if ac_type == "short_range" or ac_type == "business" or \
            ac_type == "commuter":
        A = [30.0, 0., 0.25, 0.5, 0.5, 0.16, 1.2]
    else:
        A = [8.8, 6.25, 0.68, 0.34, 0.6, 0.035, 1.5]
    return A

## @ingroup Methods-Weights-Correlations-FLOPS
def determine_fuselage_chord(fuselage_width, wing):
    """ Determine chord at wing and fuselage intersection

        Assumptions:
            Fuselage side of body is between first and second wing segments.

        Source:
            The Flight Optimization System Weight Estimation Method

        Inputs:
            fuselage_width: width of fuselage                                   [m]
            wing: data dictionary with wing properties
                    -.taper: taper ratio
                    -.sweeps.quarter_chord: quarter chord sweep angle           [deg]
                    -.thickness_to_chord: thickness to chord
                    -.spans.projected: wing span                                [m]
                    -.chords.root: root chord                                   [m]
                -.fuselages.fuselage.width: fuselage width                      [m]
       Outputs:
           chord: chord length of wing where wing intersects the fuselage wall [ft]

        Properties Used:
            N/A
    """
    root_chord      = wing.chords.root / Units.ft
    SPAN            = wing.spans.projected / Units.ft  # Wing span, ft
    SEMISPAN        = SPAN / 2
    c1              = root_chord * wing.Segments[0].root_chord_percent
    c2              = root_chord * wing.Segments[1].root_chord_percent
    y1              = wing.Segments[0].percent_span_location
    y2              = wing.Segments[1].percent_span_location
    b               = (y2 - y1) * SEMISPAN
    taper           = c2 / c1
    y               = fuselage_width / 2 * 1 / Units.ft
    chord           = c1 * (1 - (1 - taper) * 2 * y / b)
    return chord
