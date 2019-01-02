## @ingroup Methods-Geometry-Three_Dimensional
# compute_equivalent_properties.py
# 
# Created:  Jan 2019, S. Karpuk, 
# Modified: 


import SUAVE
import math
# ----------------------------------------------------------------------
#  Compute Equivalent Wing Properties for Unconvenrional Wings
# ---------------------------------------------------------------------- 

## @ingroup Methods-Geometry-Three_Dimensional
def compute_equivalent_properties(wing):
    """Compute equivalent wing properties for unconventional wings

    Source:
    None

    Inputs:
    wing.area                    [m**2]
    wing.chords.locations        [m]
    wing.chords.sections         [m]
    wing.sweeps.leading_edge     [deg]
    wing.spans.projected         [m]

    Outputs:
    wing.taper                  [Unitless]
    wing.sweeps.quarter_chord   [deg]
    wing.chords.
            root                [m]
            tip                 [m]

    Properties Used:
    N/A
    """      

    #---------
    # Unpack
    #---------
    area      = wing.area
    span      = wing.spans.projected
    Y         = wing.chords.locations
    chords    = wing.chords.sections
    sweep_LE  = wing.sweeps.leading_edge

    sumr     = 0
    sumt     = 0
    sumsweep = 0

    for i in range(len(chords)-1):
        sumr     = sumr + chords[i] * 0.5 * (chords[i] + chords[i+1]) * (Y[i+1] - Y[i])
        sumt     = sumt + chords[i+1] * 0.5 * (chords[i] + chords[i+1]) * (Y[i+1] - Y[i])
        sumsweep = sumsweep + sweep_LE[i] * 0.5 * (chords[i] + chords[i+1]) * (Y[i+1] - Y[i])

    Cwr     = 2 * sumr / area
    Cwt     = 2 * sumt / area
    sweepLE = 2 * sumsweep / area 
    K       = area / (span * (Cwr + Cwt))

    Cre   = K * Cwr
    Cte   = K * Cwt

    taper = Cte / Cre
    sweepC4 = math.degrees(math.atan(math.radians(sweepLE) + Cre / (2 * span) * (taper - 1)))
    
    #---------
    # Pack
    #---------   
    wing.chords.root          = Cre
    wing.chords.tip           = Cte
    wing.taper                = taper
    wing.sweeps.quarter_chord = sweepC4
    wing.sweeps.leading_edge  = sweepLE

    
    return wing
