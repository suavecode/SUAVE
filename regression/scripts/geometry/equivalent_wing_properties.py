# equivalent_wing_properties.py
# 
# Created:  January 2019, S. Karpuk
# Modified: 

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Data
from SUAVE.Core import Units
from SUAVE.Core import (
    Data, Container,
)

import numpy as np
from SUAVE.Methods.Geometry.Three_Dimensional.compute_equivalent_properties import compute_equivalent_properties

# ---------------------------------------------------------------------- 
#   Main
# ----------------------------------------------------------------------
def main():
    
    # --------------------------------------------------------------------------------------------
    # Testing
    # Wing platform from
    # Appendix D Gudmundsson "General Aviation Aorcraft Design: Applied Methods and Procedures"
    # --------------------------------------------------------------------------------------------

    # Setup
    wing               = Data()
    wing.chords        = Data()
    wing.spans         = Data()
    wing.sweeps        = Data()
    wing.area                   = 44.0
    wing.spans.projected        = 16.0
    wing.chords.locations       = [0.0, 4.0, 6.0, 8.0]
    wing.chords.sections        = [4.0, 3.0, 2.0]
    wing.sweeps.leading_edge    = [15, 30, 45]

    # Actual values
    actual             = Data()
    actual.wing        = Data()
    actual.wing.chords = Data()
    actual.wing.spans  = Data()
    actual.wing.sweeps = Data()
    actual.wing.area                = 44.0
    actual.wing.spans.projected     = 22.0
    actual.wing.AR                  = 7.934
    actual.wing.chords.root         = 3.249
    actual.wing.chords.tip          = 2.297
    actual.wing.taper               = actual.wing.chords.tip / actual.wing.chords.root
    actual.wing.sweeps.leading_edge = 15.36107 
    
    # Compute
    wing = compute_equivalent_properties(wing) 
    print(wing)
    # Compute Errors
    error             = Data() 
    error.area        = np.abs(wing.area-actual.wing.area)/actual.wing.area
    error.root        = np.abs(wing.chords.root-actual.wing.chords.root)/actual.wing.chords.root
    error.tip         = np.abs(wing.chords.tip-actual.wing.chords.tip)/actual.wing.chords.tip
    error.taper       = np.abs(wing.taper-actual.wing.taper)/actual.wing.taper
    error.sweep       = np.abs(wing.sweeps.leading_edge-actual.wing.sweeps.leading_edge)/actual.wing.sweeps.leading_edge

    print('Results')
    print(wing)
    print('error')
    print(error)
    
    for k,v in error.items():
        assert(np.abs(v)<1E-3)   
    return
    
if __name__ == '__main__':
    
    main()
    
    print('Fuselage planform regression test passed!')   
