# fuselage_planform_compute.py
# 
# Created:  Apr 2018, W. Maier
# Modified: Apr 2020, E. Botero

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Data

import numpy as np

from SUAVE.Methods.Geometry.Two_Dimensional.Planform.fuselage_planform import fuselage_planform

# ---------------------------------------------------------------------- 
#   Main
# ----------------------------------------------------------------------
def main():
    
    # ------------------------------------------------------------------
    # Testing
    # B737-800
    # ------------------------------------------------------------------

    # Setup
    fuselage          = SUAVE.Components.Fuselages.Fuselage()
    
    fuselage.number_coach_seats = 170.
    fuselage.seat_pitch         = 1.0 
    fuselage.seats_abreast      = 6.0
    fuselage.fineness.nose      = 1.6
    fuselage.fineness.tail      = 2.0
    fuselage.lengths.fore_space = 6.0
    fuselage.lengths.aft_space  = 5.0
    fuselage.width              = 3.74
    fuselage.heights.maximum    = 3.74

    # Compute
    fuselage_planform(fuselage) 
    
    # Truth Values
    nose_length_truth   = 5.984
    tail_length_truth   = 7.48
    cabin_length_truth  = 39.3333
    total_length_truth  = 52.79733
    wetted_area_truth   = 580.79624
    frontal_area_truth  = 10.98583535
    dia_effective_truth = 3.74

    
    # Compute Errors
    error             = Data() 
    error.nose        = np.abs(fuselage.lengths.nose-nose_length_truth)/nose_length_truth
    error.tail        = np.abs(fuselage.lengths.tail-tail_length_truth)/tail_length_truth
    error.cabin       = np.abs(fuselage.lengths.cabin-cabin_length_truth)/cabin_length_truth
    error.total       = np.abs(fuselage.lengths.total-total_length_truth)/total_length_truth
    error.wetted_area = np.abs(fuselage.areas.wetted-wetted_area_truth)/wetted_area_truth
    error.front_area  = np.abs(fuselage.areas.front_projected-frontal_area_truth)/frontal_area_truth
    error.diameter    = np.abs(fuselage.effective_diameter-dia_effective_truth)/dia_effective_truth
            
    for k,v in list(error.items()):
        assert np.any(np.abs(v)<1e-6)
    
if __name__ == '__main__':
    
    main()
    
    print('Fuselage planform regression test passed!')   