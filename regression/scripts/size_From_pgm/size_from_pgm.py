# size_from_pgm.py
# 
# Created:  May 2020, E. Botero
# Modified:

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core.Diffed_Data import diff
from SUAVE.Sizing.size_from_PGM import size_from_PGM
from SUAVE.Methods.Geometry.Two_Dimensional.Planform.rescale_non_dimensional import set_origin_non_dimensional

from copy import deepcopy

import sys
sys.path.append('../Vehicles')
from Boeing_737 import vehicle_setup as B737_setup
from Concorde import vehicle_setup as concorde_setup
from Boeing_BWB_450 import vehicle_setup as BWB_setup
from Yak54_wing_only import vehicle_setup as Yak_setup

def main():
    
    
    # vehicle data
    b737 = B737_setup()    
    check_a_vehicle(b737)
    
    # Now the Concorde
    concorde = concorde_setup()
    check_a_vehicle(concorde)
    
    # Now the BwB
    BWB = BWB_setup()
    check_a_vehicle(BWB)
    
    # Now the Yak, and make the wing really thick
    Yak = Yak_setup()
    Yak.wings.main_wing.thickness_to_chord = 1.
    Yak.wings.main_wing.taper              = 1.
    Yak.wings.main_wing.chords.root        = 24
    
    check_a_vehicle(Yak)    
    
    # Mess up one wing segment to test if it fixes it
    b737.tag = 'Messed_Up_wing'
    b737.wings.main_wing.Segments[-1].percent_span_location = .99
    b737.wings.main_wing.Segments[-1].root_chord_percent    = -.0001
    b737.wings.main_wing.Segments[-1].thickness_to_chord    = 0.
    check_a_vehicle(b737)

     
    return


def check_a_vehicle(vehicle_original):
    
    vehicle_PGM      = deepcopy(vehicle_original)
    tag = vehicle_original.tag
    
    # Pull out the non dimensional origins
    vehicle_PGM = set_origin_non_dimensional(vehicle_PGM)
    
    # Run the sizing function
    vehicle_PGM = size_from_PGM(vehicle_PGM)
    
    results = diff(vehicle_original, vehicle_PGM)
    
    # load older results, this should be commented out
    #save_results(results,tag+'diff.res')
    old_results = load_results(tag+'diff.res')   
    
    # Need to save new results or formatting doesn't match, this should not be commented out
    save_results(results,tag+'diff_new.res')
    new_results = load_results(tag+'diff_new.res')
    
    check = diff(old_results,new_results)
    
    assert(len(check)==0),'A difference in results was found for the ' + tag  
    
    return

def load_results(tag):
    return SUAVE.Input_Output.SUAVE.load(tag)

def save_results(results,tag):
    SUAVE.Input_Output.SUAVE.archive(results,tag) 

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()