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
from Boeing_737 import vehicle_setup

def main():
    
    
    # vehicle data
    vehicle_original = vehicle_setup()    
    vehicle_PGM      = deepcopy(vehicle_original)
    
    # Pull out the non dimensional origins
    vehicle_PGM = set_origin_non_dimensional(vehicle_PGM)
    
    # Run the sizing function
    vehicle_PGM = size_from_PGM(vehicle_PGM)
    
    results = diff(vehicle_original, vehicle_PGM)
    
    # load older results
    #save_results(results)
    old_results = load_results('diff.res')   
    
    # Need to save new results or formatting doesn't match
    save_results_new(results)
    new_results = load_results('diff_new.res')
    
    check = diff(old_results,new_results)
    
    if len(check)>0:
        assert 'A difference in results were found'
    else:
        print('Passed')

     
    return


def check_results(new_results,old_results):

    return

def load_results(tag):
    return SUAVE.Input_Output.SUAVE.load(tag)

def save_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'diff.res')
    
def save_results_new(results):
    SUAVE.Input_Output.SUAVE.archive(results,'diff_new.res')    

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()