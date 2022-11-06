# weights.py
# 
# Created: Nov 2022, E. Botero
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import numpy as np
from SUAVE.Core import Data
from SUAVE.Components import Physical_Component


# ----------------------------------------------------------------------        
#   Weight Pie Charts
# ----------------------------------------------------------------------   
## @ingroup Plots-Performance
def weight_pie_charts(vehicle,analyses):
    """This plots pie charts of the weights of the aircraft

    Assumptions:
    None

    Source:
    None

    Inputs:
    airfoil_geometry_files   <list of strings>

    Outputs: 
    Plots

    Properties Used:
    N/A	
    """    

    # Run the weights analyses to update
    results = analyses.weights.evaluate()
    
    # split into two categories: individual component level and type level (fuselage/wing)
    top_level_weights = Data()
    top_level_percent = Data()
    run_sum           = 0
    for key, value in vehicle.items():
        if key == 'mass_properties':
            continue
        if isinstance(value,Physical_Component.Container):
            mass                   = value.sum_mass()
            top_level_weights[key] = mass
            run_sum               += mass
            top_level_percent[key] = mass/results.total
            
    top_level_weights['Additional'] = results.total - run_sum
    
    print(top_level_weights)
    
    

    