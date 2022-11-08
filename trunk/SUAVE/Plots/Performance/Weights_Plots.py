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
    top_run_sum       = 0
    for key, value in vehicle.items():
        if key == 'mass_properties':
            continue
        if isinstance(value,Physical_Component.Container):
            mass                   = value.sum_mass()
            top_level_weights[key] = mass
            top_run_sum           += mass
            top_level_percent[key] = mass/results.total
            
    top_level_weights['Additional'] = results.total - top_run_sum
    
    individual_weights  = Data()
    individual_percents = Data()
    

    
    _, individual_weights = recursive_masses(vehicle,individual_weights)
    
    
    print(individual_weights)
    
    print(top_level_weights)


    
# This is not the same as data.do_recursive
def recursive_masses(data_structure,res):
    
    for k,val in data_structure.items():
        if k=='_diff': # If it's part of a config, skip
            continue
        if isinstance(val,Physical_Component): # Keep this mass value
            d = Data({k:val.mass_properties.mass})
            d.tag = k
            res.append(d)
        elif isinstance(val,Data):             # Recurse
            data_structure, res = recursive_masses(val,res)

    return data_structure, res