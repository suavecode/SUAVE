## @ingroup Methods-Weights-Buildups-eVTOL
# converge_evtol_weight.py

# Created: Aug 2022, M. Clarke

#-------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------- 
from SUAVE.Methods.Weights.Buildups.eVTOL.empty import empty

#-------------------------------------------------------------------------------
# Empty
#-------------------------------------------------------------------------------

## @ingroup Methods-Weights-Buildups-eVTOL 
def converge_evtol_weight(vehicle,print_iterations = False):
    '''Converges the maximum takeoff weight of an aircraft using the eVTOL 
    weight buildup routine.  
    
    Source:
    None
    
    Assumptions:
    None
    
    Inputs:
    vehicle           <data structure>
    print_iterations  <boolean>
    
    Outputs:
    None
    
    Properties Used:
    N/A
    '''
    breakdown      = empty(vehicle)
    vehicle_mtow   = vehicle.mass_properties.max_takeoff  
    build_up_mass  = breakdown.total    
    
    diff           = vehicle_mtow - build_up_mass 
    iterations     = 0 
    while(diff>1):
        vehicle.mass_properties.max_takeoff = vehicle_mtow - diff*1E-1
        breakdown      = empty(vehicle) 
        build_up_mass  = breakdown.total    
        diff           = vehicle_mtow - build_up_mass 
        vehicle_mtow   = vehicle.mass_properties.max_takeoff
        iterations     += 1
        if print_iterations:
            print(round(diff,3))
        if iterations == 100:
            print('Weight convergence failed!')
            return 
    print('Converged MTOW = ' + str(round(vehicle_mtow)) + ' kg')
    
    return