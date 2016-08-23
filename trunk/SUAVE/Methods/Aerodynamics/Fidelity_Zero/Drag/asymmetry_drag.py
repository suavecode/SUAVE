# asymmetry_drag.py
# 
# Created:  Oct 2015, T. Orra
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE Imports
import SUAVE
from SUAVE.Components import Wings
from SUAVE.Core import Units, Data
from SUAVE.Analyses import Results

# ----------------------------------------------------------------------
#  Compute asymmetry drag due to engine failure 
# ----------------------------------------------------------------------

def asymmetry_drag(state, geometry, windmilling_drag_coefficient = 0.):
    """ SUAVE.Methods.Aerodynamics.Fidelity_Zero.Drag.asymmetry_drag(state, geometry, windmilling_drag_coefficient = 0.):
        Compute asymmetry drag due to engine failure 

        Inputs:
            geometry   - data dictionary with data of vehicle and engine
        
            state      -  data dictionary with state for thrust calculation:
                            state.conditions.freestream.dynamic_pressure
                            state.conditions.freestream.gravity    
                            state.conditions.freestream.velocity   
                            state.conditions.freestream.mach_number
                            state.conditions.freestream.temperature
                            state.conditions.freestream.pressure   
                            state.conditions.propulsion.throttle                   
            
            windmilling_drag_coefficient [optional] - user input to be used in 
                                                      calculation. Estimated if not specified.        
        
        Outputs:
            asymmetry_drag

        Assumptions:
            Two engine airplane
"""
    # ==============================================
	# Unpack
    # ==============================================
    vehicle    = geometry
    propulsors = vehicle.propulsors
    wings      = vehicle.wings
    dyn_press  = state.conditions.freestream.dynamic_pressure
    
     # Defining reference area
    if vehicle.reference_area:
            reference_area = vehicle.reference_area
    else:
        n_wing = 0
        for wing in wings:
            if not isinstance(wing,Wings.Main_Wing): continue
            n_wing = n_wing + 1
            reference_area = wing.sref
        if n_wing > 1:
            print ' More than one Main_Wing in the vehicle. Last one will be considered.'
        elif n_wing == 0:
            print  'No Main_Wing defined! Using the 1st wing found'
            for wing in wings:
                if not isinstance(wing,Wings.Wing): continue
                reference_area = wing.sref
                break
            
    # getting cg x position
    xcg = vehicle.mass_properties.center_of_gravity[0] 
    
    # getting engine y position and calculating thrust
    for idx,propulsor in enumerate(propulsors):
        y_engine = propulsor.position[1]             
        # Getting engine thrust
        results = propulsor(state) # total thrust
        thrust  = results.thrust_force_vector[0,0] / propulsor.number_of_engines
        break
    
    # finding vertical tail
    for idx,wing in enumerate(wings):
        if not wing.vertical: continue
        vertical_idx = wing.tag
        break
    # if vertical tail not found, raise error
    try:
        vertical_idx
    except AttributeError:
        print ' No vertical tail found! Error calculating one engine inoperative drag'

    # getting vertical tail data (span, distance to cg)
    vertical_height = wings[vertical_idx].spans.projected
    vertical_dist   = wings[vertical_idx].aerodynamic_center[0] + wings[vertical_idx].origin[0] - xcg
    
    # colculating windmilling drag
    if windmilling_drag_coefficient == 0:
        try:
            windmilling_drag_coefficient = state.conditions.aerodynamics.drag_breakdown.windmilling_drag.windmilling_drag_coefficient
        except: pass
    
    windmilling_drag = windmilling_drag_coefficient * dyn_press * reference_area
    
    # calculating Drag force due to trim     
    trim_drag = (y_engine**2 * (thrust+windmilling_drag)**2 ) /      \
                (dyn_press * 3.141593* (vertical_height*vertical_dist)**2)

    # Compute asymmetry trim drag coefficient
    asymm_trim_drag_coefficient = trim_drag / dyn_press / reference_area

    # dump data to state
    state.conditions.aerodynamics.drag_breakdown.asymmetry_trim_coefficient = asymm_trim_drag_coefficient

    return asymm_trim_drag_coefficient