import SUAVE
from SUAVE.Methods.Aerodynamics import Fidelity_Zero as fz
from SUAVE.Methods.Aerodynamics import Supersonic_Zero as sz
import copy

def aero_compare(state,settings,geometry):

    # Fidelity Zero Values ------------

    base_state = copy.deepcopy(state)
    base_settings = copy.deepcopy(settings)
    base_geometry = copy.deepcopy(geometry)

    # Lift

    fz.Lift.wing_compressibility_correction(state, settings, geometry)
    fz.Lift.fuselage_correction(state, settings, geometry)
    fz.Lift.aircraft_total(state, settings, geometry)
    
    fz_cl = state.conditions.aerodynamics.lift_coefficient
    
    # Drag
    
    fz.Drag.parasite_drag_wing(state,settings,geometry.wings.main_wing)
    fz.Drag.parasite_drag_wing(state,settings,geometry.wings.horizontal_stabilizer)
    fz.Drag.parasite_drag_wing(state,settings,geometry.wings.vertical_stabilizer)
    fz.Drag.parasite_drag_fuselage(state, settings, geometry.fuselages.fuselage)
    fz.Drag.parasite_drag_propulsor(state, settings, geometry.propulsors.turbofan)
    #fz.Drag.parasite_drag_pylon(state, settings, geometry)
    fz.Drag.parasite_total(state, settings, geometry)
    fz.Drag.induced_drag_aircraft(state, settings, geometry)
    fz.Drag.compressibility_drag_wing(state, settings, geometry.wings.main_wing)
    fz.Drag.compressibility_drag_wing(state, settings, geometry.wings.horizontal_stabilizer)
    fz.Drag.compressibility_drag_wing(state, settings, geometry.wings.vertical_stabilizer)
    fz.Drag.compressibility_drag_wing_total(state, settings, geometry)
    fz.Drag.miscellaneous_drag_aircraft_ESDU(state, settings, geometry)
    fz.Drag.untrimmed(state, settings, geometry)
    fz.Drag.trim(state, settings, geometry)
    fz.Drag.spoiler_drag(state, settings, geometry)
    fz.Drag.total_aircraft(state, settings, geometry)
    
    fz_cd = state.conditions.aerodynamics.drag_coefficient


    fz_state    = copy.deepcopy(state)
    fz_settings = copy.deepcopy(settings)
    fz_geometry = copy.deepcopy(geometry)

    # Supersonic Zero Values ----------
    
    state    = copy.deepcopy(base_state)
    settings = copy.deepcopy(base_settings)
    geometry = copy.deepcopy(base_geometry)    
    
    # Lift
    
    sz.Lift.vortex_lift(state, settings, geometry)
    sz.Lift.wing_compressibility(state, settings, geometry)
    sz.Lift.fuselage_correction(state, settings, geometry)
    sz.Lift.aircraft_total(state, settings, geometry)
    
    sz_cl = state.conditions.aerodynamics.lift_coefficient    
    
    # Drag
    
    geometry.wings.main_wing.total_length              = 1.
    geometry.wings.horizontal_stabilizer.total_length  = 1.
    geometry.wings.vertical_stabilizer.total_length    = 1.
    
    sz.Drag.parasite_drag_wing(state,settings,geometry.wings.main_wing)
    sz.Drag.parasite_drag_wing(state,settings,geometry.wings.horizontal_stabilizer)
    sz.Drag.parasite_drag_wing(state,settings,geometry.wings.vertical_stabilizer)
    sz.Drag.parasite_drag_fuselage(state, settings, geometry.fuselages.fuselage)
    sz.Drag.parasite_drag_propulsor(state, settings, geometry.propulsors.turbofan)
    #sz.Drag.parasite_drag_pylon(state, settings, geometry) # doesn't exist?
    sz.Drag.parasite_total(state, settings, geometry)
    sz.Drag.induced_drag_aircraft(state, settings, geometry)
    # change to compressibility drag setup
    sz.Drag.compressibility_drag_total(state, settings, geometry)
    fz.Drag.miscellaneous_drag_aircraft_ESDU(state, settings, geometry) # difference here
    sz.Drag.untrimmed(state, settings, geometry)
    sz.Drag.trim(state, settings, geometry)
    #fz.Drag.spoiler_drag(state, settings, geometry) # unused in supersonic zero
    sz.Drag.total_aircraft(state, settings, geometry)

    sz_cd = state.conditions.aerodynamics.drag_coefficient   
    
    sz_state    = copy.deepcopy(state)
    sz_settings = copy.deepcopy(settings)
    sz_geometry = copy.deepcopy(geometry)    
    
    f = open('fz_breakdown.res','w')
    print >> f, fz_state.conditions.aerodynamics.drag_breakdown
    f.close
    
    f = open('sz_breakdown.res','w')
    print >> f, sz_state.conditions.aerodynamics.drag_breakdown
    f.close    
    
    pass
    
    
if __name__ == '__main__':
    # these were manually saved from a 737 run
    state = SUAVE.Input_Output.SUAVE.load('737_state.res')
    settings = SUAVE.Input_Output.SUAVE.load('737_settings.res')
    geometry = SUAVE.Input_Output.SUAVE.load('737_geometry.res')
    aero_compare(state,settings,geometry)