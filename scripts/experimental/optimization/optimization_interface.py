
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt

import copy, time

from SUAVE.Core import (
Data, Container, Data_Exception, Data_Warning,
)

import SUAVE.Plugins.VyPy.optimize as vypy_opt


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    
    # setup the interface
    interface = setup_interface()
    
    # quick test
    inputs = Data()
    inputs.projected_span  = 36.
    inputs.fuselage_length = 58.
    
    # evalute!
    results = interface.evaluate(inputs)
    
    """
    VEHICLE EVALUATION 1
    
    INPUTS
    <data object 'SUAVE.Core.Data'>
    projected_span : 36.0
    fuselage_length : 58.0
    
    RESULTS
    <data object 'SUAVE.Core.Data'>
    fuel_burn : 15700.3830236
    weight_empty : 62746.4
    """
    
    return


# ----------------------------------------------------------------------
#   Optimization Interface Setup
# ----------------------------------------------------------------------

def setup_interface():
    
    # ------------------------------------------------------------------
    #   Instantiate Interface
    # ------------------------------------------------------------------
    
    interface = SUAVE.Optimization.Interface()
    
    # ------------------------------------------------------------------
    #   Vehicle and Analyses Information
    # ------------------------------------------------------------------
    
    from full_setup import full_setup
    
    configs,analyses = full_setup()
    
    interface.configs  = configs
    interface.analyses = analyses
    
    
    # ------------------------------------------------------------------
    #   Analysis Process
    # ------------------------------------------------------------------
    
    process = interface.process
    
    # the input unpacker
    process.unpack_inputs = unpack_inputs
    
    # size the base config
    process.simple_sizing = simple_sizing
    
    # finalizes the data dependencies
    process.finalize = finalize
    
    # the missions
    process.missions = missions
    
    # performance studies
    process.takeoff_field_length    = takeoff_field_length
    process.fuel_for_missions       = fuel_for_missions
    process.short_field             = short_field
    process.mission_fuel            = mission_fuel
    process.max_range               = max_range
    process.noise                   = noise
        
    # summarize the results
    process.summary = summarize
    
    # done!
    return interface    
    
# ----------------------------------------------------------------------
#   Unpack Inputs Step
# ----------------------------------------------------------------------
    
def unpack_inputs(interface):
    
    inputs = interface.inputs
    
    print "VEHICLE EVALUATION %i" % interface.evaluation_count
    print ""
    
    print "INPUTS"
    print inputs
    
    # unpack interface
    vehicle = interface.configs.base
    vehicle.pull_base()
    
    # apply the inputs
    vehicle.wings['main_wing'].aspect_ratio         = inputs.aspect_ratio
    vehicle.wings['main_wing'].areas.reference      = inputs.reference_area
    vehicle.wings['main_wing'].sweep                = inputs.sweep * Units.deg
    vehicle.propulsors['turbo_fan'].design_thrust   = inputs.design_thrust
    vehicle.wings['main_wing'].thickness_to_chord   = inputs.wing_thickness
    vehicle.mass_properties.max_takeoff             = inputs.MTOW   
    vehicle.mass_properties.max_zero_fuel           = inputs.MTOW * inputs.MZFW_ratio
    
    vehicle.mass_properties.takeoff                 = inputs.MTOW   

    vehicle.store_diff()
     
    return None

# ----------------------------------------------------------------------
#   Apply Simple Sizing Principles
# ----------------------------------------------------------------------

def simple_sizing(interface):
      
    from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform

    # unpack
    configs  = interface.configs
    analyses = interface.analyses        
    base = configs.base
    base.pull_base()
    
    # wing areas
    for wing in base.wings:
        wing.areas.wetted   = 2.00 * wing.areas.reference
        wing.areas.affected = 0.60 * wing.areas.reference
        wing.areas.exposed  = 0.75 * wing.areas.wetted

    # wing simple sizing function
    # size main wing
    base.wings['main_wing'] = wing_planform(base.wings['main_wing'])
    # reference values for empennage scaling
    Sref = base.wings['main_wing'].areas.reference
    span = base.wings['main_wing'].spans.projected
    CMA  = base.wings['main_wing'].chords.mean_aerodynamic    
    Sh = 32.48 * (Sref * CMA)  / (124.86 * 4.1535)
    Sv = 26.40 * (Sref * span) / (124.86 * 35.35)
        
    # empennage scaling
    base.wings['horizontal_stabilizer'].areas.reference = Sh
    base.wings['vertical_stabilizer'].areas.reference = Sv
    # sizing of new empennages
    base.wings['horizontal_stabilizer'] = wing_planform(base.wings['horizontal_stabilizer']) 
    base.wings['vertical_stabilizer']   = wing_planform(base.wings['vertical_stabilizer'])   
         
    # fuselage seats
    base.fuselages['fuselage'].number_coach_seats = base.passengers        
    
    # Weight estimation
    breakdown = analyses.configs.base.weights.evaluate()
    
    # pack
    base.mass_properties.breakdown = breakdown
    base.mass_properties.operating_empty = breakdown.empty
##    base.mass_properties.takeoff         = base.mass_properties.max_takeoff
       
    # diff the new data
    base.store_diff()
    
    # Update all configs with new base data    
    for config in configs:
        config.pull_base()
    
    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------
    landing = configs.landing
    
    # make sure base data is current
    landing.pull_base()
    
    # landing weight
    landing.mass_properties.landing = 0.85 * base.mass_properties.takeoff
    
    # diff the new data
    landing.store_diff()    
        
    # done!
    return
# ----------------------------------------------------------------------
#   Finalizing Function (make part of optimization interface)[needs to come after simple sizing doh]
# ----------------------------------------------------------------------    

def finalize(interface):
    
    interface.configs.finalize()
    interface.analyses.finalize()
    
    return None

# ----------------------------------------------------------------------
#   Process Missions
# ----------------------------------------------------------------------    

def missions(interface):
    
    missions = interface.analyses.missions
    
    results = missions.evaluate()
    
    return results            
    
# ----------------------------------------------------------------------
#   Field Length Evaluation
# ----------------------------------------------------------------------    
    
def takeoff_field_length(interface):
    
    # import tofl analysis module
    estimate_tofl = SUAVE.Methods.Performance.estimate_take_off_field_length
    
    # unpack data
    analyses        = interface.analyses
    missions        = interface.analyses.missions
    config          = interface.configs.takeoff
    # defining required data for tofl evaluation
    takeoff_airport = missions.base.airport    
    ref_weight      = config.mass_properties.takeoff
    weight_max      = config.mass_properties.max_takeoff
    weight_min      = config.mass_properties.operating_empty    
                
    # evaluate
    try:
        del config.maximum_lift_coefficient
    except: pass

    # weight vector to eval tofl
    weight_vec  = np.linspace(weight_min,weight_max,10)
    takeoff_field_length = np.zeros_like(weight_vec)
    
    # loop of tofl evaluation
    for idw,weight in enumerate(weight_vec):
        config.mass_properties.takeoff = weight
        takeoff_field_length[idw] = estimate_tofl(config,analyses.configs.takeoff, takeoff_airport)
    
    # return initial value for takeoff weight
    config.mass_properties.takeoff = ref_weight
    
    # pack results
    results = Data()
    results.takeoff_field_length = takeoff_field_length
    results.takeoff_weights      = weight_vec      
        
    return results

# ----------------------------------------------------------------------
#   Run mission for fuel consumption
# ----------------------------------------------------------------------
def fuel_for_missions(interface):

    # unpack data
    config   = interface.configs.cruise
    analyses = interface.analyses

    mission         = interface.analyses.missions.fuel.mission
    mission_payload = interface.analyses.missions.fuel.payload
    
    # determine maximum range based in tow short_field
    from SUAVE.Methods.Performance import size_mission_range_given_weights
    
    # unpack
    cruise_segment_tag = 'cruise'
    
    weight_max    = config.mass_properties.max_takeoff
    weight_min    = config.mass_properties.operating_empty + 0.10 * mission_payload  # 10%
    
    takeoff_weight_vec  = np.linspace(weight_min,weight_max,3)
    distance_vec        = np.zeros_like(takeoff_weight_vec)
    fuel_vec            = np.zeros_like(takeoff_weight_vec)
    
    # call function
    distance_vec,fuel_vec = size_mission_range_given_weights(config,mission,cruise_segment_tag,mission_payload,takeoff_weight_vec)

    # pack 
    results = Data()
    results.tag            = 'missions_fuel'
    results.weights        = takeoff_weight_vec
    results.distances      = distance_vec
    results.fuels          = fuel_vec
    
    return results

# ----------------------------------------------------------------------
#   Evaluate Range from short field
# ----------------------------------------------------------------------
def short_field(interface):

    # unpack data
    results_field   = interface.results.takeoff_field_length
    results_fuel    = interface.results.fuel_for_missions
    available_tofl  = interface.analyses.missions.short_field.mission.airport.available_tofl
 
    tofl_vec        = results_field.takeoff_field_length
    weight_vec_tofl = results_field.takeoff_weights
    
    range_vec       = results_fuel.distances
    weight_vec_fuel = results_fuel.weights
    fuel_vec        = results_fuel.fuels
        
    # evaluate maximum allowable takeoff weight from a given airfield
    tow_short_field = np.interp(available_tofl,tofl_vec,weight_vec_tofl)

    # determine maximum range/fuel based in tow short_field
    range_short_field = np.interp(tow_short_field,weight_vec_fuel,range_vec)
    fuel_short_field  = np.interp(tow_short_field,weight_vec_fuel,fuel_vec)

    # pack 
    results = Data()
    results.tag            = 'short_field'
    results.takeoff_weight = tow_short_field
    results.range          = range_short_field
    results.fuel           = fuel_short_field

    return results

# ----------------------------------------------------------------------
#   Evaluate fuel for design mission
# ----------------------------------------------------------------------
def mission_fuel(interface):

    # unpack data
    design_range  = interface.analyses.missions.fuel.range  
    range_vec     = interface.results.fuel_for_missions.distances
    fuel_vec      = interface.results.fuel_for_missions.fuels
        
    # determine maximum range/fuel based in tow short_field
    fuel_design_mission  = np.interp(design_range,range_vec,fuel_vec)

    # pack results
    results = Data()
    results.tag            = 'design_mission'
    results.range          = design_range
    results.fuel           = fuel_design_mission

    return results

# ----------------------------------------------------------------------
#   Evaluate fuel for design mission
# ----------------------------------------------------------------------
def max_range(interface):

    # unpack data
    max_takeoff_weight  = interface.configs.base.mass_properties.max_takeoff
    range_vec           = interface.results.fuel_for_missions.distances
    weight_vec_fuel     = interface.results.fuel_for_missions.weights
    fuel_vec            = interface.results.fuel_for_missions.fuels
        
    # determine maximum range/fuel based in max_tow
    range = np.interp(max_takeoff_weight,weight_vec_fuel,range_vec)
    fuel  = np.interp(max_takeoff_weight,weight_vec_fuel,fuel_vec)

    # pack results
    results = Data()
    results.tag            = 'short_field'
    results.takeoff_weight = max_takeoff_weight
    results.range          = range
    results.fuel           = fuel

    return results

# ----------------------------------------------------------------------
#   Noise Evaluation
# ----------------------------------------------------------------------    

def noise(interface):
    
    # TODO - use the analysis
    
    # unpack noise analysis
    evaluate_noise = SUAVE.Methods.Noise.Correlations.shevell
    
    # unpack data
    vehicle = interface.configs.base
    results = interface.results
    mission_profile = results.missions.base
    
    weight_landing    = mission_profile.segments[-1].conditions.weights.total_mass[-1,0]
    number_of_engines = vehicle.propulsors['turbo_fan'].number_of_engines
    thrust_sea_level  = vehicle.propulsors['turbo_fan'].design_thrust
    thrust_landing    = mission_profile.segments[-1].conditions.frames.body.thrust_force_vector[-1,0]
    
    # evaluate
    results = evaluate_noise( weight_landing    , 
                              number_of_engines , 
                              thrust_sea_level  , 
                              thrust_landing     )
    
    return results    
    
# ----------------------------------------------------------------------
#   Summarize the Data
# ----------------------------------------------------------------------    

def summarize(interface):
   
    vehicle = interface.configs.base    
    results = interface.results
    mission_profile = results.missions.base    
    short_field_results = results.short_field
    
    
    # merge all segment conditions
    def stack_condition(a,b):
        if isinstance(a,np.ndarray):
            return np.vstack([a,b])
        else:
            return None
    
    conditions = None
    for segment in mission_profile.segments:
        if conditions is None:
            conditions = segment.conditions
            continue
        conditions = conditions.do_recursive(stack_condition,segment.conditions)
      
    # pack
    summary = SUAVE.Core.Results()
    
    # Geometry and weights
    summary.weight_empty    = vehicle.mass_properties.operating_empty
    summary.weight_MZFW     = vehicle.mass_properties.max_zero_fuel
    
    summary.ref_area        = vehicle.wings['main_wing'].areas.reference
    summary.aspect_ratio    = vehicle.wings['main_wing'].aspect_ratio
    summary.sweep           = vehicle.wings['main_wing'].sweep / float(1*Units.deg) 
    
    # Fuel burn      
    summary.fuel_burn  = results.mission_fuel.fuel
    summary.fuel_range = float(results.mission_fuel.range/ Units.nmi)

    summary.range_short_field_nmi = float(short_field_results.range / Units.nmi)
    summary.tow_short_field      = short_field_results.takeoff_weight
    summary.takeoff_field_length  = float(results.takeoff_field_length.takeoff_field_length[-1]) 
    
    summary.range_max_nmi         = float(results.max_range.range / Units.nmi)   
    summary.fuel_max              =  results.max_range.fuel 
    
    summary.max_zero_fuel_margin  = summary.weight_MZFW - (summary.weight_empty + vehicle.mass_properties.payload)
    
    from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_fuel_volume
    wing_fuel_volume(vehicle.wings['main_wing'])
    fuel_density = vehicle.propulsors['turbo_fan'].combustor.fuel_data.density
    fuel_available = 0.97 * vehicle.wings['main_wing'].fuel_volume * fuel_density    
    
    summary.available_fuel_margin = fuel_available - summary.fuel_max
    
        
##    summary.fuel_burn = max(conditions.weights.total_mass[:,0]) - min(conditions.weights.total_mass[:,0])    
##    summary.noise = results.noise        
##    summary.mission_time_min = max(conditions.frames.inertial.time[:,0] / Units.min)
##    summary.max_altitude_km = max(conditions.freestream.altitude[:,0] / Units.km)
    
##    summary.range_nmi = mission_profile.segments[-1].conditions.frames.inertial.position_vector[-1,0] / Units.nmi
        
##    summary.stability = Data()
##    summary.stability.cm_alpha = max(conditions.stability.static.cm_alpha[:,0])
##    summary.stability.cn_beta  = max(conditions.stability.static.cn_beta[:,0])
    
##    #summary.conditions = conditions
    
##    #TODO: revisit how this is calculated
##    summary.second_segment_climb_rate = mission_profile.segments[1].climb_rate
    
   
    printme = Data()
##    printme.range        = summary.fuel_range
    printme.fuel_burn_design_mission    = summary.fuel_burn
##    printme.ref_area     = summary.ref_area
##    printme.aspect_ratio = summary.aspect_ratio
##    printme.sweep        = summary.sweep
    printme.weight_empty = summary.weight_empty
    printme.tofl_MTOW    = summary.takeoff_field_length
##    printme.SF_tofl      = summary.tow_short_field
    printme.SF_range     = summary.range_short_field_nmi
    printme.range_max    = summary.range_max_nmi
    
    printme.max_zero_fuel_margin      = summary.max_zero_fuel_margin
    printme.available_fuel_margin     = summary.available_fuel_margin
    
    print "RESULTS"
    print printme
    
##    from SUAVE._Apagar import print_compress_drag,print_engine_data,print_mission_breakdown,print_parasite_drag   
##    # functions to printout some results
##    # parasite drag    
##    ref_conditions = Data()
##    ref_conditions.mach_number = 0.40
##    ref_conditions.reynolds_number = 12*10**6
##    
##    vehicle.aerodynamics_model = Data()
##    vehicle.aerodynamics_model.configuratio = Data()
##    vehicle.aerodynamics_model.configuration = interface.analyses.configs.base.aerodynamics.settings
##    vehicle.propulsion_model = interface.analyses.configs.base.aerodynamics.settings
##    
##    vehicle.configs = Data()
##    vehicle.configs.cruise = vehicle  
##    
##    print_parasite_drag(ref_conditions,vehicle)
##    # compressibility drag
##    print_compress_drag(vehicle)
##    # mission breakdown    
####    print_mission_breakdown(mission_profile)   
##    # engine data    
##    print_engine_data(vehicle,mission_profile)
    
    return summary




if __name__ == '__main__':
    main()
    
    
