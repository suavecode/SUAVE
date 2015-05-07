# Procedure.py
# 
# Created:  May 2015, E. Botero
# Modified: 



# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import SUAVE
from SUAVE.Core import Units, Data

import Vehicles
import Missions
import Analyses

# TODO
Procedure = Data()


# ----------------------------------------------------------------------        
#   Setup
# ----------------------------------------------------------------------   

def setup():
    
    # ------------------------------------------------------------------
    #   Instantiate Interface
    # ------------------------------------------------------------------
    
    procedure = Procedure()
    
    interface = Data()
    
    
    
    # ------------------------------------------------------------------
    #   Analysis Procedure
    # ------------------------------------------------------------------
    
    # the input unpacker
    #procedure.unpack_inputs = unpack_inputs
    
    # size the base config
    procedure.simple_sizing = simple_sizing
    
    # finalizes the data dependencies
    procedure.finalize = finalize
    
    # the missions
    procedure.missions = missions
    
    # performance studies
    procedure.takeoff_field_length    = takeoff_field_length  # generates surrogate
    procedure.fuel_for_missions       = fuel_for_missions     # generates surrogate
    procedure.short_field             = short_field         
    procedure.mission_fuel            = mission_fuel
    procedure.max_range               = max_range
    procedure.noise                   = noise
        
    # summarize the results
    procedure.summary = summarize
    
    # done!
    return procedure
    

# ----------------------------------------------------------------------        
#   Sizing
# ----------------------------------------------------------------------    

def simple_sizing(configs):
    
    base = configs.base
    base.pull_base()
    
    # zero fuel weight
    base.mass_properties.max_zero_fuel = 0.9 * base.mass_properties.max_takeoff 
    
    # wing areas
    for wing in base.wings:
        wing.areas.wetted   = 2.0 * wing.areas.reference
        wing.areas.exposed  = 0.8 * wing.areas.wetted
        wing.areas.affected = 0.6 * wing.areas.wetted
    
    # fuselage seats
    base.fuselages['fuselage'].number_coach_seats = base.passengers
    
    # diff the new data
    base.store_diff()
    
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
        takeoff_field_length[idw] = estimate_tofl(config,analyses, takeoff_airport)
    
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
    
##    print results
    
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
#   Post Process Results to give back to the optimizer
# ----------------------------------------------------------------------   

def post_process(interface):
    
    # Unpack data
    vehicle               = interface.configs.base    
    results               = interface.results
    
    # Weights
    max_zero_fuel     = vehicle.mass_properties.max_zero_fuel
    operating_empty   = vehicle.mass_properties.operating_empty
    payload           = vehicle.mass_properties.payload   
    
    # MZFW margin calculation
    results.max_zero_fuel_margin  = max_zero_fuel - (operating_empty + payload)
    
    # fuel margin calculation
    from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_fuel_volume
    wing_fuel_volume(vehicle.wings['main_wing'])
    fuel_density = vehicle.propulsors['turbo_fan'].combustor.fuel_data.density
    fuel_available = 0.97 * vehicle.wings['main_wing'].fuel_volume * fuel_density    
    results.available_fuel_margin = fuel_available - range_results.fuel
    
    return interface