# Procedure.py
# 
# Created:  Mar 2016, M. Vegh
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import SUAVE
from SUAVE.Core import Units, Data
import numpy as np
import copy
from SUAVE.Analyses.Process import Process
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Propulsion.compute_turbofan_geometry import compute_turbofan_geometry
from SUAVE.Methods.Center_of_Gravity.compute_component_centers_of_gravity import compute_component_centers_of_gravity
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift.compute_max_lift_coeff import compute_max_lift_coeff


# ----------------------------------------------------------------------        
#   Setup
# ----------------------------------------------------------------------   

def setup():
    
    # ------------------------------------------------------------------
    #   Analysis Procedure
    # ------------------------------------------------------------------ 
    
    # size the base config
    procedure = Process()
    procedure.simple_sizing = simple_sizing
    
    # find the weights
    procedure.weights = weight
    # finalizes the data dependencies
    procedure.finalize = finalize
    
    # performance studies
    procedure.missions                   = Process()
    procedure.missions.design_mission    = design_mission

    # post process the results
    procedure.post_process = post_process
        
    # done!
    return procedure


# ----------------------------------------------------------------------        
#   Target Range Function
# ----------------------------------------------------------------------    

def find_target_range(nexus,mission):
    
    segments=mission.segments
    cruise_altitude=mission.segments['climb_5'].altitude_end
    climb_1=segments['climb_1']
    climb_2=segments['climb_2']
    climb_3=segments['climb_3']
    climb_4=segments['climb_4']
    climb_5=segments['climb_5']
  
    descent_1=segments['descent_1']
    descent_2=segments['descent_2']
    descent_3=segments['descent_3']

    x_climb_1=climb_1.altitude_end/np.tan(np.arcsin(climb_1.climb_rate/climb_1.air_speed))
    x_climb_2=(climb_2.altitude_end-climb_1.altitude_end)/np.tan(np.arcsin(climb_2.climb_rate/climb_2.air_speed))
    x_climb_3=(climb_3.altitude_end-climb_2.altitude_end)/np.tan(np.arcsin(climb_3.climb_rate/climb_3.air_speed))
    x_climb_4=(climb_4.altitude_end-climb_3.altitude_end)/np.tan(np.arcsin(climb_4.climb_rate/climb_4.air_speed))
    x_climb_5=(climb_5.altitude_end-climb_4.altitude_end)/np.tan(np.arcsin(climb_5.climb_rate/climb_5.air_speed))
    x_descent_1=(climb_5.altitude_end-descent_1.altitude_end)/np.tan(np.arcsin(descent_1.descent_rate/descent_1.air_speed))
    x_descent_2=(descent_1.altitude_end-descent_2.altitude_end)/np.tan(np.arcsin(descent_2.descent_rate/descent_2.air_speed))
    x_descent_3=(descent_2.altitude_end-descent_3.altitude_end)/np.tan(np.arcsin(descent_3.descent_rate/descent_3.air_speed))
    
    cruise_range=mission.design_range-(x_climb_1+x_climb_2+x_climb_3+x_climb_4+x_climb_5+x_descent_1+x_descent_2+x_descent_3)
  
    segments['cruise'].distance=cruise_range
    
    return nexus

# ----------------------------------------------------------------------        
#   Design Mission
# ----------------------------------------------------------------------    
def design_mission(nexus):
    
    mission = nexus.missions.base
    mission.design_range = 1500.*Units.nautical_miles
    find_target_range(nexus,mission)
    results = nexus.results
    results.base = mission.evaluate()
    
    return nexus



# ----------------------------------------------------------------------        
#   Sizing
# ----------------------------------------------------------------------    

def simple_sizing(nexus):
    configs=nexus.vehicle_configurations
    base=configs.base

    #find conditions
    air_speed   = nexus.missions.base.segments['cruise'].air_speed 
    altitude    = nexus.missions.base.segments['climb_5'].altitude_end
    atmosphere  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    
    freestream  = atmosphere.compute_values(altitude)
    freestream0 = atmosphere.compute_values(6000.*Units.ft)  #cabin altitude
    
    
    diff_pressure         = np.max(freestream0.pressure-freestream.pressure,0)
    fuselage              = base.fuselages['fuselage']
    fuselage.differential_pressure = diff_pressure 
    
    #now size engine
    mach_number        = air_speed/freestream.speed_of_sound
    
    #now add to freestream data object
    freestream.velocity    = air_speed
    freestream.mach_number = mach_number
    freestream.gravity     = 9.81
    
    conditions             = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()   #assign conditions in form for network sizing
    conditions.freestream  = freestream
    
    
    
    for config in configs:
        config.wings.horizontal_stabilizer.areas.reference = (26.0/92.0)*config.wings.main_wing.areas.reference
            
        for wing in config.wings:
            
            wing = SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)
            
            wing.areas.exposed  = 0.8 * wing.areas.wetted
            wing.areas.affected = 0.6 * wing.areas.reference
            


        fuselage              = config.fuselages['fuselage']
        fuselage.differential_pressure = diff_pressure 
        
        turbofan_sizing(config.networks['turbofan'], mach_number, altitude)
        for nac in config.nacelles: 
            compute_turbofan_geometry(config.networks['turbofan'],nac)
        # diff the new data
        #config.store_diff()

    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------
    landing = nexus.vehicle_configurations.landing
    state = Data()
    state.conditions = Data()
    state.conditions.freestream = Data()

    # landing weight
    landing.mass_properties.landing = 0.85 * config.mass_properties.takeoff
    
    # Landing CL_max
    altitude                                      = nexus.missions.base.segments[-1].altitude_end
    atmosphere                                    = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data                                     = atmosphere.compute_values(altitude)
    state.conditions.freestream.velocity          = nexus.missions.base.segments['descent_3'].air_speed
    state.conditions.freestream.density           = atmo_data.density
    state.conditions.freestream.dynamic_viscosity = atmo_data.dynamic_viscosity 
    settings                                      = Data()
    settings.maximum_lift_coefficient_factor      = 1.0
    CL_max_landing, CDi                           = compute_max_lift_coeff(state,settings,landing)
    landing.maximum_lift_coefficient              = CL_max_landing
    
    
    #Takeoff CL_max
    takeoff                                       = nexus.vehicle_configurations.takeoff
    altitude                                      = nexus.missions.base.airport.altitude
    atmosphere                                    = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data                                     = atmosphere.compute_values(altitude)
    state.conditions.freestream.velocity          = nexus.missions.base.segments.climb_1.air_speed
    state.conditions.freestream.density           = atmo_data.density
    state.conditions.freestream.dynamic_viscosity = atmo_data.dynamic_viscosity 
    settings.maximum_lift_coefficient_factor      = 1.0    
    max_CL_takeoff,CDi                            = compute_max_lift_coeff(state,settings,takeoff)
    takeoff.maximum_lift_coefficient              = max_CL_takeoff

    #Base config CL_max
    base                                          = nexus.vehicle_configurations.base
    altitude                                      = nexus.missions.base.airport.altitude
    atmosphere                                    = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data                                     = atmosphere.compute_values(altitude)
    state.conditions.freestream.velocity          = nexus.missions.base.segments.climb_1.air_speed
    state.conditions.freestream.density           = atmo_data.density
    state.conditions.freestream.dynamic_viscosity = atmo_data.dynamic_viscosity 
    settings.maximum_lift_coefficient_factor      = 1.0       
    max_CL_base,CDi                               = compute_max_lift_coeff(state,settings,landing)
    base.maximum_lift_coefficient                 = max_CL_base    
    
    return nexus

# ----------------------------------------------------------------------        
#   Weights
# ----------------------------------------------------------------------    

def weight(nexus):
    vehicle=nexus.vehicle_configurations.base

    # weight analysis
    weights = nexus.analyses.base.weights.evaluate()
   
    nose_load_fraction=.06
    compute_component_centers_of_gravity(vehicle, nose_load_fraction)
    vehicle.center_of_gravity()
   
    
    weights = nexus.analyses.cruise.weights.evaluate()
    weights = nexus.analyses.landing.weights.evaluate()
    weights = nexus.analyses.takeoff.weights.evaluate()
    weights = nexus.analyses.short_field_takeoff.weights.evaluate()
    
    empty_weight    =vehicle.mass_properties.operating_empty
    for config in nexus.vehicle_configurations:
        config.mass_properties.zero_fuel_center_of_gravity  = vehicle.mass_properties.zero_fuel_center_of_gravity
        config.fuel                                         = vehicle.fuel
    return nexus


# ----------------------------------------------------------------------
#   Finalizing Function (make part of optimization nexus)[needs to come after simple sizing doh]
# ----------------------------------------------------------------------    

def finalize(nexus):
    
    nexus.analyses.finalize()   
    
    return nexus         


    
# ----------------------------------------------------------------------
#   Post Process Results to give back to the optimizer
# ----------------------------------------------------------------------   

def post_process(nexus):
    
    # Unpack data
    vehicle               = nexus.vehicle_configurations.base  
    results               = nexus.results
    summary = nexus.summary
    missions              = nexus.missions  
    
    # Static stability calculations
    CMA = -10.
    for segment in list(results.base.segments.values()):
        max_CMA=np.max(segment.conditions.stability.static.Cm_alpha[:,0])
        if max_CMA>CMA:
            CMA=max_CMA
            

            
    summary.static_stability = CMA
    
    #throttle in design mission
    max_throttle=0
    for segment in list(results.base.segments.values()):
        max_segment_throttle = np.max(segment.conditions.propulsion.throttle[:,0])
        if max_segment_throttle > max_throttle:
            max_throttle = max_segment_throttle

            
    summary.max_throttle = max_throttle
    
    # Fuel margin and base fuel calculations

    operating_empty          = vehicle.mass_properties.operating_empty
    payload                  = vehicle.payload.passengers.mass_properties.mass 
    design_landing_weight    = results.base.segments[-1].conditions.weights.total_mass[-1]
    design_takeoff_weight    = vehicle.mass_properties.takeoff
    max_takeoff_weight       = nexus.vehicle_configurations.takeoff.mass_properties.max_takeoff
    zero_fuel_weight         = payload+operating_empty
    
    summary.max_zero_fuel_margin    = (design_landing_weight - zero_fuel_weight)/zero_fuel_weight
    summary.base_mission_fuelburn   = design_takeoff_weight - results.base.segments['descent_3'].conditions.weights.total_mass[-1]
 
  

    hf = vehicle.fuselages.fuselage.heights.at_wing_root_quarter_chord
    wf = vehicle.fuselages.fuselage.width
    Lf = vehicle.fuselages.fuselage.lengths.total
    Sw = vehicle.wings.main_wing.areas.reference
    cw = vehicle.wings.main_wing.chords.mean_aerodynamic
    b  = vehicle.wings.main_wing.spans.projected
    Sh = vehicle.wings.horizontal_stabilizer.areas.reference
    Sv = vehicle.wings.vertical_stabilizer.areas.reference
    lh = vehicle.wings.horizontal_stabilizer.origin[0] + vehicle.wings.horizontal_stabilizer.aerodynamic_center[0] - vehicle.mass_properties.center_of_gravity[0]
    lv = vehicle.wings.vertical_stabilizer.origin[0] + vehicle.wings.vertical_stabilizer.aerodynamic_center[0] - vehicle.mass_properties.center_of_gravity[0]

    '''
    #when you run want to output results to a file
    unscaled_inputs = nexus.optimization_problem.inputs[:,1] #use optimization problem inputs here
    input_scaling   = nexus.optimization_problem.inputs[:,3]
    scaled_inputs   = unscaled_inputs/input_scaling
    problem_inputs=[]
    
    for value in unscaled_inputs:
        problem_inputs.append(value) 
    file=open('results.txt' , 'ab')

    file.write('fuel weight = ')
    file.write(str( summary.base_mission_fuelburn))
  
    file.write(', inputs = ')
    file.write(str(problem_inputs))
    
    file.write('\n') 
    file.close()
    '''
    
    
    
    
    return nexus    
