# Procedure.py
# 
# Created:  Nov 2015, Carlos / Tarik
# Modified: Jun 2016, T. MacDonald

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

from SUAVE.Methods.Noise.Fidelity_One.Airframe import noise_airframe_Fink
from SUAVE.Methods.Noise.Fidelity_One.Engine import noise_SAE

from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import pnl_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_tone_correction
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import epnl_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_certification_limits
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_geometric
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_counterplot

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
    procedure.initial_sizing = initial_sizing
    procedure.weights = weight
    procedure.weights_sizing = weights_sizing
    procedure.estimate_clmax = estimate_clmax
    
    # find the weights
    # finalizes the data dependencies
    procedure.finalize = finalize   
    
    # Noise evaluation
    procedure.noise                   = Process()
    procedure.noise.sideline_init     = noise_sideline_init
    procedure.noise.takeoff_init      = noise_takeoff_init
    procedure.noise.noise_sideline    = noise_sideline
    procedure.noise.noise_flyover     = noise_flyover
    procedure.noise.noise_approach    = noise_approach 
  
    # post process the results
    procedure.post_process = post_process
        
    # done!
    return procedure

# ----------------------------------------------------------------------        
#   Initial sizing - update design with design variables
# ----------------------------------------------------------------------    

def initial_sizing(nexus):    
    
    for config in nexus.vehicle_configurations:
        
        config.mass_properties.max_zero_fuel = nexus.MZFW_ratio*config.mass_properties.max_takeoff
            
        for wing in config.wings:
            wing = SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)
            
            wing.areas.exposed  = 0.8 * wing.areas.wetted
            wing.areas.affected = 0.6 * wing.areas.reference
        
        #compute atmosphere conditions for turbofan sizing
        
        air_speed   = nexus.missions.base.segments['cruise'].air_speed 
        altitude    = nexus.missions.base.segments['climb_5'].altitude_end
        atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
       
        
        freestream             = atmosphere.compute_values(altitude) #freestream conditions
        mach_number            = air_speed/freestream.speed_of_sound
        freestream.mach_number = mach_number
        freestream.velocity    = air_speed
        freestream.gravity     = 9.81
        
        conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
        conditions.freestream = freestream
        
        
        turbofan_sizing(config.propulsors['turbofan'], mach_number, altitude)
        compute_turbofan_geometry(config.propulsors['turbofan'], conditions)
        # diff the new data
        config.store_diff()  
        
    return nexus
        
# ----------------------------------------------------------------------        
#   Weights sizing loop
# ----------------------------------------------------------------------    
def weights_sizing(nexus):
    
    vehicle = nexus.vehicle_configurations.base
    mission = nexus.missions.max_range
    mission.design_range = 3100.*Units.nautical_miles
    find_target_range(nexus,mission)
    results = nexus.results
    max_payload = vehicle.mass_properties.max_payload
    payload     = vehicle.mass_properties.payload
    
    residual = 1. * vehicle.mass_properties.max_takeoff
    tol = 5. # kg
    mtow_guess = 0.
    iter = 0
    
    while abs(residual) > tol and iter < 10:
        iter = iter + 1
        mtow_guess           = mtow_guess + residual
        operating_empty      = vehicle.mass_properties.operating_empty
        max_zero_fuel_weight = operating_empty + max_payload        
        vehicle.mass_properties.max_zero_fuel = max_zero_fuel_weight
        vehicle.mass_properties.max_takeoff   = mtow_guess
        vehicle.mass_properties.takeoff       = mtow_guess
        mission.segments[0].analyses.weights.vehicle.mass_properties.takeoff = mtow_guess        
        nexus = weight(nexus)
        
        vehicle.mass_properties.max_zero_fuel = vehicle.mass_properties.operating_empty + max_payload
        nexus = weight(nexus)
        
        vehicle.mass_properties.max_zero_fuel = vehicle.mass_properties.operating_empty + max_payload
        nexus = weight(nexus)
      
        nexus.analyses.finalize()

        results.max_range = mission.evaluate()
    
        # Fuel margin and base fuel calculations
        operating_empty          = vehicle.mass_properties.operating_empty        
        max_range_landing_weight = results.max_range.segments[-1].conditions.weights.total_mass[-1]  
        final_weight_expected    = operating_empty + payload
        max_range_fuel_margin    = (max_range_landing_weight - final_weight_expected)
        residual = - max_range_fuel_margin 

    for config in nexus.vehicle_configurations:        
        config.mass_properties.max_takeoff = nexus.vehicle_configurations.base.mass_properties.max_takeoff
        config.mass_properties.takeoff = nexus.vehicle_configurations.base.mass_properties.takeoff
        config.mass_properties.max_zero_fuel = nexus.vehicle_configurations.base.mass_properties.max_zero_fuel
        config.mass_properties.operating_empty = nexus.vehicle_configurations.base.mass_properties.operating_empty
               
        # diff the new data
        config.store_diff()  
        
    return nexus

# ----------------------------------------------------------------------        
#   Weights sizing loop
# ----------------------------------------------------------------------    
def design_mission(nexus):
   
    vehicle = nexus.vehicle_configurations.base
    mission = nexus.missions.base
    mission.design_range = 1500.*Units.nautical_miles
    find_target_range(nexus,mission)
    results = nexus.results
    
    payload               = vehicle.mass_properties.payload
    operating_empty       = vehicle.mass_properties.operating_empty 
    final_weight_expected = operating_empty + payload
    
    residual = vehicle.mass_properties.takeoff - (vehicle.mass_properties.takeoff-final_weight_expected)*1500./3100.
    tol = 5. # kg
    mtow_guess = 0.
    iter = 0
        
    while abs(residual) > tol and iter < 10:
        iter = iter + 1
        mtow_guess = mtow_guess + residual
        vehicle.mass_properties.takeoff  = mtow_guess
        mission.segments[0].analyses.weights.vehicle.mass_properties.takeoff = mtow_guess      
        nexus.analyses.finalize()
        results.base = mission.evaluate()               
        landing_weight = results.base.segments[-1].conditions.weights.total_mass[-1]          
        fuel_margin    = (landing_weight - final_weight_expected)
        residual = - fuel_margin 

    return nexus

# ----------------------------------------------------------------------        
#   Weights sizing loop
# ----------------------------------------------------------------------    
def short_field_mission(nexus):
      
    vehicle = nexus.vehicle_configurations.short_field_takeoff
    mission = nexus.missions.short_field
    mission.design_range = 750.*Units.nautical_miles
    find_target_range(nexus,mission)
    results = nexus.results
    
    payload               = vehicle.mass_properties.payload
    operating_empty       = vehicle.mass_properties.operating_empty 
    final_weight_expected = operating_empty + payload
    
    residual = vehicle.mass_properties.takeoff - (vehicle.mass_properties.takeoff-final_weight_expected)*750./3100.
    tol = 5. # kg
    mtow_guess = 0.
    iter = 0
        
    while abs(residual) > tol and iter < 10:
        iter = iter + 1
        mtow_guess = mtow_guess + residual
        vehicle.mass_properties.takeoff  = mtow_guess
        mission.segments[0].analyses.weights.vehicle.mass_properties.takeoff = mtow_guess      
        nexus.analyses.finalize()
        results.short_field = mission.evaluate()               
        landing_weight = results.short_field.segments[-1].conditions.weights.total_mass[-1]          
        fuel_margin    = (landing_weight - final_weight_expected)
        residual = - fuel_margin 
##        print 'residual short field' , residual

    return nexus
# ----------------------------------------------------------------------        
#   Sizing
# ----------------------------------------------------------------------    

def estimate_clmax(nexus):
    
    # Condition to CLmax calculation: 90KTAS @ 10000ft, ISA
    state = Data()
    state.conditions = Data()
    state.conditions.freestream = Data()
    state.conditions.freestream.density           = 0.90477283
    state.conditions.freestream.dynamic_viscosity = 1.69220918e-05
    state.conditions.freestream.velocity          = 90. * Units.knots
    
    #Takeoff CL_max
    config = nexus.vehicle_configurations.takeoff
    settings = nexus.analyses.takeoff.aerodynamics.settings
    maximum_lift_coefficient,CDi = compute_max_lift_coeff(state,settings,config)
    config.maximum_lift_coefficient = maximum_lift_coefficient 
    # diff the new data
    config.store_diff()  
    
    #Takeoff CL_max - for short field config
    config = nexus.vehicle_configurations.short_field_takeoff
    settings = nexus.analyses.short_field_takeoff.aerodynamics.settings
    maximum_lift_coefficient,CDi = compute_max_lift_coeff(state,settings,config)
    config.maximum_lift_coefficient = maximum_lift_coefficient 
    # diff the new data
    config.store_diff()  
        
    # compute V2 speed for noise, based in MTOW   
    config = nexus.vehicle_configurations.takeoff       
    weight = config.mass_properties.max_takeoff
    reference_area = config.wings.main_wing.areas.reference
    max_CL_takeoff = config.maximum_lift_coefficient
    stall_speed = (2 * 9.81 * weight / (1.225 * reference_area * max_CL_takeoff)) ** 0.5
    V2_speed    = 1.20 * stall_speed            
    speed_for_noise = V2_speed + nexus.noise_V2_increase
    
    #nexus.missions.takeoff_initialization.segments.climb.air_speed   = speed_for_noise    
    nexus.missions.takeoff.segments.climb.air_speed          = speed_for_noise
    nexus.missions.takeoff.segments.cutback.air_speed        = speed_for_noise
    nexus.missions.sideline_takeoff.segments.climb.air_speed = speed_for_noise
    
    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------
    landing = nexus.vehicle_configurations.landing
    landing_conditions = Data()
    landing_conditions.freestream = Data()

    # landing weight
    landing.mass_properties.takeoff = 0.85 * config.mass_properties.takeoff
    landing.mass_properties.landing = 0.85 * config.mass_properties.takeoff
      
    # Landing CL_max
    settings = nexus.analyses.landing.aerodynamics.settings
    maximum_lift_coefficient,CDi = compute_max_lift_coeff(state,settings,landing)
    landing.maximum_lift_coefficient = maximum_lift_coefficient 
    
    # compute approach speed
    weight = landing.mass_properties.landing
    stall_speed = (2 * 9.81 * weight / (1.225 * reference_area * maximum_lift_coefficient)) ** 0.5
    Vref_speed  = 1.23 * stall_speed  
    nexus.missions.landing.segments.descent.air_speed   = Vref_speed  + 10. * Units.knots    
    
    # diff the new data
    landing.store_diff()      
    
    return nexus
        
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
#   Max Range Mission
# ----------------------------------------------------------------------    
    
def max_range_mission(nexus):
    
    mission = nexus.missions.max_range
    mission.design_range = 3100.*Units.nautical_miles
    find_target_range(nexus,mission)
    results = nexus.results
    results.max_range = mission.evaluate()
    
    return nexus
    
# ----------------------------------------------------------------------        
#   Sideline noise
# ----------------------------------------------------------------------    
    
def noise_sideline(nexus):         
    
    mission = nexus.missions.sideline_takeoff
    nexus.analyses.takeoff.noise.settings.sideline = 1
    nexus.analyses.takeoff.noise.settings.flyover  = 0
    results = nexus.results
    #results.sideline = mission.evaluate()
    #SUAVE.Input_Output.SUAVE.archive(results.sideline,'sideline.res')
    results.sideline = SUAVE.Input_Output.SUAVE.load('sideline.res')   
    
    #Determine the x0
    x0 = 0.    
    position_vector = results.sideline.conditions.climb.frames.inertial.position_vector
    degree = 3
    coefs = np.polyfit(-position_vector[:,2],position_vector[:,0],degree)
    for idx,coef in enumerate(coefs):
        x0 += coef * 304.8 ** (degree-idx)

    nexus.analyses.takeoff.noise.settings.mic_x_position = x0 
    
    noise_segment = results.sideline.segments.climb
    #SUAVE.Input_Output.SUAVE.archive(results.sideline,'sideline.res')
    noise_config  = nexus.vehicle_configurations.takeoff
    noise_analyse = nexus.analyses.takeoff
    noise_config.engine_flag = 1
    
    noise_config.print_output = 0
    noise_config.output_file  = 'Noise_Sideline.dat'
    noise_config.output_file_engine = 'Noise_Sideline_Engine.dat'
    
    
    if nexus.npoints_sideline_sign == -1:
        noise_result_takeoff_SL = 500. + nexus.missions.sideline_takeoff.segments.climb.state.numerics.number_control_points
    else:
        noise_result_takeoff_SL = compute_noise(noise_config,noise_analyse,noise_segment)    
    
    
    nexus.summary.noise = Data()
    nexus.summary.noise.sideline = noise_result_takeoff_SL     
    
    return nexus


# ----------------------------------------------------------------------        
#   Flyover noise
# ----------------------------------------------------------------------    
    
def noise_flyover(nexus):       
    
    mission = nexus.missions.takeoff
    nexus.analyses.takeoff.noise.settings.flyover = 1
    nexus.analyses.takeoff.noise.settings.sideline = 0
    results = nexus.results
    #results.flyover = mission.evaluate()  
    #SUAVE.Input_Output.SUAVE.archive(results.flyover,'flyover.res')
    results.flyover = SUAVE.Input_Output.SUAVE.load('flyover.res')   
    
    noise_segment = results.flyover.segments.climb
    noise_config  = nexus.vehicle_configurations.takeoff
    noise_analyse = nexus.analyses.takeoff
    noise_config.engine_flag = 1
    
    noise_config.print_output = 0
    noise_config.output_file  = 'Noise_Flyover_climb.dat'
    noise_config.output_file_engine = 'Noise_Flyover_climb_Engine.dat'
    
    if nexus.npoints_takeoff_sign == -1:
        noise_result_takeoff_FL_clb = 500. + nexus.missions.sideline_takeoff.segments.climb.state.numerics.number_control_points
    else:    
        noise_result_takeoff_FL_clb = compute_noise(noise_config,noise_analyse,noise_segment)    

    noise_segment = results.flyover.segments.cutback
    noise_config  = nexus.vehicle_configurations.cutback
    noise_config.print_output = 0
    noise_config.engine_flag = 1
    noise_config.output_file  = 'Noise_Flyover_cutback.dat'
    noise_config.output_file_engine = 'Noise_Flyover_cutback_Engine.dat'
    
    if nexus.npoints_takeoff_sign == -1:
        noise_result_takeoff_FL_cutback = 500. + nexus.missions.sideline_takeoff.segments.climb.state.numerics.number_control_points
    else:        
        noise_result_takeoff_FL_cutback = compute_noise(noise_config,noise_analyse,noise_segment)  

    noise_result_takeoff_FL = 10. * np.log10(10**(noise_result_takeoff_FL_clb/10)+10**(noise_result_takeoff_FL_cutback/10))
   
    nexus.summary.noise.flyover = noise_result_takeoff_FL    

    return nexus

# ----------------------------------------------------------------------        
#   Approach noise
# ----------------------------------------------------------------------    
    
def noise_approach(nexus):       
    
    mission = nexus.missions.landing
    nexus.analyses.landing.noise.settings.approach = 1    
    results = nexus.results
    results.approach = mission.evaluate()
    #SUAVE.Input_Output.SUAVE.archive(results.approach,'approach.res')
    results.approach = SUAVE.Input_Output.SUAVE.load('approach.res')  
    
    
    noise_segment = results.approach.segments.descent
    noise_config  = nexus.vehicle_configurations.landing
    noise_analyse = nexus.analyses.landing
    
    noise_config.print_output = 0
    noise_config.output_file  = 'Noise_Approach.dat'
    noise_config.output_file_engine = 'Noise_Approach_Engine.dat'
    
    noise_config.engine_flag = 1 
   
    noise_result_approach = compute_noise(noise_config,noise_analyse,noise_segment)
       
    nexus.summary.noise.approach = noise_result_approach
    
    return nexus

# ----------------------------------------------------------------------        
#   NOISE CALCULATION
# ----------------------------------------------------------------------
def compute_noise(config,analyses,noise_segment):

    turbofan = config.propulsors['turbofan']
    
    outputfile        = config.output_file
    outputfile_engine = config.output_file_engine
    print_output      = config.print_output
    engine_flag       = config.engine_flag  #remove engine noise component from the approach segment
    
    geometric = noise_geometric(noise_segment,analyses,config)
    
    airframe_noise = noise_airframe_Fink(config,analyses,noise_segment,print_output,outputfile)  

    engine_noise   = noise_SAE(turbofan,noise_segment,config,analyses,print_output,outputfile_engine)

    noise_sum = 10. * np.log10(10**(airframe_noise[0]/10)+ (engine_flag)*10**(engine_noise[0]/10))


    return noise_sum

# ----------------------------------------------------------------------        
#   Weights
# ----------------------------------------------------------------------    

def weight(nexus):   
    
    for tag,config in list(nexus.analyses.items()):
        weights = config.weights.evaluate()
    
    return nexus


# ----------------------------------------------------------------------
#   Finalizing Function (make part of optimization nexus)[needs to come after simple sizing doh]
# ----------------------------------------------------------------------    

def finalize(nexus):
    
    nexus.analyses.finalize()   
    
    return nexus         

def noise_sideline_init(nexus):
    # Update number of control points for noise
    mission = nexus.missions.sideline_takeoff       
    results = nexus.results
    results.sideline_initialization = mission.evaluate()
    
    n_points   = np.ceil(results.sideline_initialization.segments.climb.conditions.frames.inertial.time[-1] /0.5 +1)

    nexus.npoints_sideline_sign=np.sign(n_points)
    nexus.missions.sideline_takeoff.segments.climb.state.numerics.number_control_points = np.minimum(200, np.abs(n_points))  

    
    return nexus


def noise_takeoff_init(nexus):
    
    # Update number of control points for noise
    mission = nexus.missions.takeoff       
    results = nexus.results
    results.takeoff_initialization = mission.evaluate()
    
    n_points   = np.ceil(results.takeoff_initialization.segments.climb.conditions.frames.inertial.time[-1] /0.5 +1)
    nexus.npoints_takeoff_sign=np.sign(n_points)

    nexus.missions.takeoff.segments.climb.state.numerics.number_control_points = np.minimum(200, np.abs(n_points))
    

    return nexus

# ----------------------------------------------------------------------
#   Takeoff Field Length Evaluation
# ----------------------------------------------------------------------    
    
def takeoff_field_length(nexus):
    
    # import tofl analysis module
    estimate_tofl = SUAVE.Methods.Performance.estimate_take_off_field_length
    
    # unpack data
    results  = nexus.results
    summary = nexus.summary
    analyses = nexus.analyses
    missions = nexus.missions
    config   = nexus.vehicle_configurations.takeoff
    
    # defining required data for tofl evaluation
    takeoff_airport = missions.takeoff.airport    
    ref_weight      = config.mass_properties.takeoff  

    takeoff_field_length,second_segment_climb_gradient_takeoff = estimate_tofl(config,analyses, takeoff_airport,1)
    
    # pack results
    summary.takeoff_field_length = takeoff_field_length
    summary.second_segment_climb_gradient_takeoff = second_segment_climb_gradient_takeoff
    

    return nexus

# ----------------------------------------------------------------------
#   landing Field Length Evaluation
# ----------------------------------------------------------------------    
    
def landing_field_length(nexus):
    
    # import tofl analysis module
    estimate_landing = SUAVE.Methods.Performance.estimate_landing_field_length
    
    # unpack data
    results  = nexus.results
    summary = nexus.summary
    analyses = nexus.analyses
    missions = nexus.missions
    config   = nexus.vehicle_configurations.landing
    
    # defining required data for tofl evaluation
    takeoff_airport = missions.takeoff.airport    
    ref_weight      = config.mass_properties.landing  
    
    landing_field_length = estimate_landing(config,analyses, takeoff_airport)
    
    # pack results
    summary.landing_field_length = landing_field_length

    return nexus

# ----------------------------------------------------------------------
#   Short Field Length Evaluation
# ----------------------------------------------------------------------    
    
def short_takeoff_field_length(nexus):
    
    # import tofl analysis module
    estimate_tofl = SUAVE.Methods.Performance.estimate_take_off_field_length
    
    # unpack data
    results  = nexus.results
    summary = nexus.summary
    analyses = nexus.analyses
    missions = nexus.missions
    config   = nexus.vehicle_configurations.short_field_takeoff
    
    # defining required data for tofl evaluation
    takeoff_airport = missions.base.airport    
    ref_weight      = config.mass_properties.takeoff 
    short_takeoff_field_length,second_segment_climb_gradient_SF = estimate_tofl(config,analyses, takeoff_airport,1)
    
    # pack results
    summary.short_takeoff_field_length = short_takeoff_field_length 
    summary.second_segment_climb_gradient_short_field = second_segment_climb_gradient_SF
        
    return nexus
   
# ----------------------------------------------------------------------
#   Post Process Results to give back to the optimizer
# ----------------------------------------------------------------------   

def post_process(nexus):
    
    # Unpack data
    vehicle               = nexus.vehicle_configurations.base  
    short_vehicle         = nexus.vehicle_configurations.short_field_takeoff
    results               = nexus.results
    summary               = nexus.summary
    missions              = nexus.missions      
   
    #calculation of noise certification limits based on the aircraft weight
    noise_limits = noise_certification_limits(results, vehicle)
    summary.noise_approach_margin = noise_limits[0] - summary.noise.approach 
    summary.noise_flyover_margin  = noise_limits[1] - summary.noise.flyover  
    summary.noise_sideline_margin = noise_limits[2] - summary.noise.sideline 
    
    summary.noise_margin  =  summary.noise_approach_margin + summary.noise_sideline_margin + summary.noise_flyover_margin
    
    print("Sideline = ", summary.noise.sideline)
    print("Flyover = ", summary.noise.flyover)
    print("Approach = ", summary.noise.approach)
  
    return nexus    


def save_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'results_noise.res')
    return
