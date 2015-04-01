
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
from SUAVE.Methods.Performance import estimate_take_off_field_length
from SUAVE.Methods.Performance import estimate_landing_field_length 

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
    process.evaluate_field_length    = evaluate_field_length  
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
    
##    # wing areas
##    for wing in base.wings:
##        wing.areas.wetted   = 2.00 * wing.areas.reference
##        wing.areas.affected = 0.60 * wing.areas.reference
##        wing.areas.exposed  = 0.75 * wing.areas.wetted

    # wing simple sizing function
    # size main wing
    base.wings['main_wing'] = wing_planform(base.wings['main_wing'])
    # reference values for empennage scaling, keeping tail coeff. volumes constants
    Sref = base.wings['main_wing'].areas.reference
    span = base.wings['main_wing'].spans.projected
    CMA  = base.wings['main_wing'].chords.mean_aerodynamic    
    Sh = 32.48 * (Sref * CMA)  / (124.86 * 4.1535)  # hardcoded values represent the reference airplane data
    Sv = 26.40 * (Sref * span) / (124.86 * 35.35)   # hardcoded values represent the reference airplane data
        
    # empennage scaling
    base.wings['horizontal_stabilizer'].areas.reference = Sh
    base.wings['vertical_stabilizer'].areas.reference = Sv
    # sizing of new empennages
    base.wings['horizontal_stabilizer'] = wing_planform(base.wings['horizontal_stabilizer']) 
    base.wings['vertical_stabilizer']   = wing_planform(base.wings['vertical_stabilizer'])   
    
    # wing areas
    for wing in base.wings:
        wing.areas.wetted   = 2.00 * wing.areas.reference
        wing.areas.affected = 0.60 * wing.areas.reference
        wing.areas.exposed  = 0.75 * wing.areas.wetted
         
    # fuselage seats
    base.fuselages['fuselage'].number_coach_seats = base.passengers        
    
    # Weight estimation
    breakdown = analyses.configs.base.weights.evaluate()
    
    # pack
    base.mass_properties.breakdown = breakdown
    base.mass_properties.operating_empty = breakdown.empty
       
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
    
def evaluate_field_length(interface):
    configs=interface.configs
    # unpack
    configs=interface.configs
    analyses=interface.analyses
    mission=interface.analyses.missions.base
    results=interface.results
    airport = mission.airport
    
    takeoff_config = configs.takeoff
    landing_config = configs.landing
    
    
    # evaluate
    TOFL = estimate_take_off_field_length(takeoff_config,analyses,airport)
    LFL = estimate_landing_field_length(landing_config,airport)
    
    # pack
    field_length = SUAVE.Core.Data()
    field_length.takeoff = TOFL[0]
    field_length.landing = LFL[0]
    
    results.field_length = field_length
 
    
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
    
    # Unpack
    vehicle               = interface.configs.base    
    results               = interface.results
    mission_profile       = results.missions.base    
  
    # Weights
    max_zero_fuel     = vehicle.mass_properties.max_zero_fuel
    operating_empty   = vehicle.mass_properties.operating_empty
    payload           = vehicle.mass_properties.payload    
          
    # pack
    summary = SUAVE.Core.Results()
    
    # TOFL for MTOW @ SL, ISA
    summary.total_range           =mission_profile.segments[-1].conditions.frames.inertial.position_vector[-1,0]
    summary.GLW                   =mission_profile.segments[-1].conditions.weights.total_mass[-1,0] 
    summary.takeoff_field_length  =results.field_length.takeoff
    summary.landing_field_length  =results.field_length.landing
    # MZFW margin calculation
    summary.max_zero_fuel_margin  = max_zero_fuel - (operating_empty + payload)
    # fuel margin calculation
    
    # Print outs   
    printme = Data()
    printme.weight_empty = operating_empty
    printme.total_range  =summary.total_range/1000.
    printme.GLW     =summary.GLW
    printme.tofl    = summary.takeoff_field_length
    printme.lfl     = summary.landing_field_length
    
    #printme.range_max    = summary.range_max_nmi    
    #printme.max_zero_fuel_margin      = summary.max_zero_fuel_margin
    #printme.available_fuel_margin     = summary.available_fuel_margin
    
    print "RESULTS"
    print printme  
    
    inputs = interface.inputs
    import datetime
    fid = open('Results.dat','a')

    fid.write('{:18.10f} ; {:18.10f} ; {:18.10f} ; {:18.10f} ; {:18.10f} ; {:18.10f} ; {:18.10f} ;'.format( \
        inputs.aspect_ratio,
        inputs.reference_area,
        inputs.sweep,inputs.design_thrust,
        inputs.wing_thickness,
        inputs.MTOW,
        inputs.MZFW_ratio,
        operating_empty ,
        summary.takeoff_field_length , 
        summary.landing_field_length
    ))
    fid.write(datetime.datetime.now().strftime("%I:%M:%S"))
    fid.write('\n')

##    print interface.configs.takeoff.maximum_lift_coefficient
    
    return summary

if __name__ == '__main__':
    main()
    
    
