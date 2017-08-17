# sizing_loop.py
#
# Created:  Jun 2015, SUAVE Team
# Modified: 

""" setup file for a sizing loop with a 737-aircraft
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data

import numpy as np
import copy, time

import matplotlib
import pylab as plt



from SUAVE.Analyses.Process import Process
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift.compute_max_lift_coeff import compute_max_lift_coeff
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Propulsion.compute_turbofan_geometry import compute_turbofan_geometry
from SUAVE.Sizing.Sizing_Loop import Sizing_Loop
from SUAVE.Optimization.Nexus import Nexus
#from SUAVE.Optimization.write_optimization_outputs import write_optimization_outputs

import sys
sys.path.append('../noise_optimization') #import structure from noise_optimization
sys.path.append('../Vehicles')
import Analyses
import Missions
from Boeing_737 import vehicle_setup, configs_setup
matplotlib.interactive(True)
# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    
    #initialize the problem
    nexus                        = Nexus()
    vehicle = vehicle_setup()
    nexus.vehicle_configurations = configs_setup(vehicle)
    nexus.analyses               = Analyses.setup(nexus.vehicle_configurations)
    nexus.missions               = Missions.setup(nexus.analyses)
    
    #problem = Data()
    #nexus.optimization_problem       = problem
    nexus.procedure                  = setup()
    nexus.sizing_loop                = Sizing_Loop()
    nexus.total_number_of_iterations = 0
    
    evaluate_problem(nexus)
    results = nexus.results

    err      = nexus.sizing_loop.norm_error
    err_true = 0.00975078 #for 1E-2 tol
    error    = abs((err-err_true)/err)
    print 'error = ', error
    assert(error<1e-5), 'sizing loop regression failed'    
    
    #output=nexus._really_evaluate() #run; use optimization setup without inputs
    return
    
def evaluate_problem(nexus):
    for key,step in nexus.procedure.items():
        if hasattr(step,'evaluate'):
            self = step.evaluate(nexus)
        else:
            nexus = step(nexus)
        self = nexus
    return nexus
# ----------------------------------------------------------------------        
#   Setup
# ----------------------------------------------------------------------   

def setup():
    # ------------------------------------------------------------------
    #   Analysis Procedure
    # ------------------------------------------------------------------ 
    
    # size the base config
    procedure = Process()
    procedure.run_sizing_loop       = run_sizing_loop #size aircraft and run mission
    
    return procedure


   
   
def run_sizing_loop(nexus):
    vehicle    = nexus.vehicle_configurations.base
    configs    = nexus.vehicle_configurations
    analyses   = nexus.analyses
    mission    = nexus.missions.base
    

    #initial guesses
    m_guess       = 60000.       
    scaling       = np.array([1E4])
    y             = np.array([m_guess])/scaling
    min_y         = [.05]
    max_y         = [10.]
    
    
    #create sizing loop object
    sizing_loop = nexus.sizing_loop
    #assign to sizing loop
    
    sizing_loop.tolerance                                      = 1E-2 #fraction difference in mass and energy between iterations
    sizing_loop.initial_step                                   = 'Default' #Default, Table, SVR
    sizing_loop.update_method                                  = 'successive_substitution' #'successive_substitution','newton-raphson', 'broyden'
    sizing_loop.default_y                                      = y
    sizing_loop.min_y                                          = min_y
    sizing_loop.max_y                                          = max_y
    sizing_loop.default_scaling                                = scaling
    sizing_loop.sizing_evaluation                              = sizing_evaluation
    sizing_loop.maximum_iterations                             = 50
    sizing_loop.write_threshhold                               = 50.
    sizing_loop.output_filename                                = 'sizing_outputs.txt' #used if you run optimization
    
    nexus.max_iter                                             = sizing_loop.maximum_iterations  #used to pass it to constraints
  
    #run the sizing loop
    nexus = sizing_loop(nexus)
    return nexus   
   
   
# ----------------------------------------------------------------------        
#   Sizing
# ----------------------------------------------------------------------   
 
def simple_sizing(nexus):
    configs    = nexus.vehicle_configurations
    analyses   = nexus.analyses
    base       = configs.base

    
    
    m_guess                            = nexus.m_guess #take in sizing inputs
    base.mass_properties.max_takeoff   = m_guess
 
    #find conditions
    air_speed   = nexus.missions.base.segments['cruise'].air_speed 
    altitude    = nexus.missions.base.segments['climb_3'].altitude_end
    atmosphere  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    freestream  = atmosphere.compute_values(altitude)
    freestream0 = atmosphere.compute_values(6000.*Units.ft)  #cabin altitude
    
    
    diff_pressure                  = np.max(freestream0.pressure-freestream.pressure,0)
    fuselage                       = base.fuselages['fuselage']
    fuselage.differential_pressure = diff_pressure 
    
    #now size engine
    mach_number        = air_speed/freestream.speed_of_sound
    
    #now add to freestream data object
    freestream.velocity    = air_speed
    freestream.mach_number = mach_number
    freestream.gravity     = 9.81
    
    conditions             = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()   #assign conditions in form for propulsor sizing
    conditions.freestream  = freestream
    
    nose_load_fraction     = .06
   
    #now evaluate all of the vehicle configurations
    for config in configs:
        config.wings.horizontal_stabilizer.areas.reference = (26.0/92.0)*config.wings.main_wing.areas.reference
         
         
        for wing in config.wings:
            
            wing = SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)
            
            wing.areas.exposed  = 0.8 * wing.areas.wetted
            wing.areas.affected = 0.6 * wing.areas.reference
            


        fuselage                       = config.fuselages['fuselage']
        fuselage.differential_pressure = diff_pressure 
     
        #now evaluate weights

        # diff the new data
        
        config.mass_properties.max_takeoff     = m_guess #take in parameters
        config.mass_properties.takeoff         = m_guess 
        config.mass_properties.max_zero_fuel   = base.mass_properties.max_zero_fuel
        config.store_diff()
       
       
    #now evaluate the weights   
    weights = analyses.base.weights.evaluate() #base.weights.evaluate()  
    #update zfw
    empty_weight       = base.mass_properties.operating_empty
    payload            = base.mass_properties.max_payload
    zfw                = empty_weight + payload 
    
    base.max_zero_fuel = zfw  
    base.store_diff()
    for config in configs:
        config.pull_base()
    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------
    landing = nexus.vehicle_configurations.landing
    landing_conditions = Data()
    landing_conditions.freestream = Data()

    # landing weight
    landing.mass_properties.landing = base.mass_properties.max_zero_fuel
    
    # Landing CL_max
    altitude                                         = nexus.missions.base.segments[-1].altitude_end
    atmosphere                                       = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    
    freestream             = atmosphere.compute_values(altitude)
    mu                     = freestream.dynamic_viscosity
    rho                    = freestream.density

    landing_conditions.freestream.velocity           = nexus.missions.base.segments['descent_3'].air_speed
    landing_conditions.freestream.density            = rho
    landing_conditions.freestream.dynamic_viscosity  = mu/rho
    CL_max_landing,CDi                               = compute_max_lift_coeff(landing,landing_conditions)
    landing.maximum_lift_coefficient                 = CL_max_landing
    # diff the new data
    landing.store_diff()
    
    
    #Takeoff CL_max
    takeoff                                          = nexus.vehicle_configurations.takeoff
    takeoff_conditions                               = Data()
    takeoff_conditions.freestream                    = Data()    
    altitude                                         = nexus.missions.base.airport.altitude
    atmosphere                                       = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    freestream                                       = atmosphere.compute_values(altitude)
    mu                                               = freestream.dynamic_viscosity
    rho                                              = freestream.density
    takeoff_conditions.freestream.velocity           = nexus.missions.base.segments.climb_1.air_speed
    takeoff_conditions.freestream.density            = rho
    takeoff_conditions.freestream.dynamic_viscosity  = mu/rho 
    max_CL_takeoff,CDi                               = compute_max_lift_coeff(takeoff,takeoff_conditions) 
    takeoff.maximum_lift_coefficient                 = max_CL_takeoff
    
    takeoff.store_diff()
    
   

    #Base config CL_max
    base                                          = nexus.vehicle_configurations.base
    base_conditions                               = Data()
    base_conditions.freestream                    = Data()    
    altitude                                      = nexus.missions.base.airport.altitude
    atmosphere                                    = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    freestream                                    = atmosphere.compute_values(altitude)
    mu                                            = freestream.dynamic_viscosity
    rho                                           = freestream.density
    
    
    
    base_conditions.freestream.velocity           = nexus.missions.base.segments.climb_1.air_speed
    base_conditions.freestream.density            = rho
    base_conditions.freestream.dynamic_viscosity  = mu/rho 
    max_CL_base,CDi                               = compute_max_lift_coeff(base,base_conditions) 
    base.maximum_lift_coefficient                 = max_CL_base    
    base.store_diff()
    
    # done!
    
    return nexus
    
    
def sizing_evaluation(y,nexus, scaling):
    #unpack inputs
    m_guess           = y[0]*scaling[0]    
    nexus.m_guess     = m_guess
    configs           = nexus.vehicle_configurations
    base              = configs.base
    analyses          = nexus.analyses
    mission           = nexus.missions.base

    simple_sizing(nexus)
    analyses.finalize() #wont run without this
    
    #now run the mission
    results = mission.evaluate()
    #results = nexus.results
  
    #handle outputs
    segments         = results.segments
    fuel_out         = segments[0].conditions.weights.total_mass[0,0]-segments[-1].conditions.weights.total_mass[-1,0] 
    passenger_weight = base.passenger_weights.mass_properties.mass
    
    #sizing loop outputs

    mass_out     = base.mass_properties.operating_empty[0]+fuel_out+passenger_weight
    dm           = (mass_out-m_guess)/m_guess
    nexus.dm     = dm

   
    nexus.results  =results #pack up results
   
    f     = np.array([dm])
    y_out = np.array([mass_out])/scaling
    
    return f, y_out
    

# ----------------------------------------------------------------------
#   Finalizing Function (make part of optimization nexus)[needs to come after simple sizing doh]
# ----------------------------------------------------------------------    

def finalize(nexus):
    
    #nexus.vehicle_configurations.finalize()
    nexus.analyses.finalize()   
    
    return nexus         
       

if __name__ == '__main__':
    main()
    plt.show()
