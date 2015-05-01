
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
    #interface.evaluate=evaluate_interface
    # ------------------------------------------------------------------
    #   Analysis Process
    # ------------------------------------------------------------------
    
    process = interface.process
    
    # the input unpacker
    process.unpack_inputs = unpack_inputs
    
    # size the base config
    #process.simple_sizing = simple_sizing
    
  
    #sizing loop; size the aircraft
    process.sizing_loop=sizing_loop
    # finalizes the data dependencies
    #process.finalize = finalize
    # the missions
    
    #process.missions = evaluate_mission

    # performance studies
    process.evaluate_field_length    = evaluate_field_length  
    #process.noise                   = noise
        
    # summarize the results
	
    process.summary = summarize
    #process.post_process=post_process
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
    mission=interface.analyses.missions.base.segments
    
    airport=interface.analyses.missions.base.airport
    cruise_altitude=inputs.cruise_altitude*Units.km
    atmo            = airport.atmosphere

    # apply the inputs
    vehicle.wings['main_wing'].aspect_ratio         = inputs.aspect_ratio
    vehicle.wings['main_wing'].areas.reference      = inputs.reference_area
    vehicle.wings['main_wing'].taper                = inputs.taper
    #vehicle.wings['main_wing'].sweep                = inputs.sweep * Units.deg
    vehicle.wings['main_wing'].thickness_to_chord   = inputs.wing_thickness
    mission['climb_1'].altitude_end                 = inputs.climb_alt_fraction_1*cruise_altitude
    mission['climb_2'].altitude_end                 = inputs.climb_alt_fraction_2*cruise_altitude
    
    mission['climb_3'].altitude_end                 = cruise_altitude
    mission['descent_1'].altitude_end               = inputs.desc_alt_fraction_1*cruise_altitude
    mission['cruise'].distance                      = inputs.cruise_range*Units.nautical_miles
    
    mission['climb_1'].air_speed                    = inputs.Vclimb1
    mission['climb_2'].air_speed                    = inputs.Vclimb2
    mission['climb_3'].air_speed                    = inputs.Vclimb3
    #initialize fuselage pressure differential from cruise altitude
    conditions0 = atmo.compute_values(12500.*Units.ft) #cabin pressure
    conditions = atmo.compute_values(cruise_altitude)
    
    p = conditions.pressure
    p0 = conditions0.pressure
    fuselage_diff_pressure=min(p0-p,0)
    vehicle.fuselages['fuselage'].differential_pressure = fuselage_diff_pressure
    vehicle.store_diff()
     
    return None

# ----------------------------------------------------------------------
#   Apply Simple Sizing Principles
# ----------------------------------------------------------------------

def simple_sizing(interface, Ereq, Preq):      
    from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform

    # unpack
    configs  = interface.configs
    analyses = interface.analyses   
    airport=interface.analyses.missions.base.airport
    atmo            = airport.atmosphere
    mission=interface.analyses.missions.base.segments 
    base = configs.base
    base.pull_base()
    base = configs.base
    base.pull_base()
    Vcruise=mission['cruise'].air_speed
    design_thrust=Preq*1.3/Vcruise; #approximate sealevel thrust of ducted fan
    #determine geometry of fuselage as well as wings
    fuselage=base.fuselages['fuselage']
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.fuselage_planform(fuselage)
    fuselage.areas.side_projected   = fuselage.heights.maximum*fuselage.lengths.cabin*1.1 #  Not correct
    c_vt                         =.08
    c_ht                         =.6
    w2v                          =8.54
    w2h                          =8.54
    base.wings['main_wing'] = wing_planform(base.wings['main_wing'])
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.vertical_tail_planform_raymer(base.wings['vertical_stabilizer'],base.wings['main_wing'], w2v, c_vt)
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.horizontal_tail_planform_raymer(base.wings['horizontal_stabilizer'],base.wings['main_wing'], w2h, c_ht)
   
    base.wings['horizontal_stabilizer'] = wing_planform(base.wings['horizontal_stabilizer']) 
    base.wings['vertical_stabilizer']   = wing_planform(base.wings['vertical_stabilizer'])   
    # wing areas
    for wing in base.wings:
        wing.areas.wetted   = 2.00 * wing.areas.reference
        wing.areas.affected = 0.60 * wing.areas.reference
        wing.areas.exposed  = 0.75 * wing.areas.wetted
  
    battery=base.energy_network['battery']
    ducted_fan=base.propulsors['ducted_fan']
    #SUAVE.Methods.Power.Battery.Sizing.initialize_from_energy_and_power(battery,Ereq,Preq)
    battery.mass_properties.mass  = Ereq/battery.specific_energy
    battery.max_energy=Ereq
    battery.max_power =Preq
    battery.current_energy=[battery.max_energy] #initialize list of current energy
    from SUAVE.Methods.Power.Battery.Variable_Mass import find_mass_gain_rate
    m_air       =-find_mass_gain_rate(battery,Ereq) #normally uses power as input to find mdot, but can use E to find m 
    m_water     =battery.find_water_mass(Ereq)
    #now add the electric motor weight
    motor_mass=ducted_fan.number_of_engines*SUAVE.Methods.Weights.Correlations.Propulsion.air_cooled_motor((Preq)*Units.watts/ducted_fan.number_of_engines)
    propulsion_mass=SUAVE.Methods.Weights.Correlations.Propulsion.integrated_propulsion(motor_mass/ducted_fan.number_of_engines,ducted_fan.number_of_engines)
    
    
    #more geometry sizing of ducted fan
    cruise_altitude= mission['climb_3'].altitude_end
    conditions = atmo.compute_values(cruise_altitude)
    sizing_segment = SUAVE.Components.Propulsors.Segments.Segment()
    sizing_segment.M   = mission['cruise'].air_speed/conditions.speed_of_sound       
    sizing_segment.alt = cruise_altitude
    sizing_segment.T   = conditions.temperature        
    sizing_segment.p   = conditions.pressure
    ducted_fan.design_thrust =design_thrust
    ducted_fan.mass_properties.mass=propulsion_mass
    ducted_fan.engine_sizing_ducted_fan(sizing_segment) 
    
    #compute overall mass properties
    breakdown = analyses.configs.base.weights.evaluate()
    breakdown.battery=battery.mass_properties.mass
    breakdown.water =m_water
    breakdown.air   =m_air
    base.mass_properties.breakdown=breakdown
    #print breakdown
    m_fuel=0.
    
    base.mass_properties.operating_empty     = breakdown.empty 
    
    #weight =SUAVE.Methods.Weights.Correlations.Tube_Wing.empty_custom_eng(vehicle, ducted_fan)
    m_full=breakdown.empty+battery.mass_properties.mass+breakdown.payload+m_water
    m_end=m_full+m_air
    base.mass_properties.takeoff                 = m_full
    base.store_diff()
   
    # Update all configs with new base data    
    for config in configs:
        config.pull_base()

    
    ##############################################################################
    # ------------------------------------------------------------------
    #   Define Landing Configurations
    # ------------------------------------------------------------------
 
    landing_config=configs.landing
    
    landing_config.wings['main_wing'].flaps.angle =  50. * Units.deg
    landing_config.wings['main_wing'].slats.angle  = 25. * Units.deg
    landing_config.mass_properties.landing = m_end
    landing_config.store_diff()
        
    
    #analyses.weights=configs.base.mass_properties.takeoff
    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ---------------------------------
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

def evaluate_mission(configs,mission):
    
    # ------------------------------------------------------------------    
    #   Run Mission
    # ------------------------------------------------------------------
    results = mission.evaluate()
    #determine energy characteristiscs
    e_current_min=1E20
    Pmax=0.
    for i in range(len(results.segments)):
            if np.min(results.segments[i].conditions.propulsion.battery_energy[:,0])<e_current_min:
                e_current_min=np.min(results.segments[i].conditions.propulsion.battery_energy[:,0])
            if np.max(np.abs(results.segments[i].conditions.propulsion.battery_draw[:,0]))>Pmax:
                Pmax=np.max(np.abs(results.segments[i].conditions.propulsion.battery_draw[:,0]))       
    results.e_total=results.segments[0].conditions.propulsion.battery_energy[0,0]-e_current_min
    results.Pmax=Pmax
    print 'e_current_min=',e_current_min
    print "e_total=", results.e_total
    print "Pmax=", Pmax
  
    return results
'''
def missions(interface):
    
    missions = interface.analyses.missions
    
    results = missions.evaluate()
    
    return results            
'''    
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
    LFL = estimate_landing_field_length(landing_config, analyses,airport)
    
    # pack
    field_length = SUAVE.Core.Data()
    field_length.takeoff = TOFL[0][0]
    field_length.landing = LFL[0][0]

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
    inputs                = interface.inputs
    vehicle               = interface.configs.base    
    results               = interface.results
    mission_profile       = results.sizing_loop   
  
    # Weights
    operating_empty   = vehicle.mass_properties.operating_empty
    payload           = vehicle.mass_properties.payload    
          
    # pack
    summary = SUAVE.Core.Results()
    
    # TOFL for MTOW @ SL, ISA
    
    summary.total_range           =mission_profile.segments[-1].conditions.frames.inertial.position_vector[-1,0]
    summary.GLW                   =mission_profile.segments[-1].conditions.weights.total_mass[-1,0] 
    summary.takeoff_field_length  =results.field_length.takeoff
    summary.landing_field_length  =results.field_length.landing
    summary.climb_alt_constr     =inputs.climb_alt_fraction_1-inputs.climb_alt_fraction_2
    # MZFW margin calculation
    #summary.max_zero_fuel_margin  = max_zero_fuel - (operating_empty + payload)
    # fuel margin calculation
    
    # Print outs   
    printme = Data()
    printme.weight_empty = operating_empty
    printme.total_range  =summary.total_range/Units.nautical_miles
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
    fid.write('{:18.10f} ; {:18.10f} ; {:18.10f} ; {:18.10f} ; {:18.10f} ; '.format( \
        inputs.aspect_ratio,
        inputs.reference_area,
        #inputs.sweep,
        inputs.wing_thickness,
        #operating_empty ,
        summary.takeoff_field_length , 
        summary.landing_field_length
    ))
    fid.write(datetime.datetime.now().strftime("%I:%M:%S"))
    fid.write('\n')

##    print interface.configs.takeoff.maximum_lift_coefficient
    
    return summary
def sizing_loop(interface):
    m_guess=11489.
    Ereq_guess=8080653526
    Preq_guess=1100363.636


    Ereq=[Ereq_guess]
    mass=[m_guess]
 
    tol=.01
    dE=1.
    dm=1.
    configs=interface.configs
    analyses=interface.analyses
    mission=analyses.missions.base
    max_iter=10
    j=0
    while abs(dm)>tol or abs(dE)>tol: #sizing loop for the vehicle
        Ereq_guess=Ereq[j]
        m_guess=mass[j]
        simple_sizing(interface, Ereq_guess, Preq_guess);
        battery=configs.base.energy_network['battery']
        configs.cruise.energy_network['battery']=battery #make it so all configs handle the exact same battery object
        configs.takeoff.energy_network['battery']=battery
        configs.landing.energy_network['battery']=battery
        #initialize battery in mission
        mission.segments[0].battery_energy=battery.max_energy
       
        configs.finalize()
        analyses.finalize()
        results = evaluate_mission(configs,mission)
       
        mass.append(results.segments[-1].conditions.weights.total_mass[-1,0] )
        Ereq.append(results.e_total)
        Preq_guess=results.Pmax
        dm=(mass[j+1]-mass[j])/mass[j]
        dE=(Ereq[j+1]-Ereq[j])/Ereq[j]
        #display convergence of aircraft
        print 'mass=', mass[j+1]
        print 'takeoff_mass=',results.segments[0].conditions.weights.total_mass[0,0]
        print 'dm=', dm
        print 'dE=', dE
        print 'Ereq_guess=', Ereq_guess 
        print 'Preq=', results.Pmax
        j=j+1
        
        if j>max_iter:
            print "maximum number of iterations exceeded"
            break
     
    return results
	
def post_process(interface):
	
    mission=interface.analyses.missions.base
    configs=interface.configs
    results=interface.results.sizing_loop
    battery=configs.base.propulsors.network['battery']
    #battery = vehicle.energy.Storages['Battery']
    #battery_lis = vehicle.Energy.Storages['Battery_Li_S']
    # ------------------------------------------------------------------    
    #   Thrust Angle
    # ------------------------------------------------------------------
    '''
    title = "Thrust Angle History"
    plt.figure(0)
    for i in range(len(results.segments)):
        plt.plot(results.segments[i].t/60,np.degrees(results.segments[i].gamma),'bo-')
    plt.xlabel('Time (mins)'); plt.ylabel('Thrust Angle (deg)'); plt.title(title)
    plt.grid(True)
    '''
    # ------------------------------------------------------------------    
    #   Throttle
    # ------------------------------------------------------------------
    plt.figure("Throttle History")
    axes = plt.gca()
    for i in range(len(results.segments)):
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        eta  = results.segments[i].conditions.propulsion.throttle[:,0]
        axes.plot(time, eta, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Throttle')
    axes.grid(True)

    # ------------------------------------------------------------------    
    #   Angle of Attack
    # ------------------------------------------------------------------
    plt.figure("Angle of Attack History")
    axes = plt.gca()    
    for i in range(len(results.segments)):     
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        aoa = results.segments[i].conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        axes.plot(time, aoa, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Angle of Attack (deg)')
    axes.grid(True)        
    
    # ------------------------------------------------------------------    
    #   Vehicle Mass
    # ------------------------------------------------------------------
    plt.figure("Vehicle Mass")
    axes = plt.gca()
    for i in range(len(results.segments)):
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        mass = results.segments[i].conditions.weights.total_mass[:,0]
        axes.plot(time, mass, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Vehicle Mass (kg)')
    axes.grid(True)
    
    
    
    # ------------------------------------------------------------------    
    #   Mass Gain Rate
    # ------------------------------------------------------------------
    plt.figure("Mass Accumulation Rate")
    axes = plt.gca()    
    for i in range(len(results.segments)):     
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        mdot = results.segments[i].conditions.propulsion.fuel_mass_rate[:,0]
        axes.plot(time, -mdot, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Mass Accumulation Rate(kg/s)')
    axes.grid(True)    
    
    
 

    
    
    # ------------------------------------------------------------------    
    #   Altitude
    # ------------------------------------------------------------------
    plt.figure("Altitude")
    axes = plt.gca()    
    for i in range(len(results.segments)):   
        time     = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        altitude = results.segments[i].conditions.freestream.altitude[:,0] /Units.km
        axes.plot(time, altitude, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Altitude (km)')
    axes.grid(True)
    
    
    # ------------------------------------------------------------------    
    #   Horizontal Distance
    # ------------------------------------------------------------------
    '''
    title = "Ground Distance"
    plt.figure(title)
    for i in range(len(results.segments)):
        plt.plot(results.segments[i].t/60,results.segments[i].conditions.frames.inertial.position_vector[:,0])
    plt.xlabel('Time (mins)'); plt.ylabel('ground distance(km)'); plt.title(title)
    plt.grid(True)
    '''
    
    # ------------------------------------------------------------------    
    #   Energy
    # ------------------------------------------------------------------
    
    title = "Energy and Power"
    fig=plt.figure(title)
    #print results.segments[0].Ecurrent[0]
    for segment in results.segments:
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        axes = fig.add_subplot(2,1,1)
        axes.plot(time, segment.conditions.propulsion.battery_energy/battery.max_energy,'bo-')
        #print 'E=',segment.conditions.propulsion.battery_energy
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('State of Charge of the Battery')
        axes.grid(True)
        axes = fig.add_subplot(2,1,2)
        axes.plot(time, -segment.conditions.propulsion.battery_draw,'bo-')
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Electric Power (Watts)')
        axes.grid(True)
    """
    for i in range(len(results.segments)):
        plt.plot(results.segments[i].t/60, results.segments[i].Ecurrent_lis/battery_lis.TotalEnergy,'ko-')
      
        #results.segments[i].Ecurrent_lis/battery_lis.TotalEnergy
    """
    # ------------------------------------------------------------------    
    #   Aerodynamics
    # ------------------------------------------------------------------
    fig = plt.figure("Aerodynamic Forces")
    for segment in results.segments.values():
        
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Lift   = -segment.conditions.frames.wind.lift_force_vector[:,2]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]

        axes = fig.add_subplot(3,1,1)
        axes.plot( time , Lift , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Lift (N)')
        axes.grid(True)
        
        axes = fig.add_subplot(3,1,2)
        axes.plot( time , Drag , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Drag (N)')
        axes.grid(True)
        
        axes = fig.add_subplot(3,1,3)
        axes.plot( time , Thrust , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Thrust (N)')
        axes.grid(True)
        
    # ------------------------------------------------------------------    
    #   Aerodynamics 2
    # ------------------------------------------------------------------
    fig = plt.figure("Aerodynamic Coefficients")
    for segment in results.segments.values():
        
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        CLift  = segment.conditions.aerodynamics.lift_coefficient[:,0]
        CDrag  = segment.conditions.aerodynamics.drag_coefficient[:,0]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]

        axes = fig.add_subplot(4,1,1)
        axes.plot( time , CLift , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('CL')
        axes.grid(True)
        
        axes = fig.add_subplot(4,1,2)
        axes.plot( time , CDrag , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('CD')
        axes.grid(True)
        
        axes = fig.add_subplot(4,1,3)
        axes.plot( time , Drag   , 'bo-' )
        axes.plot( time , Thrust , 'ro-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Drag and Thrust (N)')
        axes.grid(True)
        
        axes = fig.add_subplot(4,1,4)
        axes.plot( time , CLift/CDrag , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('CL/CD')
        axes.grid(True)
    plt.show()
    raw_input('Press Enter To Quit')
    return     
if __name__ == '__main__':
    main()
    
    
