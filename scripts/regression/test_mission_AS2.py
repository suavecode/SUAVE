# full_setup.py
# 
# Created:  SUave Team, Aug 2014
# Modified: 

""" setup file for a mission with a 737
"""


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

# the analysis functions
from the_aircraft_function import the_aircraft_function
from plot_mission import plot_mission


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    
    # define the problem
    vehicle, mission = full_setup()
    
    # run the problem
    results = the_aircraft_function(vehicle,mission)
    
    # plot the new results
    plot_mission(vehicle,mission,results,'bo-')    
    
    # load older results
    #save_results(results)
    old_results = load_results()
    
    # plt the old results
    plot_mission(vehicle,mission,old_results,'k-')
    
    # check the results
    check_results(results,old_results)
    
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():

    vehicle = vehicle_setup()
    mission = mission_setup(vehicle)
    
    return vehicle, mission


# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Aerion AS2'

    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff          = 52163    # kg
    vehicle.mass_properties.operating_empty      = 22500    # kg
    vehicle.mass_properties.takeoff              = 52163    # kg
    vehicle.mass_properties.max_zero_fuel        = 0.9 * vehicle.mass_properties.max_takeoff 
    vehicle.mass_properties.cargo                = 1000.  * Units.kilogram

    vehicle.mass_properties.center_of_gravity         = [26.3 * Units.feet, 0, 0]
    vehicle.mass_properties.moments_of_inertia.tensor = [[10 ** 5, 0, 0],[0, 10 ** 6, 0,],[0,0, 10 ** 7]] # Not Correct

    # envelope properties
    vehicle.envelope.ultimate_load = 3.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area       = 124.862       
    vehicle.passengers           = 8
    vehicle.systems.control      = "fully powered" 
    vehicle.systems.accessories  = "long range"
    
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'main_wing'
    wing.areas.reference = 125.4    #
    wing.aspect_ratio    = 3.63     #
    wing.spans.projected = 21.0     #
    wing.sweep           = 0 * Units.deg
    wing.symmetric       = True
    wing.thickness_to_chord = 0.03
    wing.taper           = 0.7

    # size the wing planform
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    # size the wing planform ----------------------------------
    # These can be determined by the wing sizing function
    # Note that the wing sizing function will overwrite span
    wing.chords.root  = 12.9
    wing.chords.tip   = 1.0
    wing.areas.wetted = wing.areas.reference*2.0 
    # The span that would normally be overwritten here doesn't match
    # ---------------------------------------------------------    
    
    wing.chords.mean_aerodynamic = 7.0
    wing.areas.exposed = 0.8*wing.areas.wetted
    wing.areas.affected = 0.6*wing.areas.wetted
    wing.span_efficiency = 0.74
    wing.twists.root = 0.0*Units.degrees
    wing.twists.tip  = 2.0*Units.degrees
    wing.origin             = [20,0,0]
    wing.aerodynamic_center = [5,0,0]     
    wing.vertical = False
    wing.eta = 1.0
    
    wing.high_lift    = False                 #
    wing.high_mach    = True
    wing.vortex_lift  = False
    wing.transition_x_upper = 0.9
    wing.transition_x_lower = 0.9
    
    #print wing
    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------        
    #   Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'horizontal_stabilizer'
    
    
    wing.areas.reference = 24.5     #
    wing.aspect_ratio    = 2.0      #
    wing.spans.projected = 7.0      #
    wing.sweep           = 0 * Units.deg
    wing.symmetric       = True
    wing.thickness_to_chord = 0.03
    wing.taper           = 0.5
    wing.origin          = [46,0,0]
    wing.aerodynamic_center = [2,0,0]
    wing.vertical   = False 
    wing.eta         = 0.9

    # size the wing planform
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chords.mean_aerodynamic = 3.0
    wing.areas.exposed = 0.8*wing.areas.wetted
    wing.areas.affected = 0.6*wing.areas.wetted
    wing.span_efficiency = 0.74
    wing.twists.root = 0.0*Units.degrees
    wing.twists.tip  = 2.0*Units.degrees
    wing.vertical = False
    
    wing.high_lift    = False                 #
    wing.high_mach    = True
    wing.vortex_lift  = False
    wing.transition_x_upper = 0.9
    wing.transition_x_lower = 0.9
    
    #print wing
    # add to vehicle
    vehicle.append_component(wing)    
    
    
    # ------------------------------------------------------------------
    #   Vertcal Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'vertical_stabilizer'    
    
    wing.areas.reference = 33.91    #
    wing.aspect_ratio    = 1.3      #
    wing.spans.projected = 3.5      #
    wing.sweep           = 45 * Units.deg
    wing.symmetric       = False
    wing.thickness_to_chord = 0.04
    wing.taper           = 0.5

    # size the wing planform
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chords.mean_aerodynamic = 4.2
    wing.areas.exposed = 1.0*wing.areas.wetted
    wing.areas.affected = 0.0*wing.areas.wetted
    wing.span_efficiency = 0.9
    wing.twists.root = 0.0*Units.degrees
    wing.twists.tip  = 0.0*Units.degrees
    wing.origin          = [40,0,0]
    wing.aerodynamic_center = [2,0,0]    

    wing.vertical   = True 
    wing.t_tail     = False
    wing.eta         = 1.0

    wing.high_lift    = False                 #
    wing.high_mach    = True
    wing.vortex_lift  = False
    wing.vertical = True
    wing.transition_x_upper = 0.9
    wing.transition_x_lower = 0.9    
    
        
    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    
    fuselage.number_coach_seats = 8
    fuselage.seats_abreast = 2
    fuselage.seat_pitch = 1
    fuselage.fineness.nose = 4.0 # These finenesses are smaller than the real value due to limitations of existing functions

    fuselage.fineness.tail = 4.0
    fuselage.lengths.fore_space = 16.3
    fuselage.lengths.aft_space  = 16.3
    fuselage.width = 2.35
    fuselage.heights.maximum            = 2.55    #
    fuselage.areas.side_projected       = 4.* 59.8 #  Not correct
    fuselage.heights.at_quarter_length  = 4. # Not correct
    fuselage.heights.at_three_quarters_length   = 4. # Not correct
    fuselage.heights.at_wing_root_quarter_chord = 4. # Not correct
    fuselage.differential_pressure = 10**5   * Units.pascal    # Maximum differential pressure
    
    # size fuselage planform
    # A function exists to do this
    fuselage.lengths.nose  = 12.0
    fuselage.lengths.tail  = 25.5
    fuselage.lengths.cabin = 11.5
    fuselage.lengths.total = 49.0


    fuselage.areas.wetted  = 615.0
    fuselage.areas.front_projected = 5.1
    fuselage.effective_diameter    = 2.4
    
    # add to vehicle
    vehicle.append_component(fuselage)
    
    
    # ------------------------------------------------------------------
    #  Turbojet
    # ------------------------------------------------------------------    
    
    #turbojet = SUAVE.Components.Propulsors.Turbojet2PASS()
    turbojet = SUAVE.Components.Propulsors.Turbojet_SupersonicPASS()
    turbojet.tag = 'turbo_fan'
    
    turbojet.propellant = SUAVE.Attributes.Propellants.Jet_A1()
    
    turbojet.analysis_type                 = '1D'     #
    turbojet.diffuser_pressure_ratio       = 1.0      # 1.0 either not known or not relevant
    turbojet.fan_pressure_ratio            = 1.0      #
    turbojet.fan_nozzle_pressure_ratio     = 1.0      #
    turbojet.lpc_pressure_ratio            = 5.0      #
    turbojet.hpc_pressure_ratio            = 10.0     #
    turbojet.burner_pressure_ratio         = 1.0      #
    turbojet.turbine_nozzle_pressure_ratio = 1.0      #
    turbojet.Tt4                           = 1500.0   #
    turbojet.thrust.design                 = 15000.0 * Units.lb  # 31350 lbs
    turbojet.number_of_engines             = 3.0      #
    turbojet.engine_length                 = 8.0      # meters - includes 3.4m inlet
    turbojet.lengths = Data()
    turbojet.lengths.engine_total               = 8.0
    
    # turbojet sizing conditions
    sizing_segment = SUAVE.Components.Propulsors.Segments.Segment()
    
    # Note: Sizing designed to give roughly nominal values - M = 2.02 is not achieved at 35,000 ft
    
    sizing_segment.M   = 2.02                    #
    sizing_segment.alt = 35000 * Units.ft        #
    sizing_segment.T   = 218.0                   #
    sizing_segment.p   = 0.239*10**5             #
    
    # size the turbojet
    turbojet.engine_sizing_1d(sizing_segment) 
    turbojet.nacelle_dia                        = 0.5
    
    # add to vehicle
    vehicle.append_component(turbojet)


    # ------------------------------------------------------------------
    #   Simple Aerodynamics Model
    # ------------------------------------------------------------------ 
    
    aerodynamics = SUAVE.Attributes.Aerodynamics.Supersonic_Zero()
    aerodynamics.initialize(vehicle)

    # build stability model
    stability = SUAVE.Attributes.Flight_Dynamics.Fidelity_Zero()
    stability.initialize(vehicle)
    aerodynamics.stability = stability
    vehicle.aerodynamics_model = aerodynamics
    
    # ------------------------------------------------------------------
    #   Simple Propulsion Model
    # ------------------------------------------------------------------     
    
    vehicle.propulsion_model = vehicle.propulsors

    # ------------------------------------------------------------------
    #   Define Configurations
    # ------------------------------------------------------------------

    # --- Takeoff Configuration ---
    config = vehicle.new_configuration("takeoff")
    # this configuration is derived from the baseline vehicle

    # --- Cruise Configuration ---
    config = vehicle.new_configuration("cruise")
    # this configuration is derived from vehicle.configs.takeoff

    # --- Takeoff Configuration ---
    takeoff_config = vehicle.configs.takeoff
    takeoff_config.wings['main_wing'].flaps_angle =  20. * Units.deg
    takeoff_config.wings['main_wing'].slats_angle  = 25. * Units.deg
    # V2_V2_ratio may be informed by user. If not, use default value (1.2)
    takeoff_config.V2_VS_ratio = 1.21
    # CLmax for a given configuration may be informed by user. If not, is calculated using correlations
    takeoff_config.maximum_lift_coefficient = 2.
    #takeoff_config.max_lift_coefficient_factor = 1.0

    # --- Landing Configuration ---
    landing_config = vehicle.new_configuration("landing")
    landing_config.wings['main_wing'].flaps_angle =  30. * Units.deg
    landing_config.wings['main_wing'].slats_angle  = 25. * Units.deg
    # Vref_V2_ratio may be informed by user. If not, use default value (1.23)
    landing_config.Vref_VS_ratio = 1.23
    # CLmax for a given configuration may be informed by user
    landing_config.maximum_lift_coefficient = 2.
    #landing_config.max_lift_coefficient_factor = 1.0
    landing_config.mass_properties.landing = 0.85 * vehicle.mass_properties.takeoff
    

    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------
    
    return vehicle

#: def define_vehicle()


# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------
def mission_setup(vehicle):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Missions.Mission()
    mission.tag = 'The Test Mission'

    # atmospheric model
    planet = SUAVE.Attributes.Planets.Earth()
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    
    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    
    mission.airport = airport
    

    
    # ------------------------------------------------------------------
    #   Sixth Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Analyses.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb - 6"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        

    segment.altitude_start = 0.0    * Units.km
    segment.altitude_end = 3.05     * Units.km
    segment.air_speed    = 128.6    * Units['m/s']
    segment.climb_rate   = 4000    * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment)     
    
    # ------------------------------------------------------------------
    #   Seventh Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Analyses.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb - 7"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        

    segment.altitude_end = 4.57     * Units.km
    segment.air_speed    = 205.8    * Units['m/s']
    segment.climb_rate   = 1000    * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment) 
    
    # ------------------------------------------------------------------
    #   Eighth Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Analyses.Missions.Segments.Climb.Linear_Mach_Constant_Rate()
    segment.tag = "Climb - 8"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        
    
    segment.altitude_end = 9.77 * Units.km # 
    segment.mach_number_start = 0.64
    segment.mach_number_end  = 1.0 
    segment.climb_rate   = 1000    * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment)  
    
    # ------------------------------------------------------------------
    #   Eighth Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Analyses.Missions.Segments.Climb.Linear_Mach_Constant_Rate()
    segment.tag = "Climb - 9"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        
    
    segment.altitude_end = 15.54 * Units.km # 51000 ft
    segment.mach_number_start = 1.0
    segment.mach_number_end  = 1.4
    segment.climb_rate   = 1000    * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment)   
    
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Analyses.Missions.Segments.Cruise.Constant_Mach_Constant_Altitude()
    segment.tag = "Cruise"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet        
    
    segment.altitude   = 15.54  * Units.km     # Optional
    segment.mach       = 1.4
    segment.distance   = 4000.0 * Units.nmi
        
    mission.append_segment(segment)

    # ------------------------------------------------------------------    
    #   First Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Analyses.Missions.Segments.Descent.Linear_Mach_Constant_Rate()
    segment.tag = "Descent - 1"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet   
    
    segment.altitude_end = 6.8  * Units.km
    segment.mach_number_start = 1.4
    segment.mach_number_end = 1.0
    segment.descent_rate = 5.0   * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------    
    #   Second Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Analyses.Missions.Segments.Descent.Linear_Mach_Constant_Rate()
    segment.tag = "Descent - 2"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet   
    
    segment.altitude_end = 3.0  * Units.km
    segment.mach_number_start = 1.0
    segment.mach_number_end = 0.65
    segment.descent_rate = 5.0   * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
      
    
    # ------------------------------------------------------------------    
    #   Second Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Analyses.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "Descent - 5"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet    
    
    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 130.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']

    # append to mission
    mission.append_segment(segment)       

    
    # ------------------------------------------------------------------    
    #   Mission definition complete    
    # ------------------------------------------------------------------
    
    return mission



def check_results(new_results,old_results):
    
    # check segment values
    check_list = [
        'mission_profile.segments.Cruise.conditions.aerodynamics.angle_of_attack',
        'mission_profile.segments.Cruise.conditions.aerodynamics.drag_coefficient',
        'mission_profile.segments.Cruise.conditions.aerodynamics.lift_coefficient',
        'mission_profile.segments.Cruise.conditions.aerodynamics.cm_alpha',
        'mission_profile.segments.Cruise.conditions.aerodynamics.cn_beta',
        'mission_profile.segments.Cruise.conditions.propulsion.throttle',
        'mission_profile.segments.Cruise.conditions.propulsion.fuel_mass_rate',
    ]
    
    # gets a key recursively from a '.' string
    def get_key(data,keys):
        if isinstance(keys,str):
            keys = keys.split('.')
        k = keys.pop(0)
        if keys:
            return get_key(data[k],keys)
        else:
            return data[k]    
    
    # do the check
    for k in check_list:
        print k
        
        old_val = np.max( get_key(old_results,k) )
        new_val = np.max( get_key(new_results,k) )
        err = (new_val-old_val)/old_val
        print 'Error at Max:' , err
        assert np.abs(err) < 1e-6 , 'Max Check Failed : %s' % k
        
        old_val = np.min( get_key(old_results,k) )
        new_val = np.min( get_key(new_results,k) )
        err = (new_val-old_val)/old_val
        print 'Error at Min:' , err
        assert np.abs(err) < 1e-6 , 'Min Check Failed : %s' % k        
        
        print ''
    
    # check high level outputs
    def check_vals(a,b):
        if isinstance(a,Data):
            for k in a.keys():
                err = check_vals(a[k],b[k])
                if err is None: continue
                print 'outputs' , k
                print 'Error:' , err
                print ''
                assert np.abs(err) < 1e-6 , 'Outputs Check Failed : %s' % k  
        else:
            return (a-b)/a

    # do the check
    check_vals(old_results.output,new_results.output)
    
    return

    
def load_results():
    return SUAVE.Plugins.VyPy.data.load('results_mission_AS2.pkl')
    
def save_results(results):
    SUAVE.Plugins.VyPy.data.save(results,'results_mission_AS2.pkl')
    
if __name__ == '__main__': 
    main()    
    plt.show()
        