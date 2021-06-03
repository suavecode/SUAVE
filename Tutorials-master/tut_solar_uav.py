# tut_solar_UAV.py
# 
# Created:  Jul 2014, E. Botero
# Modified: Aug 2017, E. Botero

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units, Data

import numpy as np
import pylab as plt
import time

from SUAVE.Plots.Mission_Plots import *
from SUAVE.Components.Energy.Networks.Solar import Solar
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    
    # build the vehicle, configs, and analyses
    configs, analyses = full_setup()
    
    configs.finalize()
    analyses.finalize()    
    
    # weight analysis
    weights = analyses.configs.base.weights
    breakdown = weights.evaluate()          
    
    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()
    
    # plot results    
    plot_mission(results)
    
    return

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():
    
    # vehicle data
    vehicle  = vehicle_setup()
    configs  = configs_setup(vehicle)
    
    # vehicle analyses
    configs_analyses = analyses_setup(configs)
    
    # mission analyses
    mission  = mission_setup(configs_analyses,vehicle)
    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses
    
    return configs, analyses

# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Solar'
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    # mass properties
    vehicle.mass_properties.takeoff         = 250. * Units.kg
    vehicle.mass_properties.operating_empty = 250. * Units.kg
    vehicle.mass_properties.max_takeoff     = 250. * Units.kg 
    
    # basic parameters
    vehicle.reference_area                    = 80.       
    vehicle.envelope.ultimate_load            = 2.0
    vehicle.envelope.limit_load               = 1.5
    vehicle.envelope.maximum_dynamic_pressure = 0.5*1.225*(40.**2.) #Max q

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------   

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    
    wing.areas.reference         = vehicle.reference_area
    wing.spans.projected         = 40.0 * Units.meter
    wing.aspect_ratio            = (wing.spans.projected**2)/wing.areas.reference 
    wing.sweeps.quarter_chord    = 0.0 * Units.deg
    wing.symmetric               = True
    wing.thickness_to_chord      = 0.12
    wing.taper                   = 1.0
    wing.vertical                = False
    wing.high_lift               = True 
    wing.dynamic_pressure_ratio  = 1.0
    wing.chords.mean_aerodynamic = wing.areas.reference/wing.spans.projected
    wing.chords.root             = wing.areas.reference/wing.spans.projected
    wing.chords.tip              = wing.areas.reference/wing.spans.projected
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees
    wing.highlift                = False  
    wing.vertical                = False 
    wing.number_ribs             = 26.
    wing.number_end_ribs         = 2.
    wing.transition_x_upper      = 0.6
    wing.transition_x_lower      = 1.0
    wing.origin                  = [[3.0,0.0,0.0]] # meters
    wing.aerodynamic_center      = [1.0,0.0,0.0] # meters
    
    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Horizontal_Tail()
    wing.tag = 'horizontal_stabilizer'
    
    wing.aspect_ratio         = 20. 
    wing.sweeps.quarter_chord = 0 * Units.deg
    wing.thickness_to_chord   = 0.12
    wing.taper                = 1.0
    wing.areas.reference      = vehicle.reference_area * .15
    wing.areas.wetted         = 2.0 * wing.areas.reference
    wing.areas.exposed        = 0.8 * wing.areas.wetted
    wing.areas.affected       = 0.6 * wing.areas.wetted       
    wing.spans.projected      = np.sqrt(wing.aspect_ratio*wing.areas.reference)
    wing.twists.root          = 0.0 * Units.degrees
    wing.twists.tip           = 0.0 * Units.degrees      
    
    wing.vertical                = False 
    wing.symmetric               = True
    wing.dynamic_pressure_ratio  = 0.9      
    wing.number_ribs             = 5.0
    wing.chords.root             = wing.areas.reference/wing.spans.projected
    wing.chords.tip              = wing.areas.reference/wing.spans.projected
    wing.chords.mean_aerodynamic = wing.areas.reference/wing.spans.projected  
    wing.origin                  = [[10.,0.0,0.0]] # meters
    wing.aerodynamic_center      = [0.5,0.0,0.0] # meters
  
    # add to vehicle
    vehicle.append_component(wing)    
    
    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Vertical_Tail()
    wing.tag = 'vertical_stabilizer'    
    
    
    wing.aspect_ratio         = 20.       
    wing.sweeps.quarter_chord = 0 * Units.deg
    wing.thickness_to_chord   = 0.12
    wing.taper                = 1.0
    wing.areas.reference      = vehicle.reference_area * 0.1
    wing.spans.projected      = np.sqrt(wing.aspect_ratio*wing.areas.reference)

    wing.chords.root             = wing.areas.reference/wing.spans.projected
    wing.chords.tip              = wing.areas.reference/wing.spans.projected
    wing.chords.mean_aerodynamic = wing.areas.reference/wing.spans.projected 
    wing.areas.wetted            = 2.0 * wing.areas.reference
    wing.areas.exposed           = 0.8 * wing.areas.wetted
    wing.areas.affected          = 0.6 * wing.areas.wetted    
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees  
    wing.origin                  = [[10.,0.0,0.0]] # meters
    wing.aerodynamic_center      = [0.5,0.0,0.0] # meters
    wing.symmetric               = True          
    wing.vertical                = True 
    wing.t_tail                  = False
    wing.dynamic_pressure_ratio  = 1.0
    wing.number_ribs             = 5.
  
    # add to vehicle
    vehicle.append_component(wing)  
    
    #------------------------------------------------------------------
    # Propulsor
    #------------------------------------------------------------------
    
    # build network
    net = Solar()
    net.number_of_engines = 1.
    net.nacelle_diameter  = 0.2 * Units.meters
    net.engine_length     = 0.01 * Units.meters
    net.areas             = Data()
    net.areas.wetted      = 0.01*(2*np.pi*0.01/2.)
    
    # Component 1 the Sun?
    sun = SUAVE.Components.Energy.Processes.Solar_Radiation()
    net.solar_flux = sun
    
    # Component 2 the solar panels
    panel = SUAVE.Components.Energy.Converters.Solar_Panel()
    panel.area                 = vehicle.reference_area * 0.9
    panel.efficiency           = 0.25
    panel.mass_properties.mass = panel.area*(0.60 * Units.kg)
    net.solar_panel            = panel
    
    # Component 3 the ESC
    esc = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc.efficiency = 0.95 # Gundlach for brushless motors
    net.esc        = esc
    
    # Component 5 the Propeller
    # Design the Propeller
    prop = SUAVE.Components.Energy.Converters.Propeller()
    prop.number_blades       = 2.0
    prop.freestream_velocity = 40.0 * Units['m/s']# freestream
    prop.angular_velocity    = 150. * Units['rpm']
    prop.tip_radius          = 4.25 * Units.meters
    prop.hub_radius          = 0.05 * Units.meters
    prop.design_Cl           = 0.7
    prop.design_altitude     = 14.0 * Units.km
    prop.design_thrust       = None
    prop.design_power        = 3500.0 * Units.watts
    prop                     = propeller_design(prop)
    
    net.propeller            = prop

    # Component 4 the Motor
    motor = SUAVE.Components.Energy.Converters.Motor()
    motor.resistance           = 0.008
    motor.no_load_current      = 4.5  * Units.ampere
    motor.speed_constant       = 120. * Units['rpm'] # RPM/volt converted to (rad/s)/volt    
    motor.propeller_radius     = prop.tip_radius
    motor.propeller_Cp         = prop.design_power_coefficient
    motor.gear_ratio           = 12. # Gear ratio
    motor.gearbox_efficiency   = .98 # Gear box efficiency
    motor.expected_current     = 160. # Expected current
    motor.mass_properties.mass = 2.0  * Units.kg
    net.motor                  = motor    
    
    # Component 6 the Payload
    payload = SUAVE.Components.Energy.Peripherals.Payload()
    payload.power_draw           = 50. * Units.watts 
    payload.mass_properties.mass = 5.0 * Units.kg
    net.payload                  = payload
    
    # Component 7 the Avionics
    avionics = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.power_draw = 50. * Units.watts
    net.avionics        = avionics      

    # Component 8 the Battery
    bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    bat.mass_properties.mass = 90.0 * Units.kg
    bat.specific_energy      = 600. * Units.Wh/Units.kg
    bat.resistance           = 0.05
    bat.max_voltage          = 45.0
    initialize_from_mass(bat,bat.mass_properties.mass)
    net.battery              = bat
   
    #Component 9 the system logic controller and MPPT
    logic = SUAVE.Components.Energy.Distributors.Solar_Logic()
    logic.system_voltage  = 40.0
    logic.MPPT_efficiency = 0.95
    net.solar_logic       = logic
    
    # add the solar network to the vehicle
    vehicle.append_component(net)  

    return vehicle

# ----------------------------------------------------------------------
#   Define the Configurations
# ---------------------------------------------------------------------

def configs_setup(vehicle):
    
    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------
    
    configs = SUAVE.Components.Configs.Config.Container()
    
    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)
    
    # ------------------------------------------------------------------
    #   Cruise Configuration
    # ------------------------------------------------------------------
    
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'cruise'
    
    configs.append(config)
    
    return configs


# ----------------------------------------------------------------------
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------

def analyses_setup(configs):
    
    analyses = SUAVE.Analyses.Analysis.Container()
    
    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config)
        analyses[tag] = analysis
    
    return analyses

def base_analysis(vehicle):

    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------     
    analyses = SUAVE.Analyses.Vehicle()
    
    # ------------------------------------------------------------------
    #  Basic Geometry Relations
    sizing = SUAVE.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)
    
    # ------------------------------------------------------------------
    #  Weights
    weights = SUAVE.Analyses.Weights.Weights_UAV()
    weights.settings.empty = \
        SUAVE.Methods.Weights.Correlations.Human_Powered.empty
    weights.vehicle = vehicle
    analyses.append(weights)
    
    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)
    
    # ------------------------------------------------------------------
    #  Energy
    energy = SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.propulsors #what is called throughout the mission (at every time step))
    analyses.append(energy)
    
    # ------------------------------------------------------------------
    #  Planet Analysis
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)
    
    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)   
    
    # done!
    return analyses    


# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------
def mission_setup(analyses,vehicle):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'The Test Mission'

    mission.atmosphere  = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    mission.planet      = SUAVE.Attributes.Planets.Earth()
    
    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments
    
    # base segment
    base_segment = Segments.Segment()   
    ones_row     = base_segment.state.ones_row
    base_segment.process.iterate.unknowns.network            = vehicle.propulsors.solar.unpack_unknowns
    base_segment.process.iterate.residuals.network           = vehicle.propulsors.solar.residuals    
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.state.unknowns.propeller_power_coefficient  = vehicle.propulsors.solar.propeller.design_power_coefficient  * ones_row(1)/15.
    base_segment.state.residuals.network                     = 0. * ones_row(1)      
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Analyses.Mission.Segments.Cruise.Constant_Mach_Constant_Altitude(base_segment)
    segment.tag = "cruise1"
    
    # connect vehicle configuration
    segment.analyses.extend(analyses.cruise)
    
    # segment attributes     
    segment.state.numerics.number_control_points = 64
    segment.start_time     = time.strptime("Tue, Jun 21 11:30:00  2020", "%a, %b %d %H:%M:%S %Y",)
    segment.altitude       = 15.0  * Units.km 
    segment.mach           = 0.12
    segment.distance       = 3050.0 * Units.km
    segment.battery_energy = vehicle.propulsors.solar.battery.max_energy*0.2 #Charge the battery to start
    segment.latitude       = 37.4300   # this defaults to degrees (do not use Units.degrees)
    segment.longitude      = -122.1700 # this defaults to degrees
    
    mission.append_segment(segment)    

    # ------------------------------------------------------------------    
    #   Mission definition complete    
    # ------------------------------------------------------------------
    
    return mission

def missions_setup(base_mission):

    # the mission container
    missions = SUAVE.Analyses.Mission.Mission.Container()
    
    # ------------------------------------------------------------------
    #   Base Mission
    # ------------------------------------------------------------------
    
    missions.base = base_mission
    
    # done!
    return missions  

# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def plot_mission(results):

    # Plot Flight Conditions 
    plot_flight_conditions(results) 
    
    # Plot Solar Conditions 
    plot_solar_flux(results)

    # Plot Aerodynamic Coefficients
    plot_aerodynamic_coefficients(results)  

    # Plot Aircraft Flight Speed
    plot_aircraft_velocities(results)

    # Plot Aircraft Electronics
    plot_electronic_conditions(results)

    # Plot Propeller Conditions 
    plot_propeller_conditions(results) 

    # Plot Electric Motor and Propeller Efficiencies 
    plot_eMotor_Prop_efficiencies(results)

    
    return 

if __name__ == '__main__':
    main()
    
    plt.show()