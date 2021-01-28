# BWB.py
# 
# Created:  Aug 2014, E. Botero
# Modified: Jan 2019, W. Maier

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
import numpy as np
import pylab as plt
import copy, time

from SUAVE.Core import (
Data, Container
)

from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Propulsion import compute_turbofan_geometry
# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup():

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'BWB'    

    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff    = 79015.8   # kg
    vehicle.mass_properties.takeoff        = 79015.8   # kg
    vehicle.mass_properties.max_zero_fuel  = 0.9 * vehicle.mass_properties.max_takeoff
    vehicle.mass_properties.cargo          = 10000.  * Units.kilogram   

    # envelope properties
    vehicle.envelope.ultimate_load         = 2.5
    vehicle.envelope.limit_load            = 1.5
                                           
    # basic parameters                     
    vehicle.reference_area                 = 125.0     
    vehicle.passengers                     = 170
    vehicle.systems.control                = "fully powered" 
    vehicle.systems.accessories            = "sst"


    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    
    
    wing.aspect_ratio            = 5.86 #12.
    wing.sweeps.quarter_chord    = 15. * Units.deg
    wing.thickness_to_chord      = 0.14
    wing.taper                   = 0.1
    wing.dihedral                = 3.0 * Units.degrees
    wing.spans.projected         = 39.0
    wing.chords.root             = 17.0
    wing.chords.tip              = 1.0
    wing.chords.mean_aerodynamic = (2.0/3.0)*(wing.chords.root + wing.chords.root -(wing.chords.root*wing.chords.root)/(wing.chords.root+wing.chords.root))
    wing.areas.reference         = 259.4
    wing.twists.root             = 1.0 * Units.degrees
    wing.twists.tip              = -4.0 * Units.degrees
    wing.origin                  = [3.,0.,-.25]
    wing.aerodynamic_center      = [3,0,-.25]
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True
    wing.dynamic_pressure_ratio  = 1.0
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'fuselage_edge'
    segment.percent_span_location = 7.0/wing.spans.projected
    segment.twist                 = -2. * Units.deg
    segment.root_chord_percent    = 0.88 
    segment.dihedral_outboard     = 10.0 * Units.deg
    segment.sweeps.quarter_chord  = 40.0*Units.deg
    segment.thickness_to_chord    = 0.18
        
    wing.Segments.append(segment)
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'Outboard'
    segment.percent_span_location = 0.3
    segment.twist                 = 0.0 * Units.deg
    segment.root_chord_percent    = 0.35
    segment.dihedral_outboard     = 4.0 * Units.deg
    segment.sweeps.quarter_chord  = 20.0 * Units.deg
    segment.thickness_to_chord    = 0.1
    
    wing.Segments.append(segment)    

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------

    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag                                = 'fuselage_bwb'
    fuselage.fineness.nose                      = 0.65
    fuselage.fineness.tail                      = 0.5
    fuselage.lengths.nose                       = 4.0
    fuselage.lengths.tail                       = 4.0
    fuselage.lengths.cabin                      = 12.0
    fuselage.lengths.total                      = 22.0
    fuselage.lengths.fore_space                 = 1.0
    fuselage.lengths.aft_space                  = 1.0  
    fuselage.width                              = 8.0
    fuselage.heights.maximum                    = 3.8
    fuselage.heights.at_quarter_length          = 3.7
    fuselage.heights.at_three_quarters_length   = 2.5
    fuselage.heights.at_wing_root_quarter_chord = 4.0
    fuselage.areas.side_projected               = 100.
    fuselage.areas.wetted                       = 400.
    fuselage.areas.front_projected              = 40.
    
    R = (fuselage.heights.maximum-fuselage.width)/(fuselage.heights.maximum-fuselage.width)
    fuselage.effective_diameter    = (fuselage.width/2 + fuselage.heights.maximum/2.)*(64.-3.*R**4.)/(64.-16.*R**2.)
    fuselage.differential_pressure = 5.0e4 * Units.pascal # Maximum differential pressure

    # add to vehicle
    vehicle.append_component(fuselage)

    # ------------------------------------------------------------------
    #   Turbofan Network
    # ------------------------------------------------------------------    

    #instantiate the gas turbine network
    turbofan = SUAVE.Components.Energy.Networks.Turbofan()
    turbofan.tag = 'turbofan'

    # setup
    turbofan.number_of_engines = 2.0
    turbofan.sealevel_static_thrust = 2* 10000 *Units.N
   
    # add  gas turbine network gt_engine to the vehicle 
    vehicle.append_component(turbofan)      


    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------

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

    # done!
    return configs