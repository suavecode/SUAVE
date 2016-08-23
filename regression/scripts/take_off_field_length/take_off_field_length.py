# test_take_off_field_length.py
#
# Created:  Tarik, Carlos, Celso, Jun 2014
# Modified: M. Vegh, Feb 2016

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave Imports
import SUAVE
from SUAVE.Core            import Data
from SUAVE.Core import Units
from SUAVE.Core import Units
from SUAVE.Methods.Performance.estimate_take_off_field_length import estimate_take_off_field_length
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform as size_planform
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Propulsion.compute_turbofan_geometry import compute_turbofan_geometry
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing

# package imports
import numpy as np
import pylab as plt

# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------
def main():

    # ----------------------------------------------------------------------
    #   Main
    # ----------------------------------------------------------------------    
    vehicle = vehicle_setup()
    configs=configs_setup(vehicle)
    # --- Takeoff Configuration ---
    configuration = configs.takeoff
    configuration.wings['main_wing'].flaps_angle =  20. * Units.deg
    configuration.wings['main_wing'].slats_angle  = 25. * Units.deg
    # V2_V2_ratio may be informed by user. If not, use default value (1.2)
    configuration.V2_VS_ratio = 1.21
   
    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.base = base_analysis(vehicle)

    # CLmax for a given configuration may be informed by user
    # configuration.maximum_lift_coefficient = 2.XX

    # --- Airport definition ---
    airport = SUAVE.Attributes.Airports.Airport()
    airport.tag = 'airport'
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere =  SUAVE.Analyses.Atmospheric.US_Standard_1976()
    
    w_vec = np.linspace(40000.,52000.,10)
    engines = (2,3,4)
    takeoff_field_length = np.zeros((len(w_vec),len(engines)))
    second_seg_clb_grad  = np.zeros((len(w_vec),len(engines)))
    
    compute_clb_grad = 1 # flag for Second segment climb estimation
    
    for id_eng,engine_number in enumerate(engines):
        
        configuration.propulsors.turbofan.number_of_engines = engine_number
        
        for id_w,weight in enumerate(w_vec):
            configuration.mass_properties.takeoff = weight
            takeoff_field_length[id_w,id_eng],second_seg_clb_grad[id_w,id_eng] = \
                    estimate_take_off_field_length(configuration,analyses,airport,compute_clb_grad)
    
   
    truth_TOFL = np.array([[ 1106.1894316 ,   720.57532937,   521.72582498],
                           [ 1169.44251482,   758.14235055,   548.54256171],
                           [ 1235.49201944,   797.22058262,   576.39554512],
                           [ 1304.40380631,   837.83307113,   605.29666019],
                           [ 1376.24593627,   880.00348867,   635.25804638],
                           [ 1451.08870497,   923.75614652,   666.29210347],
                           [ 1529.00467733,   969.11600619,   698.41149721],
                           [ 1610.06872186,  1016.10869073,   731.6291648 ],
                           [ 1694.3580448 ,  1064.76049592,   765.95832037],
                           [ 1781.95222403,  1115.09840143,   801.41246029]])
                             
    truth_clb_grad = np.array([[ 0.0663248  , 0.24446563 , 0.42260645],
                               [0.06064292 , 0.23261737 , 0.40459183],
                               [0.05530959 , 0.22151265 , 0.38771572],
                               [0.05029414 , 0.21108444 , 0.37187475],
                               [0.04556936 , 0.20127349 , 0.35697762],
                               [0.04111106 , 0.19202724 , 0.34294343],
                               [0.03689762 , 0.18329891 , 0.3297002 ],
                               [0.0329097  , 0.17504671 , 0.31718373],
                               [0.02912991 , 0.16723321 , 0.30533651],
                               [0.02554262 , 0.15982479 , 0.29410696]]   )
                               
                       
                               
    print 'second_seg_clb_grad=', second_seg_clb_grad
    TOFL_error = np.max(np.abs(truth_TOFL-takeoff_field_length))                           
    GRAD_error = np.max(np.abs(truth_clb_grad-second_seg_clb_grad))
    
    print 'Maximum Take OFF Field Length Error= %.4e' % TOFL_error
    print 'Second Segment Climb Gradient Error= %.4e' % GRAD_error    
    
    import pylab as plt
    title = "TOFL vs W"
    plt.figure(1); plt.hold
    plt.plot(w_vec,takeoff_field_length[:,0], 'k-', label = '2 Engines')
    plt.plot(w_vec,takeoff_field_length[:,1], 'r-', label = '3 Engines')
    plt.plot(w_vec,takeoff_field_length[:,2], 'b-', label = '4 Engines')

    plt.title(title); plt.grid(True)
    plt.plot(w_vec,truth_TOFL[:,0], 'k--o', label = '2 Engines [truth]')
    plt.plot(w_vec,truth_TOFL[:,1], 'r--o', label = '3 Engines [truth]')
    plt.plot(w_vec,truth_TOFL[:,2], 'b--o', label = '4 Engines [truth]')
    legend = plt.legend(loc='lower right', shadow = 'true')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Takeoff field length (m)')    
        
    
    title = "2nd Segment Climb Gradient vs W"
    plt.figure(2); plt.hold
    plt.plot(w_vec,second_seg_clb_grad[:,0], 'k-', label = '2 Engines')
    plt.plot(w_vec,second_seg_clb_grad[:,1], 'r-', label = '3 Engines')
    plt.plot(w_vec,second_seg_clb_grad[:,2], 'b-', label = '4 Engines')

    plt.title(title); plt.grid(True)
    plt.plot(w_vec,truth_clb_grad[:,0], 'k--o', label = '2 Engines [truth]')
    plt.plot(w_vec,truth_clb_grad[:,1], 'r--o', label = '3 Engines [truth]')
    plt.plot(w_vec,truth_clb_grad[:,2], 'b--o', label = '4 Engines [truth]')
    legend = plt.legend(loc='lower right', shadow = 'true')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Second Segment Climb Gradient (%)')    
    
    assert( TOFL_error   < 1e-5 )
    assert( GRAD_error   < 1e-5 )

    return 
    
    
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

    # ------------------------------------------------------------------
    #   Takeoff Configuration
    # ------------------------------------------------------------------
    
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'takeoff'
    
    config.wings['main_wing'].flaps.angle = 20. * Units.deg
    config.wings['main_wing'].slats.angle = 25. * Units.deg
    
    config.V2_VS_ratio = 1.21
    config.max_lift_coefficient_factor = 0.95
##    config.maximum_lift_coefficient = 2.
    
    configs.append(config)
    return configs   

def vehicle_setup():
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'EMBRAER E190AR'  
    
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff               = 51800.0   # kg
    vehicle.mass_properties.operating_empty           = 29100.0   # kg
    vehicle.mass_properties.takeoff                   = 51800.0   # kg
    ## vehicle.mass_properties.max_zero_fuel             = 0.9 * vehicle.mass_properties.max_takeoff 
    vehicle.mass_properties.cargo                     = 0.  * Units.kilogram   
    
    vehicle.mass_properties.center_of_gravity         = [60 * Units.feet, 0, 0]  # Not correct
    vehicle.mass_properties.moments_of_inertia.tensor = [[10 ** 5, 0, 0],[0, 10 ** 6, 0,],[0,0, 10 ** 7]] # Not Correct
    
    # envelope properties
    vehicle.envelope.ultimate_load = 3.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area         = 92.0       
    vehicle.passengers             = 114
    vehicle.systems.control        = "fully powered" 
    vehicle.systems.accessories    = "medium range"
    
    
    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'

    wing.aspect_ratio            = 8.4
    wing.sweeps.quarter_chord    = 23.0 * Units.deg
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 0.28
    wing.span_efficiency         = 1.0

##    wing.spans.projected         = 27.8
##
##    wing.chords.root             = 5.203
##    wing.chords.tip              = 1.460
##    wing.chords.mean_aerodynamic = 3.680

    wing.areas.reference         = 92.0
    wing.areas.wetted            = 2.0 * wing.areas.reference
    wing.areas.exposed           = 0.8 * wing.areas.wetted
    wing.areas.affected          = 0.6 * wing.areas.reference

    wing.twists.root             = 2.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [13.2,0,0] # Need to fix
##    wing.aerodynamic_center      = [3,0,0]

    wing.vertical                = False
    wing.symmetric               = True

    wing.high_lift               = True
    wing.flaps.type              = "double_slotted"
    wing.flaps.chord             = 0.280

    wing.dynamic_pressure_ratio  = 1.0

    # add to vehicle
    size_planform(wing)
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'horizontal_stabilizer'

    wing.aspect_ratio            = 5.5
    wing.sweeps.quarter_chord    = 34.5 * Units.deg
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 0.11
    wing.span_efficiency         = 0.9

##    wing.spans.projected         = 11.958
##
##    wing.chords.root             = 3.030
##    wing.chords.tip              = 0.883
##    wing.chords.mean_aerodynamic = 2.3840

    wing.areas.reference         = 26.0
    wing.areas.wetted            = 2.0 * wing.areas.reference
    wing.areas.exposed           = 0.8 * wing.areas.wetted
    wing.areas.affected          = 0.6 * wing.areas.reference

    wing.twists.root             = 2.0 * Units.degrees
    wing.twists.tip              = 2.0 * Units.degrees

    wing.origin                  = [31.,0,0] # need to fix
##    wing.aerodynamic_center      = [2,0,0]

    wing.vertical                = False
    wing.symmetric               = True

    wing.dynamic_pressure_ratio  = 0.9

    # add to vehicle
    size_planform(wing)
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'vertical_stabilizer'

    wing.aspect_ratio            = 1.7      #
    wing.sweeps.quarter_chord    = 35 * Units.deg
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 0.31
    wing.span_efficiency         = 0.9

##    wing.spans.projected         = 5.270     #
##
##    wing.chords.root             = 4.70
##    wing.chords.tip              = 1.45
##    wing.chords.mean_aerodynamic = 3.36

    wing.areas.reference         = 16.0    #
    wing.areas.wetted            = 2.0 * wing.areas.reference
    wing.areas.exposed           = 0.8 * wing.areas.wetted
    wing.areas.affected          = 0.6 * wing.areas.reference

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [29.5,0,0]
##    wing.aerodynamic_center      = [2,0,0]

    wing.vertical                = True
    wing.symmetric               = False

    wing.dynamic_pressure_ratio  = 1.0

    # add to vehicle
    size_planform(wing)
    vehicle.append_component(wing)
    # ------------------------------------------------------------------
    #  Turbofan Network
    # ------------------------------------------------------------------


    #initialize the gas turbine network
    gt_engine                   = SUAVE.Components.Energy.Networks.Turbofan()
    gt_engine.tag               = 'turbofan'

    gt_engine.number_of_engines = 2.0
    gt_engine.bypass_ratio      = 5.4
    gt_engine.engine_length     = 2.71
    gt_engine.nacelle_diameter  = 2.05
    gt_engine.position[1]       = 4.50
    
    #compute engine areas
    Awet    = 1.1*np.pi*gt_engine.nacelle_diameter*gt_engine.engine_length 
    
    #assign engine areas
    gt_engine.areas.wetted  = Awet
    
    
    #set the working fluid for the network
    working_fluid               = SUAVE.Attributes.Gases.Air

    #add working fluid to the network
    gt_engine.working_fluid = working_fluid


    #Component 1 : ram,  to convert freestream static to stagnation quantities
    ram = SUAVE.Components.Energy.Converters.Ram()
    ram.tag = 'ram'

    #add ram to the network
    gt_engine.ram = ram


    #Component 2 : inlet nozzle
    inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    inlet_nozzle.tag = 'inlet nozzle'

    inlet_nozzle.polytropic_efficiency = 0.98
    inlet_nozzle.pressure_ratio        = 0.98 #	turbofan.fan_nozzle_pressure_ratio     = 0.98     #0.98

    #add inlet nozzle to the network
    gt_engine.inlet_nozzle = inlet_nozzle


    #Component 3 :low pressure compressor
    low_pressure_compressor = SUAVE.Components.Energy.Converters.Compressor()
    low_pressure_compressor.tag = 'lpc'

    low_pressure_compressor.polytropic_efficiency = 0.91
    low_pressure_compressor.pressure_ratio        = 1.9

    #add low pressure compressor to the network
    gt_engine.low_pressure_compressor = low_pressure_compressor



    #Component 4 :high pressure compressor
    high_pressure_compressor = SUAVE.Components.Energy.Converters.Compressor()
    high_pressure_compressor.tag = 'hpc'

    high_pressure_compressor.polytropic_efficiency = 0.91
    high_pressure_compressor.pressure_ratio        = 10.0

    #add the high pressure compressor to the network
    gt_engine.high_pressure_compressor = high_pressure_compressor


    #Component 5 :low pressure turbine
    low_pressure_turbine = SUAVE.Components.Energy.Converters.Turbine()
    low_pressure_turbine.tag='lpt'

    low_pressure_turbine.mechanical_efficiency = 0.99
    low_pressure_turbine.polytropic_efficiency = 0.93

    #add low pressure turbine to the network
    gt_engine.low_pressure_turbine = low_pressure_turbine



    #Component 5 :high pressure turbine
    high_pressure_turbine = SUAVE.Components.Energy.Converters.Turbine()
    high_pressure_turbine.tag='hpt'

    high_pressure_turbine.mechanical_efficiency = 0.99
    high_pressure_turbine.polytropic_efficiency = 0.93

    #add the high pressure turbine to the network
    gt_engine.high_pressure_turbine = high_pressure_turbine


    #Component 6 :combustor
    combustor = SUAVE.Components.Energy.Converters.Combustor()
    combustor.tag = 'Comb'

    combustor.efficiency                = 0.99
    combustor.alphac                    = 1.0
    combustor.turbine_inlet_temperature = 1500
    combustor.pressure_ratio            = 0.95
    combustor.fuel_data                 = SUAVE.Attributes.Propellants.Jet_A()

    #add the combustor to the network
    gt_engine.combustor = combustor



    #Component 7 :core nozzle
    core_nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()
    core_nozzle.tag = 'core nozzle'

    core_nozzle.polytropic_efficiency = 0.95
    core_nozzle.pressure_ratio        = 0.99

    #add the core nozzle to the network
    gt_engine.core_nozzle = core_nozzle


    #Component 8 :fan nozzle
    fan_nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()
    fan_nozzle.tag = 'fan nozzle'

    fan_nozzle.polytropic_efficiency = 0.95
    fan_nozzle.pressure_ratio        = 0.99

    #add the fan nozzle to the network
    gt_engine.fan_nozzle = fan_nozzle



    #Component 9 : fan
    fan = SUAVE.Components.Energy.Converters.Fan()
    fan.tag = 'fan'

    fan.polytropic_efficiency = 0.93
    fan.pressure_ratio        = 1.7

    #add the fan to the network
    gt_engine.fan = fan

    #Component 10 : thrust (to compute the thrust)
    thrust = SUAVE.Components.Energy.Processes.Thrust()
    thrust.tag ='compute_thrust'

    #total design thrust (includes all the engines)
    thrust.total_design             = 37278.0* Units.N #Newtons

    #design sizing conditions
    altitude      = 35000.0*Units.ft
    mach_number   = 0.78
    isa_deviation = 0.

    # add thrust to the network
    gt_engine.thrust = thrust

    #size the turbofan
    #create conditions object for sizing the turbofan
    atmosphere             = SUAVE.Analyses.Atmospheric.US_Standard_1976()
  
    conditions             = atmosphere.compute_values(altitude)
    
    turbofan_sizing(gt_engine,mach_number,altitude)
    compute_turbofan_geometry(gt_engine, conditions)
    
    # add  gas turbine network gt_engine to the vehicle
    vehicle.append_component(gt_engine)


    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------

    return vehicle

def base_analysis(vehicle):
    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------     
    analyses = SUAVE.Analyses.Vehicle()
   
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)
    
    # ------------------------------------------------------------------
    #  Energy Analysis
    energy  = SUAVE.Analyses.Energy.Energy()
    energy.network=vehicle.propulsors
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
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    plt.show()