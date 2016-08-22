# test_landing_field_length.py
#
# Created:  Tarik, Carlos, Celso, Jun 2014
# Modified: Emilio Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave Imports
import SUAVE
from SUAVE.Core            import Data
from SUAVE.Core import Units
from SUAVE.Core import Units
from SUAVE.Methods.Performance.estimate_landing_field_length import estimate_landing_field_length

# package imports
import numpy as np
import pylab as plt

# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------

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
    vehicle.mass_properties.takeoff = 50000. #

    # basic parameters
    vehicle.reference_area  = 92.    # m^2

    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'

    wing.areas.reference      = vehicle.reference_area
    wing.sweeps.quarter_chord = 22. * Units.deg  # deg
    wing.thickness_to_chord   = 0.11
    wing.taper                = 0.28          

    wing.chords.mean_aerodynamic = 3.66
    wing.areas.affected          = 0.6*wing.areas.reference # part of high lift system
    wing.flaps.type   = 'double_slotted'
    wing.flaps.chord  = 0.28
    wing.high_lift    =True
    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #  Turbofan
    # ------------------------------------------------------------------

    
    #instantiate the gas turbine network
    turbofan = SUAVE.Components.Energy.Networks.Turbofan()
    turbofan.tag = 'turbofan'
    
    # setup
    turbofan.number_of_engines = 2.0
    turbofan.design_thrust     = 20300.0
    turbofan.engine_length     = 3.0
    turbofan.nacelle_diameter  = 1.0
    
    # working fluid
    turbofan.working_fluid = SUAVE.Attributes.Gases.Air()
    
    
    # ------------------------------------------------------------------
    #   Component 1 - Ram
    
    # to convert freestream static to stagnation quantities
    
    # instantiate
    ram = SUAVE.Components.Energy.Converters.Ram()
    ram.tag = 'ram'
    
    # add to the network
    turbofan.append(ram)


    # ------------------------------------------------------------------
    #  Component 2 - Inlet Nozzle
    
    # instantiate
    inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    inlet_nozzle.tag = 'inlet_nozzle'
    
    # setup
    inlet_nozzle.polytropic_efficiency = 0.98
    inlet_nozzle.pressure_ratio        = 0.98
    
    # add to network
    turbofan.append(inlet_nozzle)
    
    
    # ------------------------------------------------------------------
    #  Component 3 - Low Pressure Compressor
    
    # instantiate 
    compressor = SUAVE.Components.Energy.Converters.Compressor()    
    compressor.tag = 'low_pressure_compressor'

    # setup
    compressor.polytropic_efficiency = 0.91
    compressor.pressure_ratio        = 1.9     
    
    # add to network
    turbofan.append(compressor)

    
    # ------------------------------------------------------------------
    #  Component 4 - High Pressure Compressor
    
    # instantiate
    compressor = SUAVE.Components.Energy.Converters.Compressor()    
    compressor.tag = 'high_pressure_compressor'
    
    # setup
    compressor.polytropic_efficiency =  0.91
    compressor.pressure_ratio        =  10.0    
    
    # add to network
    turbofan.append(compressor)


    # ------------------------------------------------------------------
    #  Component 5 - Low Pressure Turbine
    
    # instantiate
    turbine = SUAVE.Components.Energy.Converters.Turbine()   
    turbine.tag='low_pressure_turbine'
    
    # setup
    turbine.mechanical_efficiency = 0.99
    turbine.polytropic_efficiency = 0.99     
    
    # add to network
    turbofan.append(turbine)
    
      
    # ------------------------------------------------------------------
    #  Component 6 - High Pressure Turbine
    
    # instantiate
    turbine = SUAVE.Components.Energy.Converters.Turbine()   
    turbine.tag='high_pressure_turbine'

    # setup
    turbine.mechanical_efficiency = 0.99
    turbine.polytropic_efficiency = 0.99     
    
    # add to network
    turbofan.append(turbine)
      
    
    # ------------------------------------------------------------------
    #  Component 7 - Combustor
    
    # instantiate    
    combustor = SUAVE.Components.Energy.Converters.Combustor()   
    combustor.tag = 'combustor'
    
    # setup
    combustor.efficiency                = 0.99 
    combustor.alphac                    = 1.0     
    combustor.turbine_inlet_temperature = 1500
    combustor.pressure_ratio            = 0.95
    combustor.fuel_data                 = SUAVE.Attributes.Propellants.Jet_A()    
    
    # add to network
    turbofan.append(combustor)

    
    # ------------------------------------------------------------------
    #  Component 8 - Core Nozzle
    
    # instantiate
    nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    nozzle.tag = 'core_nozzle'
    
    # setup
    nozzle.polytropic_efficiency = 0.95
    nozzle.pressure_ratio        = 0.99    
    
    # add to network
    turbofan.append(nozzle)


    # ------------------------------------------------------------------
    #  Component 9 - Fan Nozzle
    
    # instantiate
    nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    nozzle.tag = 'fan_nozzle'

    # setup
    nozzle.polytropic_efficiency = 0.95
    nozzle.pressure_ratio        = 0.98    
    
    # add to network
    turbofan.append(nozzle)
    
    
    # ------------------------------------------------------------------
    #  Component 10 - Fan
    
    # instantiate
    fan = SUAVE.Components.Energy.Converters.Fan()   
    fan.tag = 'fan'

    # setup
    fan.polytropic_efficiency = 0.93
    fan.pressure_ratio        = 1.7    
    
    # add to network
    turbofan.append(fan)
    
    
    # ------------------------------------------------------------------
    #  Component 10 - Thrust
    
    # to compute thrust
    
    # instantiate
    thrust = SUAVE.Components.Energy.Processes.Thrust()       
    thrust.tag ='thrust'
    
    # setup
    thrust.bypass_ratio                       = 5.4
    thrust.compressor_nondimensional_massflow = 40.0
    thrust.reference_temperature              = 288.15
    thrust.reference_pressure                 = 1.01325*10**5
    thrust.number_of_engines                  = turbofan.number_of_engines   
    
    # add to network
    turbofan.thrust = thrust
    
    
    # add turbofan to vehicle
    vehicle.propulsors.append(turbofan)

    # ------------------------------------------------------------------
    #   Simple Propulsion Model
    # ------------------------------------------------------------------

    vehicle.propulsion_model = vehicle.propulsors


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
    #  Propulsion Analysis
    propulsion = SUAVE.Analyses.Energy.Propulsion()
    propulsion.vehicle = vehicle
    analyses.append(propulsion)
    
    # done!
    return analyses    
    
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
    #   Landing Configuration
    # ------------------------------------------------------------------
    
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'landing'
    
    config.wings['main_wing'].flaps.angle = 15. * Units.deg
  
    
    config.V2_VS_ratio = 1.20
  
    configs.append(config)
    return configs       
    
def main():

    # ----------------------------------------------------------------------
    #   Main
    # ----------------------------------------------------------------------

    # --- Vehicle definition ---
    vehicle = vehicle_setup()
    configs=configs_setup(vehicle)
    # --- Landing Configuration ---
    landing_config = configs.landing
    landing_config.wings['main_wing'].flaps.angle =  30. * Units.deg
    landing_config.wings['main_wing'].slats.angle = 25. * Units.deg
    landing_config.wings['main_wing'].high_lift  = True
    # Vref_V2_ratio may be informed by user. If not, use default value (1.23)
    landing_config.Vref_VS_ratio = 1.23
    # CLmax for a given configuration may be informed by user
    # landing_config.maximum_lift_coefficient = 2.XX

    # --- Airport definition ---
    airport = SUAVE.Attributes.Airports.Airport()
    airport.tag = 'airport'
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere =  SUAVE.Analyses.Atmospheric.US_Standard_1976()

    # =====================================
    # Landing field length evaluation
    # =====================================
    w_vec = np.linspace(20000.,44000.,10)
    landing_field_length = np.zeros_like(w_vec)
    for id_w,weight in enumerate(w_vec):
        landing_config.mass_properties.landing = weight
        landing_field_length[id_w] = estimate_landing_field_length(landing_config,landing_config,airport)

    truth_LFL = np.array([705.78061286, 766.55136124, 827.32210962, 888.092858, 948.86360638, 
                          1009.63435476, 1070.40510314, 1131.17585152, 1191.9465999, 1252.71734828])
    
    LFL_error = np.max(np.abs(landing_field_length-truth_LFL))
    
    print 'Maximum Landing Field Length Error= %.4e' % LFL_error
    
    title = "LFL vs W"
    plt.figure(1); plt.hold
    plt.plot(w_vec,landing_field_length, 'k-', label = 'Landing Field Length')

    plt.title(title); plt.grid(True)
    legend = plt.legend(loc='lower right', shadow = 'true')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Landing Field Length (m)')
    
    plt.figure(2); plt.plot(w_vec,truth_LFL)
    #assert( LFL_error   < 1e-5 )

    return 
    
# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    plt.show()
        
