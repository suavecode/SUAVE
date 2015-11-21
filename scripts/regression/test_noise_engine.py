# test_noise.py
#
# Created:  Carlos
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave Imports

import SUAVE
from SUAVE.Core import Units
import time

from SUAVE.Methods.Noise.Fidelity_One.Airframe import noise_fidelity_one
from SUAVE.Methods.Noise.Fidelity_One.Engine import noise_SAE
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import flight_trajectory

from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import pnl_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_tone_correction
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import epnl_noise

import numpy as np


from SUAVE.Core import (
Data, Container, Data_Exception, Data_Warning,
)

from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():

    configs, analyses = full_setup()

  #  simple_sizing(configs)

    configs.finalize()
    analyses.finalize()
    
    tinital=time.time()
    
     
     #noise
     
  #  mics_array = analyses.microphone_array
  #  n_mics=np.size(mics_array)
    
  # for id in range (0,n_mics): 
    turbofan = configs.base.propulsors[0]
        
  # mic_position = mics_array[id]
    
    trajectory = flight_trajectory(configs,turbofan,analyses)

    airframe_noise=noise_fidelity_one(configs,analyses,trajectory) 

    engine_noise = noise_SAE(turbofan,trajectory,configs,analyses)
    
    total_aircraft_noise=total_noise(airframe_noise,engine_noise)
        
#   print mics_array[id]
#    print airframe_noise[0], engine_noise[0], total_aircraft_noise #modificado
    
    tfinal=time.time()
    
    total_time=tfinal-tinital
    
    print total_time
    print airframe_noise[0], engine_noise[0], total_aircraft_noise


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():

    # vehicle data
    vehicle  = vehicle_setup()
    configs  = configs_setup(vehicle)

    analyses = base_analyses(vehicle)

    return configs, analyses

def vehicle_setup():

     # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'EMBRAER_190'


    # ------------------------------------------------------------------
    #   Vehicle-level Properties for Airframe Noise Calculation
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    #   Landing 
    
    # ------------------------------------------------------------------

    vehicle.landing_gear = Data()
    vehicle.landing_gear.main_tire_diameter = 3.5000
    vehicle.landing_gear.nose_tire_diameter = 2.2000
    vehicle.landing_gear.main_strut_length = 5.66
    vehicle.landing_gear.nose_strut_length = 4.5
    vehicle.landing_gear.main_units = 2     #number of main landing gear units
    vehicle.landing_gear.nose_units = 1     #number of nose landing gear
    vehicle.landing_gear.main_wheels = 2    #number of wheels on the main landing gear
    vehicle.landing_gear.nose_wheels = 2    #number of wheels on the nose landing gear    


    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'

    wing.spans.projected         = 91.1089

    wing.areas.reference         = 995.9846


    # ------------------------------------------------------------------
    #   Flaps
    # ------------------------------------------------------------------
    
    wing.flaps.chord = 3.3497
    wing.flaps.area = 97.1112
    wing.flaps.number_slots = 2


    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'horizontal_stabilizer'

    wing.spans.projected         = 39.656      #

    wing.areas.reference         = 279.862    #

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'vertical_stabilizer'

    wing.spans.projected         = 17.290      #

    wing.areas.reference         = 174.375    #

    # add to vehicle
    vehicle.append_component(wing)



    # ------------------------------------------------------------------
    #   Turbofan Network
    # ------------------------------------------------------------------

    #instantiate the gas turbine network
    turbofan = SUAVE.Components.Energy.Networks.Turbofan()
    turbofan.tag = 'turbo_fan'

    # setup
    turbofan.number_of_engines = 2.0
    turbofan.bypass_ratio      = 5.4
    turbofan.engine_length     = 2.71
    turbofan.nacelle_diameter  = 2.05

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
    compressor.pressure_ratio        = 1.14

    # add to network
    turbofan.append(compressor)


    # ------------------------------------------------------------------
    #  Component 4 - High Pressure Compressor

    # instantiate
    compressor = SUAVE.Components.Energy.Converters.Compressor()
    compressor.tag = 'high_pressure_compressor'

    # setup
    compressor.polytropic_efficiency = 0.91
    compressor.pressure_ratio        = 13.415

    # add to network
    turbofan.append(compressor)


    # ------------------------------------------------------------------
    #  Component 5 - Low Pressure Turbine

    # instantiate
    turbine = SUAVE.Components.Energy.Converters.Turbine()
    turbine.tag='low_pressure_turbine'

    # setup
    turbine.mechanical_efficiency = 0.99
    turbine.polytropic_efficiency = 0.93

    # add to network
    turbofan.append(turbine)


    # ------------------------------------------------------------------
    #  Component 6 - High Pressure Turbine

    # instantiate
    turbine = SUAVE.Components.Energy.Converters.Turbine()
    turbine.tag='high_pressure_turbine'

    # setup
    turbine.mechanical_efficiency = 0.99
    turbine.polytropic_efficiency = 0.93

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
    combustor.turbine_inlet_temperature = 1450
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

    # for noise
    nozzle.jet_diameter = 1.

    # add to network
    turbofan.append(nozzle)


    # ------------------------------------------------------------------
    #  Component 9 - Fan Nozzle

    # instantiate
    nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()
    nozzle.tag = 'fan_nozzle'

    # setup
    nozzle.polytropic_efficiency = 0.95
    nozzle.pressure_ratio        = 0.99

    # for noise
    nozzle.jet_diameter = 2.

    # add to network
    turbofan.append(nozzle)


    # ------------------------------------------------------------------
    #  Component 10 - Fan

    # instantiate
    fan = SUAVE.Components.Energy.Converters.Fan()
    fan.tag = 'fan'

    # setup
    fan.polytropic_efficiency = 0.93
    fan.pressure_ratio  = 1.7

    # for noise
    fan.rotation = 3300 #1940 #3000#542 #2.

    # add to network
    turbofan.append(fan)


    # ------------------------------------------------------------------
    #Component 10 : thrust (to compute the thrust)
    thrust = SUAVE.Components.Energy.Processes.Thrust()
    thrust.tag ='compute_thrust'

    #total design thrust (includes all the engines)
    thrust.total_design             = 2*24000. * Units.N #Newtons

    #design sizing conditions
    altitude      = 35000.0*Units.ft
    mach_number   = 0.78
    isa_deviation = 0.

    # add to network
    turbofan.thrust = thrust

    #size the turbofan
  #Modification August
    turbofan_sizing(turbofan,mach_number,altitude)

    # add  gas turbine network gt_engine to the vehicle
    vehicle.append_component(turbofan)


    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------

    return vehicle


def base_analyses(vehicle):

    analyses = SUAVE.Analyses.Vehicle()
 # ------------------------------------------------------------------
    #  Planet Analysis
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)

    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)
    
    #Array of microphone positions - 31/08/2015
    
    analyses.microphone_array=np.array([-300,-150,0,150,300])

    return analyses

def configs_setup(vehicle):

    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------

    configs = SUAVE.Components.Configs.Config.Container()

    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    base_config.wings['main_wing'].flaps.angle = 7. * Units.deg
    
    #0 for gear up and 1 for gear down
    base_config.landing_gear.gear_condition = 'up' #0   
   
    configs.append(base_config)
    
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'flight'

    config.initial_position = -2000*Units.m
    config.initial_time = 0.0
    config.velocity = 166*Units.knots
    config.altitute = 500*Units.ft
    config.angle_of_climb=6.8    #[degrees]
    config.glide_slope=3
    
    config.angle_of_attack = 7
    
    #Choose trajectory
    config.flyover = 0
    config.approach = 0
    config.sideline = 0
    config.constant_flight=1
    config.output_filename='Teste'

    configs.append(config)
    


    return configs



def total_noise (airframe_noise, engine_noise):
    
    SPL_airframe = airframe_noise[1]
    SPL_engine = engine_noise[1]
      
   # nsteps=np.size(SPL_engine)
    
    #for id in range (0,nsteps):
    SPL_total= 10 * np.log10(10**(0.1*SPL_airframe)+10**(0.1*SPL_engine))
    
    
     #Calculation of the Perceived Noise Level EPNL based on the sound time history
    PNL_total =  pnl_noise(SPL_total)    
    
   #Calculation of the tones corrections on the SPL for each component and total
    tone_correction_total = noise_tone_correction(SPL_total) 
   
    #Calculation of the PLNT for each component and total
    PNLT_total=PNL_total+tone_correction_total
 
    #Calculation of the EPNL for each component and total
    EPNL_total=epnl_noise(PNLT_total)

    return (EPNL_total)

if __name__ == '__main__':
    main()

