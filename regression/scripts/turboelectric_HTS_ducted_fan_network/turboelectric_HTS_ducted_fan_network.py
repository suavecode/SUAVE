### @ingroup Regression-scripts-turboelectric_HTS_ducted_fan_network
#turboelectric_HTS_ducted_fan_network.py
# 
# Created:  Nov 2021, S. Claridge
# 


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE

import numpy as np

from SUAVE.Components.Energy.Networks.Turboelectric_HTS_Ducted_Fan import Turboelectric_HTS_Ducted_Fan   
from SUAVE.Methods.Propulsion.serial_HTS_turboelectric_sizing import serial_HTS_turboelectric_sizing

from SUAVE.Attributes.Gases import Air
from SUAVE.Attributes.Solids.Copper import Copper

import cProfile, pstats, io

from SUAVE.Core import (
Data, Units,
)
from SUAVE.Methods.Propulsion.ducted_fan_sizing import ducted_fan_sizing
   
### @ingroup Regression-scripts-turboelectric_HTS_ducted_fan_network

def profile(fnc):
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

@profile
def main():   
    
    # call the network function
    energy_network()
    
    return


def energy_network():

   # ------------------------------------------------------------------
    #   Evaluation Conditions
    # ------------------------------------------------------------------    
    
    # Conditions        
    ones_1col = np.ones([1,1])       
    alt       = 10.0
    
    # Setup conditions
    planet                           = SUAVE.Attributes.Planets.Earth()   
    atmosphere                       = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data                        = atmosphere.compute_values(alt,0,True) 
    working_fluid                    = SUAVE.Attributes.Gases.Air()    
    conditions                       = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
  
    # freestream conditions
    conditions.freestream.altitude                     = ones_1col*alt   
    conditions.freestream.mach_number                  = ones_1col*0.8
    conditions.freestream.pressure                     = ones_1col*atmo_data.pressure
    conditions.freestream.temperature                  = ones_1col*atmo_data.temperature
    conditions.freestream.density                      = ones_1col*atmo_data.density
    conditions.freestream.dynamic_viscosity            = ones_1col*atmo_data.dynamic_viscosity
    conditions.freestream.gravity                      = ones_1col*planet.compute_gravity(alt)
    conditions.freestream.isentropic_expansion_factor  = ones_1col*working_fluid.compute_gamma(atmo_data.temperature,atmo_data.pressure)                                                                                             
    conditions.freestream.Cp                           = ones_1col*working_fluid.compute_cp(atmo_data.temperature,atmo_data.pressure)
    conditions.freestream.R                            = ones_1col*working_fluid.gas_specific_constant
    conditions.freestream.speed_of_sound               = ones_1col*atmo_data.speed_of_sound
    conditions.freestream.velocity                     = conditions.freestream.mach_number*conditions.freestream.speed_of_sound
    conditions.velocity                                = conditions.freestream.mach_number*conditions.freestream.speed_of_sound
    conditions.q                                       = 0.5*conditions.freestream.density*conditions.velocity**2
    conditions.g0                                      = conditions.freestream.gravity
    
    # propulsion conditions
    conditions.propulsion.throttle                     =  ones_1col*1.0
        
    # ------------------------------------------------------------------
    #   Design/sizing conditions
    # ------------------------------------------------------------------    
    
    # Conditions        
    ones_1col = np.ones([1,1])    
    alt_size  = 10000.0
    # Setup conditions
    planet     = SUAVE.Attributes.Planets.Earth()   
    atmosphere                       = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data                        = atmosphere.compute_values(alt_size,0,True) 
    working_fluid                    = SUAVE.Attributes.Gases.Air()    
    conditions_sizing = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()

    # freestream conditions
    conditions_sizing.freestream.altitude                     = ones_1col*alt_size     
    conditions_sizing.freestream.mach_number                  = ones_1col*0.8
    conditions_sizing.freestream.pressure                     = ones_1col*atmo_data.pressure
    conditions_sizing.freestream.temperature                  = ones_1col*atmo_data.temperature
    conditions_sizing.freestream.density                      = ones_1col*atmo_data.density
    conditions_sizing.freestream.dynamic_viscosity            = ones_1col*atmo_data.dynamic_viscosity
    conditions_sizing.freestream.gravity                      = ones_1col*planet.compute_gravity(alt_size)
    conditions_sizing.freestream.isentropic_expansion_factor  = ones_1col*working_fluid.compute_gamma(atmo_data.temperature,atmo_data.pressure)                                                                                             
    conditions_sizing.freestream.Cp                           = ones_1col*working_fluid.compute_cp(atmo_data.temperature,atmo_data.pressure)
    conditions_sizing.freestream.R                            = ones_1col*working_fluid.gas_specific_constant
    conditions_sizing.freestream.speed_of_sound               = ones_1col*atmo_data.speed_of_sound
    conditions_sizing.freestream.velocity                     = conditions_sizing.freestream.mach_number*conditions_sizing.freestream.speed_of_sound
    conditions_sizing.velocity                                = conditions_sizing.freestream.mach_number*conditions_sizing.freestream.speed_of_sound
    conditions_sizing.q                                       = 0.5*conditions_sizing.freestream.density*conditions_sizing.velocity**2
    conditions_sizing.g0                                      = conditions_sizing.freestream.gravity
    
    # propulsion conditions
    conditions_sizing.propulsion.throttle                    =  ones_1col*1.0

    state_sizing                = Data()
    state_sizing.numerics       = Data()
    state_sizing.conditions     = conditions_sizing
    state_off_design            = Data()
    state_off_design.numerics   = Data()
    state_off_design.conditions = conditions
    

    
    # ------------------------------------------------------------------
    #   Turboelectric HTS Ducted Fan Network 
    # ------------------------------------------------------------------    
    
    # Instantiate the Turboelectric HTS Ducted Fan Network 
    # This also instantiates the component parts of the efan network, then below each part has its properties modified so they are no longer the default properties as created here at instantiation. 
    efan = Turboelectric_HTS_Ducted_Fan()
    efan.tag = 'turbo_fan'

    # Outline of Turboelectric drivetrain components. These are populated below.
    # 1. Propulsor     Ducted_fan
    #   1.1 Ram
    #   1.2 Inlet Nozzle
    #   1.3 Fan Nozzle
    #   1.4 Fan
    #   1.5 Thrust
    # 2. Motor         
    # 3. Powersupply   
    # 4. ESC           
    # 5. Rotor         
    # 6. Lead          
    # 7. CCS           
    # 8. Cryocooler    
    # 9. Heat Exchanger
    # The components are then sized

    # ------------------------------------------------------------------
    #   Component 1 - Ducted Fan
    
    efan.ducted_fan                    = SUAVE.Components.Energy.Networks.Ducted_Fan()
    efan.ducted_fan.tag                = 'ducted_fan'
    efan.ducted_fan.number_of_engines  = 12.
    efan.number_of_engines             = efan.ducted_fan.number_of_engines
    efan.ducted_fan.engine_length      = 1.1            * Units.meter


    # Positioning variables for the propulsor locations - 
    xStart = 15.0
    xSpace = 1.0
    yStart = 3.0
    ySpace = 1.8
    efan.ducted_fan.origin =   [    [xStart+xSpace*5, -(yStart+ySpace*5), -2.0],
                                    [xStart+xSpace*4, -(yStart+ySpace*4), -2.0],
                                    [xStart+xSpace*3, -(yStart+ySpace*3), -2.0],
                                    [xStart+xSpace*2, -(yStart+ySpace*2), -2.0],
                                    [xStart+xSpace*1, -(yStart+ySpace*1), -2.0],
                                    [xStart+xSpace*0, -(yStart+ySpace*0), -2.0],
                                    [xStart+xSpace*5,  (yStart+ySpace*5), -2.0],
                                    [xStart+xSpace*4,  (yStart+ySpace*4), -2.0],
                                    [xStart+xSpace*3,  (yStart+ySpace*3), -2.0],
                                    [xStart+xSpace*2,  (yStart+ySpace*2), -2.0],
                                    [xStart+xSpace*1,  (yStart+ySpace*1), -2.0],
                                    [xStart+xSpace*0,  (yStart+ySpace*0), -2.0]     ] # meters 

    # copy the ducted fan details to the turboelectric ducted fan network to enable drag calculations
    efan.engine_length      = efan.ducted_fan.engine_length   
    efan.origin             = efan.ducted_fan.origin


    # working fluid
    efan.ducted_fan.working_fluid = SUAVE.Attributes.Gases.Air()
    
    # ------------------------------------------------------------------
    #   Component 1.1 - Ram
    
    # to convert freestream static to stagnation quantities
    # instantiate
    ram = SUAVE.Components.Energy.Converters.Ram()
    ram.tag = 'ram'
    
    # add to the network
    efan.ducted_fan.append(ram)

    # ------------------------------------------------------------------
    #  Component 1.2 - Inlet Nozzle
    
    # instantiate
    inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    inlet_nozzle.tag = 'inlet_nozzle'
    
    # setup
    inlet_nozzle.polytropic_efficiency = 0.98
    inlet_nozzle.pressure_ratio        = 0.98
    
    # add to network
    efan.ducted_fan.append(inlet_nozzle)

    # ------------------------------------------------------------------
    #  Component 1.3 - Fan Nozzle
    
    # instantiate
    fan_nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    fan_nozzle.tag = 'fan_nozzle'

    # setup
    fan_nozzle.polytropic_efficiency = 0.95
    fan_nozzle.pressure_ratio        = 0.99    
    
    # add to network
    efan.ducted_fan.append(fan_nozzle)
    
    # ------------------------------------------------------------------
    #  Component 1.4 - Fan
    
    # instantiate
    fan = SUAVE.Components.Energy.Converters.Fan()
    fan.tag = 'fan'

    # setup
    fan.polytropic_efficiency = 0.93
    fan.pressure_ratio        = 1.7    
    
    # add to network
    efan.ducted_fan.append(fan)

    # ------------------------------------------------------------------
    # Component 1.5 : thrust

    # To compute the thrust
    thrust = SUAVE.Components.Energy.Processes.Thrust()       
    thrust.tag ='compute_thrust'
 
    # total design thrust (includes all the propulsors)
    thrust.total_design             = 2.*24000. * Units.N #Newtons
 
    # design sizing conditions
    altitude      = 35000.0*Units.ft
    mach_number   =     0.78 
    isa_deviation =     0.
    
    # add to network
    efan.ducted_fan.thrust = thrust
    # ------------------------------------------------------------------
    # Component 2 : HTS motor
    
    efan.motor = SUAVE.Components.Energy.Converters.Motor_Lo_Fid()
    efan.motor.tag = 'motor'
    # number_of_motors is not used as the motor count is assumed to match the engine count

    # Set the origin of each motor to match its ducted fan
    efan.motor.origin = efan.ducted_fan.origin
    efan.motor.gear_ratio         = 1.0
    efan.motor.gearbox_efficiency = 1.0
    efan.motor.motor_efficiency   = 0.96

    # ------------------------------------------------------------------
    #  Component 3 - Powersupply
    
    efan.powersupply                        = SUAVE.Components.Energy.Converters.Turboelectric()
    efan.powersupply.tag                    = 'powersupply'
    efan.number_of_powersupplies            = 2.
    efan.powersupply.propellant             = SUAVE.Attributes.Propellants.Jet_A()
    efan.powersupply.oxidizer               = Air()
    efan.powersupply.number_of_engines      = 2.0                   # number of turboelectric machines, not propulsors
    efan.powersupply.efficiency             = .37                   # Approximate average gross efficiency across the product range.
    efan.powersupply.volume                 = 2.36    *Units.m**3.  # 3m long from RB211 datasheet. 1m estimated radius.
    efan.powersupply.rated_power            = 37400.0 *Units.kW
    efan.powersupply.mass_properties.mass   = 2500.0  *Units.kg     # 2.5 tonnes from Rolls Royce RB211 datasheet 2013.
    efan.powersupply.specific_power         = efan.powersupply.rated_power/efan.powersupply.mass_properties.mass
    efan.powersupply.mass_density           = efan.powersupply.mass_properties.mass /efan.powersupply.volume 

    # ------------------------------------------------------------------
    
    efan.esc = SUAVE.Components.Energy.Distributors.HTS_DC_Supply()     # Could make this where the ESC is defined as a Siemens SD104
    efan.esc.tag = 'esc'

    efan.esc.efficiency             =   0.95                 # Siemens SD104 SiC Power Electronicss reported to be this efficient

    # ------------------------------------------------------------------
    #  Component 5 - HTS rotor (part of the propulsor motor)
    
    efan.rotor = SUAVE.Components.Energy.Converters.Motor_HTS_Rotor()
    efan.rotor.tag = 'rotor'

    efan.rotor.temperature              =    50.0       # [K]
    efan.rotor.skin_temp                =   300.0       # [K]       Temp of rotor outer surface is not ambient
    efan.rotor.current                  =  1000.0       # [A]       Most of the cryoload will scale with this number if not using HTS Dynamo
    efan.rotor.resistance               =     0.0001    # [ohm]     20 x 100 nOhm joints should be possible (2uOhm total) so 1mOhm is an overestimation.
    efan.rotor.number_of_engines        = efan.ducted_fan.number_of_engines      
    efan.rotor.length                   =     0.573     * Units.meter       # From paper: DOI:10.2514/6.2019-4517 Would be good to estimate this from power instead.
    efan.rotor.diameter                 =     0.310     * Units.meter       # From paper: DOI:10.2514/6.2019-4517 Would be good to estimate this from power instead.
    rotor_end_area                      = np.pi*(efan.rotor.diameter/2.0)**2.0
    rotor_end_circumference             = np.pi*efan.rotor.diameter
    efan.rotor.surface_area             = 2.0 * rotor_end_area + efan.rotor.length*rotor_end_circumference
    efan.rotor.R_value                  =   125.0                           # [K.m2/W]  2.0 W/m2 based on experience at Robinson Research

    # ------------------------------------------------------------------
    #  Component 6 - Copper Supply Leads of propulsion motor rotors
    
    efan.lead = SUAVE.Components.Energy.Distributors.Cryogenic_Lead()
    efan.lead.tag = 'lead'
    copper = Copper()
    efan.lead.cold_temp                 = efan.rotor.temperature   # [K]
    efan.lead.hot_temp                  = efan.rotor.skin_temp     # [K]
    efan.lead.current                   = efan.rotor.current       # [A]
    efan.lead.length                    = 0.3                      # [m]
    efan.lead.material                  = copper
    efan.leads                          = efan.ducted_fan.number_of_engines * 2.0      # Each motor has two leads to make a complete circuit

    # ------------------------------------------------------------------
    #  Component 7 - Rotor Constant Current Supply (CCS)
    
    efan.ccs = SUAVE.Components.Energy.Distributors.HTS_DC_Supply()
    efan.ccs.tag = 'ccs'

    efan.ccs.efficiency             =   0.95               # Siemens SD104 SiC Power Electronics reported to be this efficient


    # ------------------------------------------------------------------
    #  Component 8 - Cryocooler, to cool the HTS Rotor
 
    efan.cryocooler = SUAVE.Components.Energy.Cooling.Cryocooler()
    efan.cryocooler.tag = 'cryocooler'

    efan.cryocooler.cooler_type        = 'GM'
    efan.cryocooler.min_cryo_temp      = efan.rotor.temperature     # [K]
    efan.cryocooler.ambient_temp       = 300.0                      # [K]

    # ------------------------------------------------------------------
    #  Component 9 - Cryogenic Heat Exchanger, to cool the HTS Rotor
    efan.heat_exchanger = SUAVE.Components.Energy.Cooling.Cryogenic_Heat_Exchanger()
    efan.heat_exchanger.tag = 'heat_exchanger'

    efan.heat_exchanger.cryogen                         = SUAVE.Attributes.Cryogens.Liquid_H2()
    efan.heat_exchanger.cryogen_inlet_temperature       =     20.0                  # [K]
    efan.heat_exchanger.cryogen_outlet_temperature      = efan.rotor.temperature    # [K]
    efan.heat_exchanger.cryogen_pressure                = 100000.0                  # [Pa]
    efan.heat_exchanger.cryogen_is_fuel                 =      0.0

    # Sizing Conditions. The cryocooler may have greater power requirement at low altitude as the cooling requirement may be static during the flight but the ambient temperature may change.
    cryo_temp       =  50.0     # [K]
    amb_temp        = 300.0     # [K]

    # Powertrain Sizing


    # Size powertrain components
    ducted_fan_sizing(efan.ducted_fan,mach_number,altitude)
    serial_HTS_turboelectric_sizing(efan,mach_number,altitude, cryo_cold_temp = cryo_temp, cryo_amb_temp = amb_temp)

    print("Design thrust ",efan.ducted_fan.design_thrust)
    
    print("Sealevel static thrust ",efan.ducted_fan.sealevel_static_thrust)
    
    results_design          = efan(state_sizing)
    results_off_design      = efan(state_off_design)
    F                       = results_design.thrust_force_vector
    mdot                    = results_design.vehicle_mass_rate
    mdot_fuel               = results_design.vehicle_fuel_rate
    mdot_additional_fuel    = results_design.vehicle_additional_fuel_rate 
    F_off_design            = results_off_design.thrust_force_vector
    mdot_off_design         = results_off_design.vehicle_mass_rate
    
    # Test the model 
    # Specify the expected values
    expected = Data()
    expected.thrust                 = 47826.12361690928
    expected.mdot                   = 0.807913394579505
    expected.mdot_fuel              = 0.79080567
    expected.mdot_additional_fuel   = 0.01710773
    
    #error data function
    error =  Data()
    error.thrust_error                  = (F[0][0] -  expected.thrust)/expected.thrust
    error.mdot_error                    = (mdot[0][0]-expected.mdot)/expected.mdot
    error.mdot_fuel_error               = (mdot_fuel[0][0]-expected.mdot_fuel)/expected.mdot_fuel
    error.mdot_additional_fuel_error    = (mdot_additional_fuel[0][0]-expected.mdot_additional_fuel)/expected.mdot_additional_fuel
    
    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)    
        
    return
    
if __name__ == '__main__':
    
    main()    
    