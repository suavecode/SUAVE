## @ingroup Methods-Propulsion
# Serial_HTS_dynamo_Turboelectric_Sizing.py
# 
# Created:  K. Hamilton, Apr 2020
# Modified: S. Claridge, Feb 2022
#        


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Core import Data
from SUAVE.Methods.Power.Turboelectric.Sizing.initialize_from_power import initialize_from_power



## @ingroup Methods-Propulsion
def serial_HTS_dynamo_turboelectric_sizing(Turboelectric_HTS_Dynamo_Ducted_Fan, mach_number = None, altitude = None, delta_isa = 0, conditions = None, cryo_cold_temp = 50.0, cryo_amb_temp = 300.0):  
    """
    create and evaluate a serial hybrid network that follows the power flow:
    Turboelectric Generators -> Motor Drivers -> Electric Poropulsion Motors
    where the electric motors have cryogenically cooled HTS rotors that follow the power flow:
    Turboelectric Generators -> HTS Dynamo -> HTS Rotor Coils
    and
    Turboelectric Generators -> Cryocooler <- HTS Rotor Heat Load
    There is also the capability for the HTS components to be cryogenically cooled using liquid or gaseous cryogen, howver this is not sized other than applying a factor to the cryocooler required power.

    creates and evaluates a ducted_fan network based on an atmospheric sizing condition
    creates and evaluates a serial hybrid network that includes a HTS motor driven ducted fan, turboelectric generator, and the required supporting equipment including cryogenic cooling and HTS dynamo current supply.

    Assumptions:
        One powertrain model represents all engines in the model.
        There are no transmission losses between components
        the shaft torque and power required from the fan is the same as what would be required from the fan of a turbofan engine.

    Source:
        N/A

    Inputs:
        Turboelectric_HTS_Dynamo_Ducted_Fan    Serial HTYS hybrid ducted fan network object (to be modified)
        mach_number
        altitude                        [meters]
        delta_isa                       temperature difference [K]
        conditions                      ordered dict object

    Outputs:
        N/A

    Properties Used:
        N/A
    """
    
    # Unpack components
    ducted_fan      = Turboelectric_HTS_Dynamo_Ducted_Fan.ducted_fan       # Propulsion fans
    motor           = Turboelectric_HTS_Dynamo_Ducted_Fan.motor            # Propulsion fan motors
    turboelectric   = Turboelectric_HTS_Dynamo_Ducted_Fan.powersupply      # Electricity providers
    esc             = Turboelectric_HTS_Dynamo_Ducted_Fan.esc              # Propulsion motor speed controllers
    rotor           = Turboelectric_HTS_Dynamo_Ducted_Fan.rotor            # Propulsion motor HTS rotors
    hts_dynamo      = Turboelectric_HTS_Dynamo_Ducted_Fan.hts_dynamo       # HTS Dynamo current supply
    dynamo_esc      = Turboelectric_HTS_Dynamo_Ducted_Fan.dynamo_esc       # HTS Dynamo speed controller
    cryocooler      = Turboelectric_HTS_Dynamo_Ducted_Fan.cryocooler       # HTS rotor cryocoolers
    heat_exchanger  = Turboelectric_HTS_Dynamo_Ducted_Fan.heat_exchanger   # HTS rotor cryocooling via cryogen

    # Dummy values for specifications not currently used for analysis
    motor_current   = 100.0
    
    # check if altitude is passed or conditions is passed
    if(conditions):
        # use conditions
        pass
        
    else:
        # check if mach number and temperature are passed
        if(mach_number==None or altitude==None):
            
            # raise an error
            raise NameError('The sizing conditions require an altitude and a Mach number')
        
        else:
            # call the atmospheric model to get the conditions at the specified altitude
            atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
            atmo_data  = atmosphere.compute_values(altitude,delta_isa)
            planet     = SUAVE.Attributes.Planets.Earth()
            
            p   = atmo_data.pressure          
            T   = atmo_data.temperature       
            rho = atmo_data.density          
            a   = atmo_data.speed_of_sound    
            mu  = atmo_data.dynamic_viscosity   
        
            # setup conditions
            conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()            

            # freestream conditions           
            conditions.freestream.altitude                    = np.atleast_1d(altitude)
            conditions.freestream.mach_number                 = np.atleast_1d(mach_number)
            conditions.freestream.pressure                    = np.atleast_1d(p)
            conditions.freestream.temperature                 = np.atleast_1d(T)
            conditions.freestream.density                     = np.atleast_1d(rho)
            conditions.freestream.dynamic_viscosity           = np.atleast_1d(mu)
            conditions.freestream.gravity                     = np.atleast_1d(planet.compute_gravity(altitude)                                                                                                    )
            conditions.freestream.isentropic_expansion_factor = np.atleast_1d(ducted_fan.working_fluid.compute_gamma(T,p))
            conditions.freestream.Cp                          = np.atleast_1d(ducted_fan.working_fluid.compute_cp(T,p))
            conditions.freestream.R                           = np.atleast_1d(ducted_fan.working_fluid.gas_specific_constant)
            conditions.freestream.speed_of_sound              = np.atleast_1d(a)
            conditions.freestream.velocity                    = conditions.freestream.mach_number*conditions.freestream.speed_of_sound
            
            # propulsion conditions
            conditions.propulsion.throttle           =  np.atleast_1d(1.0)
    
    # Setup Components   
    ram                       = ducted_fan.ram
    inlet_nozzle              = ducted_fan.inlet_nozzle
    fan                       = ducted_fan.fan
    fan_nozzle                = ducted_fan.fan_nozzle
    thrust                    = ducted_fan.thrust
    
    bypass_ratio              = ducted_fan.bypass_ratio #0
    number_of_engines         = Turboelectric_HTS_Dynamo_Ducted_Fan.number_of_engines

    # Creating the network by manually linking the different components
    
    # set the working fluid to determine the fluid properties
    ram.inputs.working_fluid = ducted_fan.working_fluid
    
    # Flow through the ram , this computes the necessary flow quantities and stores it into conditions
    ram(conditions)

    # link inlet nozzle to ram 
    inlet_nozzle.inputs = ram.outputs
    
    # Flow through the inlet nozzle
    inlet_nozzle(conditions)
        
    # Link the fan to the inlet nozzle
    fan.inputs = inlet_nozzle.outputs
    
    # flow through the fan
    fan(conditions)        
    
    # link the fan nozzle to the fan
    fan_nozzle.inputs =  fan.outputs
    
    # flow through the fan nozzle
    fan_nozzle(conditions)
    
    # compute the thrust using the thrust component
    
    # link the thrust component to the fan nozzle
    thrust.inputs.fan_exit_velocity                        = fan_nozzle.outputs.velocity
    thrust.inputs.fan_area_ratio                           = fan_nozzle.outputs.area_ratio
    thrust.inputs.fan_nozzle                               = fan_nozzle.outputs
    thrust.inputs.number_of_engines                        = number_of_engines
    thrust.inputs.bypass_ratio                             = bypass_ratio
    thrust.inputs.total_temperature_reference              = fan_nozzle.outputs.stagnation_temperature
    thrust.inputs.total_pressure_reference                 = fan_nozzle.outputs.stagnation_pressure
    thrust.inputs.flow_through_core                        = 0.
    thrust.inputs.flow_through_fan                         = 1.
    
    # nonexistant components used to run thrust
    thrust.inputs.core_exit_velocity                       = 0.
    thrust.inputs.core_area_ratio                          = 0.
    thrust.inputs.core_nozzle                              = Data()
    thrust.inputs.core_nozzle.velocity                     = 0.
    thrust.inputs.core_nozzle.area_ratio                   = 0.
    thrust.inputs.core_nozzle.static_pressure              = 0.      

    # compute the thrust
    thrust.size(conditions) 
    mass_flow  = thrust.mass_flow_rate_design

    # compute total shaft power required (i.e. the sum of the shaft power provided by all the fans)
    shaft_power                                             = fan.outputs.work_done * mass_flow
    total_shaft_power                                       = shaft_power * number_of_engines
    Turboelectric_HTS_Dynamo_Ducted_Fan.design_shaft_power  = total_shaft_power

    # Shaft power seems to be half the expected. 3 MW expected per motor. 1.336 MW reported

    # update the design thrust value
    ducted_fan.design_thrust = thrust.total_design

    # compute the sls_thrust
    
    # call the atmospheric model to get the conditions at the specified altitude
    atmosphere_sls = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data = atmosphere_sls.compute_values(0.0,0.0)
    
    p   = atmo_data.pressure          
    T   = atmo_data.temperature       
    rho = atmo_data.density          
    a   = atmo_data.speed_of_sound    
    mu  = atmo_data.dynamic_viscosity   

    # setup conditions
    conditions_sls = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()            

    # freestream conditions
    conditions_sls.freestream.altitude                    = np.atleast_1d(0.)
    conditions_sls.freestream.mach_number                 = np.atleast_1d(0.01)
    conditions_sls.freestream.pressure                    = np.atleast_1d(p)
    conditions_sls.freestream.temperature                 = np.atleast_1d(T)
    conditions_sls.freestream.density                     = np.atleast_1d(rho)
    conditions_sls.freestream.dynamic_viscosity           = np.atleast_1d(mu)
    conditions_sls.freestream.gravity                     = np.atleast_1d(planet.sea_level_gravity)
    conditions_sls.freestream.isentropic_expansion_factor = np.atleast_1d(ducted_fan.working_fluid.compute_gamma(T,p))
    conditions_sls.freestream.Cp                          = np.atleast_1d(ducted_fan.working_fluid.compute_cp(T,p))
    conditions_sls.freestream.R                           = np.atleast_1d(ducted_fan.working_fluid.gas_specific_constant)
    conditions_sls.freestream.speed_of_sound              = np.atleast_1d(a)
    conditions_sls.freestream.velocity                    = conditions_sls.freestream.mach_number * conditions_sls.freestream.speed_of_sound
    
    # propulsion conditions
    conditions_sls.propulsion.throttle           =  np.atleast_1d(1.0)    
    
    state_sls = Data()
    state_sls.numerics = Data()
    state_sls.conditions = conditions_sls   
    results_sls = ducted_fan.evaluate_thrust(state_sls)
    
    Turboelectric_HTS_Dynamo_Ducted_Fan.sealevel_static_thrust = results_sls.thrust_force_vector[0,0] / number_of_engines

    # The shaft power is now known.
    # To size the turboelectric generators the total required powertrain power is required.
    # This is the sum of all the component s unpacked at the start.
    # Each component sizing depends on the downstream component size.
    # There are two streams: the main powertrain stream, and the cryogenic HTS rotor stream.
    # Power conditioning out of the generator is considered as part of the turboelectric component.
    # The two streams only join at this final stage.

    # Get total power required by the main powertrain stream by applying power loss of each component in sequence
    # Each component is considered as one instance, i.e. one engine
    motor_input_power           = shaft_power/(motor.motor_efficiency * motor.gearbox_efficiency)
    esc_input_power             = esc.power(motor_current, motor_input_power)
    drive_power                 = esc_input_power

    # Get power required by the cryogenic rotor stream
    # The sizing conditions here are ground level conditions as this is highest cryocooler demand
    HTS_current                 = rotor.current
    rotor_input_power           = rotor.power(HTS_current, rotor.skin_temp)


    # -------------- Current Supply Dynamo ---------------
    # Here the HTS dynamo could be sized, however for now this just calculates the heating and power used by the basic dynamo as there is no sizing required for the basic dynamo model that has constant performance
    # Get the shaft power required by the hts dynamo, and the heat load produced by the dynamo
    dynamo_powers               = hts_dynamo.shaft_power(cryo_cold_temp, HTS_current, rotor_input_power)
    dynamo_input_power          = dynamo_powers[0]
    hts_dynamo_cooling_power    = dynamo_powers[1]
    dynamo_esc_input_power      = dynamo_esc.power_in(hts_dynamo, dynamo_input_power, HTS_current)
    
    # Rename dynamo cooling requirement as lead cooling requirement in order to limit code changes compared to non-dynamo powertrain model
    leads_cooling_power         = hts_dynamo_cooling_power
    ccs_input_power             = dynamo_esc_input_power
    # -------------- Current Supply dynamo ends ----------

    # Rotor cooling power not attributable to leads or dynamo, i.e the heatflow through the cryostat, and the ohmic heating due to non-superconducting joints
    rotor_cooling_power         = rotor.outputs.cryo_load
    cooling_power               = rotor_cooling_power + leads_cooling_power  # Cryocooler must cool both rotor and supply leads
    cryocooler_input_power      = 0.0
    
    if Turboelectric_HTS_Dynamo_Ducted_Fan.cryogen_proportion < 1.0:
        cryocooler.size_cryocooler(cooling_power, cryo_cold_temp, cryo_amb_temp)
        cryocooler_input_power  = cryocooler.rated_power

    rotor_power                 = ccs_input_power + cryocooler_input_power

    # Add power required by each stream
    engine_power                = drive_power + rotor_power
    total_engine_power          = engine_power * number_of_engines

    # Size the turboelectric generator(s) based total power requirement
    turboelectric_output_power  = total_engine_power / turboelectric.number_of_engines
    initialize_from_power(turboelectric,turboelectric_output_power,conditions)

    # Pack up each component rated power into each component
    # As this will be used for sizing the mass of these components the individual power is used
    motor.rated_power           = shaft_power
    esc.rated_power             = motor_input_power
    esc.rated_current           = HTS_current
    hts_dynamo.rated_power      = dynamo_input_power
    dynamo_esc.rated_power      = dynamo_esc_input_power
    turboelectric.rated_power   = turboelectric_output_power