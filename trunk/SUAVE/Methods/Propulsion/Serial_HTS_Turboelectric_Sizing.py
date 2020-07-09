## @ingroup Methods-Propulsion
# Serial_HTS_Turboelectric_Sizing.py
# 
# Created:  K. Hamilton Mar 2020
# Modified: 
#        

""" create and evaluate a serial hybrid network that follows the power flow:
Turboelectric Generators -> Motor Drivers -> Electric Poropulsion Motors
where the electric motors have cryogenically cooled HTS rotors that follow the power flow:
Turboelectric Generators -> Current Supplies -> HTS Rotor Coils
and
Turboelectric Generators -> Cryocooler <- HTS Rotor Heat Load
There is also the capability for the HTS components to be cryogenically cooled using liquid or gaseous cryogen, howver this is not sized other than applying a factor to the cryocooler required power.
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Core import Data
from SUAVE.Methods.Power.Turboelectric.Sizing.initialize_from_power import initialize_from_power
from SUAVE.Methods.Cooling.Leads.copper_lead import initialize_copper_lead
from SUAVE.Methods.Cooling.Leads.copper_lead import Q_offdesign
from SUAVE.Methods.Cooling.Cryocooler.Sizing.size_cryocooler import size_cryocooler

## @ingroup Methods-Propulsion
def serial_hts_turboelectric_sizing(Turboelectric_HTS_Ducted_Fan,mach_number = None, altitude = None, delta_isa = 0, conditions = None, cryo_cold_temp = 50.0, cryo_amb_temp = 300.0):  
    """
    creates and evaluates a ducted_fan network based on an atmospheric sizing condition
    creates and evaluates a serial hybrid network that includes a HTS motor driven ducted fan, turboelectric generator, and the required supporting equipment including cryogenic cooling

    Inputs:
    Turboelectric_HTS_Ducted_Fan    Serial HTYS hybrid ducted fan network object (to be modified)
    mach_number
    altitude                        [meters]
    delta_isa                       temperature difference [K]
    conditions                      ordered dict object
    """
    
    # Unpack components
    ducted_fan      = Turboelectric_HTS_Ducted_Fan.ducted_fan       # Propulsion fans
    motor           = Turboelectric_HTS_Ducted_Fan.motor            # Propulsion fan motors
    turboelectric   = Turboelectric_HTS_Ducted_Fan.powersupply      # Electricity providers
    esc             = Turboelectric_HTS_Ducted_Fan.esc              # Propulsion motor speed controllers
    rotor           = Turboelectric_HTS_Ducted_Fan.rotor            # Propulsion motor HTS rotors
    current_lead    = Turboelectric_HTS_Ducted_Fan.lead             # HTS rotor current supply leads
    ccs             = Turboelectric_HTS_Ducted_Fan.ccs              # HTS rotor constant current supplies
    cryocooler      = Turboelectric_HTS_Ducted_Fan.cryocooler       # HTS rotor cryocoolers
    heat_exchanger  = Turboelectric_HTS_Ducted_Fan.heat_exchanger   # HTS rotor cryocooling via cryogen

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
    number_of_engines         = Turboelectric_HTS_Ducted_Fan.number_of_engines

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
    shaft_power                                     = fan.outputs.work_done * mass_flow
    total_shaft_power                               = shaft_power * number_of_engines
    Turboelectric_HTS_Ducted_Fan.design_shaft_power = total_shaft_power

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
    
    Turboelectric_HTS_Ducted_Fan.sealevel_static_thrust = results_sls.thrust_force_vector[0,0] / number_of_engines

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
    # initialize copper lead optimses the leads for the conditions set elsewhere, i.e. the lead is not sized here as it should be sized for the maximum ambient temperature
    initialize_copper_lead(current_lead)
    current_lead_powers         = Q_offdesign(current_lead, HTS_current)
    lead_power                  = current_lead_powers[1]
    leads_power                 = 2 * lead_power             # multiply lead loss by number of leads to get total loss
    ccs_output_power            = leads_power + rotor_input_power
    ccs_input_power             = ccs.power(HTS_current, ccs_output_power)
    # The cryogenic components are also part of the rotor power stream
    lead_cooling_power          = current_lead_powers[0]
    leads_cooling_power         = 2 * lead_cooling_power   # multiply lead cooling requirement by number of leads to get total cooling requirement
    total_lead_cooling_power    = leads_cooling_power * number_of_engines
    rotor_cooling_power         = rotor.outputs.cryo_load
    cooling_power               = rotor_cooling_power + leads_cooling_power  # Cryocooler must cool both rotor and supply leads
    cryocooler_input_power      = 0.0
    if Turboelectric_HTS_Ducted_Fan.cryogen_proportion < 1.0:
        size_cryocooler(cryocooler, cooling_power, cryo_cold_temp, cryo_amb_temp)
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
    ccs.rated_power             = ccs_output_power
    ccs.rated_current           = HTS_current
    turboelectric.rated_power   = turboelectric_output_power