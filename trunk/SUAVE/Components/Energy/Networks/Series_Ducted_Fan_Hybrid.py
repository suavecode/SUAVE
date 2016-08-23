# Series Ducted Fan Hybrid.py
# 
# Created:  Aug 2016, E. Botero & D. Bianchi
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
from SUAVE.Components.Propulsors.Propulsor import Propulsor

from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------
class Series_Ducted_Fan_Hybrid(Propulsor):
    def __defaults__(self): 
        self.motor             = None
        self.propeller         = None
        self.esc               = None
        self.avionics          = None
        self.payload           = None
        self.battery           = None
        self.nacelle_diameter  = None
        self.engine_length     = None
        self.number_of_engines = None
        self.voltage           = None
        self.thrust_angle      = 0.0
        self.tag               = 'network'
        self.areas             = Data()
    
    # manage process with a driver function
    def evaluate_thrust(self,state):
    
        # unpack
        conditions = state.conditions
        numerics   = state.numerics
        motor      = self.motor
        propeller  = self.propeller
        esc        = self.esc
        avionics   = self.avionics
        payload    = self.payload
        battery    = self.battery
        
        # Set battery energy
        battery.current_energy = conditions.propulsion.battery_energy  

        # Step 1 battery power
        #esc.inputs.voltagein = state.unknowns.battery_voltage_under_load
        esc.inputs.voltagein = self.voltage
        # Step 2
        esc.voltageout(conditions)
        # link
        motor.inputs.voltage = esc.outputs.voltageout 
        # step 3
        motor.omega(conditions)
        
        # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
        eta        = conditions.propulsion.throttle[:,0,None]
        P[eta>1.0] = P[eta>1.0]*eta[eta>1.0]
        F[eta>1.0] = F[eta>1.0]*eta[eta>1.0]

        # Run the avionics
        avionics.power()

        # Run the payload
        payload.power()
        
        # Run the motor for current
        motor.current(conditions)
        # link
        esc.inputs.currentout =  motor.outputs.current
        
        # Run the esc
        esc.currentin()

        # Calculate avionics and payload power
        avionics_payload_power = avionics.outputs.power + payload.outputs.power

        # Calculate avionics and payload current
        avionics_payload_current = avionics_payload_power/self.voltage
        
        mdotc = state.unknowns.corrected_mass_flow_rate
        pic   = state.unknowns.pressure_ratio
        
        P0 = conditions.freestream.pressure
        T0 = conditions.freestream.temperature
        u0 = conditions.freestream.velocity
        gamma = conditions.freestream.gamma
        Cp    = conditions.freestream.specific_heat
        R     = conditions.freestream.gas_specific_constant
        Tref  = self.reference_temperature
        Pref  = self.reference_temperature
    
        # Ram calculations
        Dh = .5*u0*u0
        h0  = Cp*T0
    
        ram = self.ram
        ram.inputs.total_temperature = T0
        ram.inputs.total_pressure    = P0
        ram.inputs.total_enthalpy    = h0
        ram.inputs.delta_enthalpy    = Dh
        ram.inputs.working_fluid.specific_heat     = Cp
        ram.inputs.working_fluid.gamma             = gamma
        ram.compute_flow()
    
        Tt0 = ram.outputs.total_temperature
        Pt0 = ram.outputs.total_pressure
        ht0 = ram.outputs.total_enthalpy        
        
        # Inlet Nozzle
    
        inlet_nozzle = self.inlet_nozzle
        inlet_nozzle.inputs.total_temperature = Tt0
        inlet_nozzle.inputs.total_pressure    = Pt0
        inlet_nozzle.inputs.total_enthalpy    = ht0
        inlet_nozzle.compute_flow()
    
        Tt2 = inlet_nozzle.outputs.total_temperature
        Pt2 = inlet_nozzle.outputs.total_pressure
        ht2 = inlet_nozzle.outputs.total_enthalpy        
        
        # Fan

        fan = self.fan
        fan.corrected_mass_flow   = mdotc
        fan.pressure_ratio        = pic
        fan.compute_performance()
        eta_f = fan.polytropic_efficiency
        Nfc   = fan.corrected_speed
        
        Nf   = Nfc*np.sqrt(Tt2/Tref)
        mdot = mdotc*(Pt2/Pref)/(Tt2/Tref)

        fan.inputs.working_fluid.specific_heat = Cp
        fan.inputs.working_fluid.gamma         = gamma
        fan.inputs.working_fluid.R             = R
        fan.inputs.total_temperature           = Tt2
        fan.inputs.total_pressure              = Pt2
        fan.inputs.total_enthalpy              = ht2
        fan.compute()

        Tt2_1 = fan.outputs.total_temperature
        Pt2_1 = fan.outputs.total_pressure
        ht2_1 = fan.outputs.total_enthalpy
        
        P_fan = (ht2_1-ht2)*mdot/eta_f # power required
        Q_fan = P_fan/Nf               # torque required
        conditions.propulsion.fan_torque = Q_fan
        
        motor.inputs.torque = Q_fan
        motor.inputs.omega  = Nf
        motor.voltage_current(conditions)
        
        # torque is automatically matched?
        state.conditions.propulsion.motor_torque           = Q_fan
        state.conditions.propulsion.motor_voltage_required = motor.outputs.voltage   
        

        # Fan Nozzle

        fan_nozzle = self.fan_nozzle
        fan_nozzle.inputs.working_fluid.specific_heat = Cp
        fan_nozzle.inputs.working_fluid.gamma         = gamma
        fan_nozzle.inputs.working_fluid.R             = R
        fan_nozzle.inputs.total_temperature = Tt2_1
        fan_nozzle.inputs.total_pressure    = Pt2_1
        fan_nozzle.inputs.total_enthalpy    = ht2_1

        fan_nozzle.compute()

        Tt7 = fan_nozzle.outputs.total_temperature
        Pt7 = fan_nozzle.outputs.total_pressure
        ht7 = fan_nozzle.outputs.total_enthalpy     
        
        
        # Fan Exhaust
    
        fan_exhaust = self.fan_exhaust
        fan_exhaust.pressure_ratio = P0/Pt7
        fan_exhaust.inputs.working_fluid.specific_heat = Cp
        fan_exhaust.inputs.working_fluid.gamma         = gamma
        fan_exhaust.inputs.working_fluid.R             = R
        fan_exhaust.inputs.total_temperature = Tt7
        fan_exhaust.inputs.total_pressure    = Pt7
        fan_exhaust.inputs.total_enthalpy    = ht7
    
        fan_exhaust.compute()
    
        T8 = fan_exhaust.outputs.static_temperature
        u8 = fan_exhaust.outputs.flow_speed        
        
        # Compute Thrust
        
        f = 0 # to be changed to account for generator
        thrust = self.thrust
        thrust.inputs.normalized_fuel_flow_rate = f
        thrust.inputs.fan_exhaust_flow_speed    = u8
    
        conditions.freestream.speed_of_sound = a0
        conditions.freestream.velocity       = u0
        thrust.compute(conditions)        
        
        Fsp = thrust.outputs.specific_thrust
        
        F  = Fsp*mdot*a0

        # link
        battery.inputs.current  = esc.outputs.currentin*self.number_of_engines + avionics_payload_current
        battery.inputs.power_in = -(esc.outputs.voltageout*esc.outputs.currentin*self.number_of_engines + avionics_payload_power)
        battery.energy_calc(numerics)        
        
        
        # Create the outputs
        F    = self.number_of_engines * F * [np.cos(self.thrust_angle),0,-np.sin(self.thrust_angle)]      
        mdot = np.zeros_like(F)

        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot
        
        
        return results
    
    def size(self,state,mach_number,altitude,delta_isa = 0.):
        
        conditions = state.conditions
        
        P0 = conditions.freestream.pressure
        T0 = conditions.freestream.temperature
        u0 = conditions.freestream.velocity
        gamma = conditions.freestream.gamma
        Cp    = conditions.freestream.specific_heat
        R     = conditions.freestream.gas_specific_constant
        Tref  = self.reference_temperature
        Pref  = self.reference_temperature
    
        # Ram calculations
        Dh = .5*u0*u0
        h0  = Cp*T0
    
        ram = self.ram
        ram.inputs.total_temperature = T0
        ram.inputs.total_pressure    = P0
        ram.inputs.total_enthalpy    = h0
        ram.inputs.delta_enthalpy    = Dh
        ram.inputs.working_fluid.specific_heat     = Cp
        ram.inputs.working_fluid.gamma             = gamma
        ram.compute_flow()
    
        Tt0 = ram.outputs.total_temperature
        Pt0 = ram.outputs.total_pressure
        ht0 = ram.outputs.total_enthalpy        
        
        # Inlet Nozzle
    
        inlet_nozzle = self.inlet_nozzle
        inlet_nozzle.inputs.total_temperature = Tt0
        inlet_nozzle.inputs.total_pressure    = Pt0
        inlet_nozzle.inputs.total_enthalpy    = ht0
        inlet_nozzle.compute_flow()
    
        Tt2 = inlet_nozzle.outputs.total_temperature
        Pt2 = inlet_nozzle.outputs.total_pressure
        ht2 = inlet_nozzle.outputs.total_enthalpy        
        
        # Fan

        fan = self.fan
        fan.set_design_condition()

        fan.inputs.working_fluid.specific_heat = Cp
        fan.inputs.working_fluid.gamma         = gamma
        fan.inputs.working_fluid.R             = R
        fan.inputs.total_temperature           = Tt2
        fan.inputs.total_pressure              = Pt2
        fan.inputs.total_enthalpy              = ht2
        fan.compute()

        Tt2_1 = fan.outputs.total_temperature
        Pt2_1 = fan.outputs.total_pressure
        ht2_1 = fan.outputs.total_enthalpy
        

        # Fan Nozzle

        fan_nozzle = self.fan_nozzle
        fan_nozzle.inputs.working_fluid.specific_heat = Cp
        fan_nozzle.inputs.working_fluid.gamma         = gamma
        fan_nozzle.inputs.working_fluid.R             = R
        fan_nozzle.inputs.total_temperature = Tt2_1
        fan_nozzle.inputs.total_pressure    = Pt2_1
        fan_nozzle.inputs.total_enthalpy    = ht2_1

        fan_nozzle.compute()

        Tt7 = fan_nozzle.outputs.total_temperature
        Pt7 = fan_nozzle.outputs.total_pressure
        ht7 = fan_nozzle.outputs.total_enthalpy     
        
        
        # Fan Exhaust
    
        fan_exhaust = self.fan_exhaust
        fan_exhaust.pressure_ratio = P0/Pt7
        fan_exhaust.inputs.working_fluid.specific_heat = Cp
        fan_exhaust.inputs.working_fluid.gamma         = gamma
        fan_exhaust.inputs.working_fluid.R             = R
        fan_exhaust.inputs.total_temperature = Tt7
        fan_exhaust.inputs.total_pressure    = Pt7
        fan_exhaust.inputs.total_enthalpy    = ht7
    
        fan_exhaust.compute()
    
        T8 = fan_exhaust.outputs.static_temperature
        u8 = fan_exhaust.outputs.flow_speed        
        
        # Compute Thrust
        
        f = 0 # to be changed to account for generator
        thrust = self.thrust
        thrust.inputs.normalized_fuel_flow_rate = f
        thrust.inputs.fan_exhaust_flow_speed    = u8
    
        conditions.freestream.speed_of_sound = a0
        conditions.freestream.velocity       = u0
        thrust.compute(conditions)        
        
        Fsp = thrust.outputs.specific_thrust   
        
        # Determine design thrust per engine
        FD = self.thrust.total_design/(self.number_of_engines)
    
        # Mass Flow Calculation
        mdot_design = FD/(Fsp*a0)
        mdotc_design = mdot_design/(Pt2/Pref)*(Tt2/Tref)
        fan.efficiency_map.design_mass_flow = mdotc_design
        fan.speed_map.design_mass_flow      = mdotc_design
    
        # Fan Sizing
    
        #fan.size(mdot_design,ep.M2)
        #A2 = fan.entrance_area
    
        ## Fan Nozzle Area
    
        #fan_nozzle.size(mdot_design,u8,T8,P0)
        #A7 = fan_nozzle.exit_area
    
    
    def unpack_unknowns(self,segment,state):
        """"""        
        
        # Here we are going to unpack the unknowns (Cp) provided for this network
        state.conditions.propulsion.compressor.corrected_mass_flow_rate = state.unknowns.corrected_mass_flow_ratio
        state.conditions.propulsion.compressor.pressure_ratio           = state.unknowns.pressure_ratio
        
        return
    
    def residuals(self,segment,state):
        """"""        
        
        # Here we are going to pack the residuals (torque,voltage) from the network
        
        # Unpack
        q_motor     = state.conditions.propulsion.motor_torque
        q_fan       = state.conditions.propulsion.fan_torque
        v_required  = state.conditions.propulsion.motor_voltage_required
        v_specified = state.conditions.propulsion.motor_voltage_specified
        
        # Return the residuals
        state.residuals.network[:,0] = q_motor[:,0] - q_prop[:,0]
        state.residuals.network[:,1] = v_required[:,0] - v_specified[:,0]
        
        return    
            
    __call__ = evaluate_thrust
