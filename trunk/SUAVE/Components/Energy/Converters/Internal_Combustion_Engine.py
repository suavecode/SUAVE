## @ingroup Components-Energy-Converters
# Internal_Combustion_Engine.py
#
# Created:  Aug, 2016: D. Bianchi
# Modified  Feb, 2020: M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE
from SUAVE.Core import Data, Units

# package imports
import numpy as np
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Internal Combustion Engine Class
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
class Internal_Combustion_Engine(Energy_Component):
    """This is an internal combustion engine component.
    
    Assumptions:
    None

    Source:
    None
    """           
    def __defaults__(self):

        self.sea_level_power                 = 0.0
        self.flat_rate_altitude              = 0.0
        self.rated_speed                     = 0.0
        self.inputs.speed                    = 0.0
        self.power_specific_fuel_consumption = 0.36 # lb/hr/hp :: Ref: Table 5.1, Modern diesel engines, Saeed Farokhi, Aircraft Propulsion (2014)

    def power(self,conditions):
        """ The internal combustion engine output power and specific power consumption
        Inputs:
            Engine:
                sea-level power
                flat rate altitude
                rated_speed (RPM)
                throttle setting
                inputs.speed (RPM)
            Freestream conditions:
                altitude
                delta_isa
        Outputs:
            Brake power (or Shaft power)
            Power (brake) specific fuel consumption
            Fuel flow
            Torque
        """

        # Unpack
        altitude                         = conditions.freestream.altitude
        delta_isa                        = conditions.freestream.delta_ISA
        throttle                         = conditions.propulsion.combustion_engine_throttle
        PSLS                             = self.sea_level_power
        h_flat                           = self.flat_rate_altitude
        speed                            = self.inputs.speed
        power_specific_fuel_consumption  = self.power_specific_fuel_consumption


        altitude_virtual = altitude - h_flat       # shift in power lapse due to flat rate
        altitude_virtual[altitude_virtual<0.] = 0. # don't go below sea level
        
        atmo             = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        atmo_values      = atmo.compute_values(altitude_virtual,delta_isa)
        p                = atmo_values.pressure
        T                = atmo_values.temperature
        rho              = atmo_values.density
        a                = atmo_values.speed_of_sound
        mu               = atmo_values.dynamic_viscosity

        # computing the sea-level ISA atmosphere conditions
        atmo_values = atmo.compute_values(0,0)
        p0          = atmo_values.pressure[0,0]
        T0          = atmo_values.temperature[0,0]
        rho0        = atmo_values.density[0,0]
        a0          = atmo_values.speed_of_sound[0,0]
        mu0         = atmo_values.dynamic_viscosity[0,0]

        # calculating the density ratio:
        sigma = rho / rho0

        # calculating available power based on Gagg and Ferrar model (ref: S. Gudmundsson, 2014 - eq. 7-16)
        Pavailable                    = PSLS * (sigma - 0.117) / 0.883        
        Pavailable[h_flat > altitude] = PSLS

        # applying throttle setting
        output_power                  = Pavailable * throttle 
        output_power[output_power<0.] = 0. 
        SFC                           = power_specific_fuel_consumption * Units['lb/hp/hr']

        #fuel flow rate
        a               = np.zeros_like(altitude)
        fuel_flow_rate  = np.fmax(output_power*SFC,a)

        #torque
        torque = output_power/speed
        
        # store to outputs
        self.outputs.power                           = output_power
        self.outputs.power_specific_fuel_consumption = power_specific_fuel_consumption
        self.outputs.fuel_flow_rate                  = fuel_flow_rate
        self.outputs.torque                          = torque

        return self.outputs
    
    
    def calculate_throttle(self,conditions):
        """ The internal combustion engine output power and specific power consumption
        
        source: 
        
        Inputs:
            Engine:
                sea-level power
                flat rate altitude
                rated_speed (RPM)
                throttle setting
                inputs.power
            Freestream conditions:
                altitude
                delta_isa
        Outputs:
            Brake power (or Shaft power)
            Power (brake) specific fuel consumption
            Fuel flow
            Torque
            throttle setting
        """

        # Unpack
        altitude                         = conditions.freestream.altitude
        delta_isa                        = conditions.freestream.delta_ISA
        PSLS                             = self.sea_level_power
        h_flat                           = self.flat_rate_altitude
        power_specific_fuel_consumption  = self.power_specific_fuel_consumption
        output_power                     = self.inputs.power*1.0


        altitude_virtual = altitude - h_flat       # shift in power lapse due to flat rate
        altitude_virtual[altitude_virtual<0.] = 0. # don't go below sea level
        
        atmo             = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        atmo_values      = atmo.compute_values(altitude_virtual,delta_isa)
        p                = atmo_values.pressure
        T                = atmo_values.temperature
        rho              = atmo_values.density
        a                = atmo_values.speed_of_sound
        mu               = atmo_values.dynamic_viscosity

        # computing the sea-level ISA atmosphere conditions
        atmo_values = atmo.compute_values(0,0)
        p0          = atmo_values.pressure[0,0]
        T0          = atmo_values.temperature[0,0]
        rho0        = atmo_values.density[0,0]
        a0          = atmo_values.speed_of_sound[0,0]
        mu0         = atmo_values.dynamic_viscosity[0,0]

        # calculating the density ratio:
        sigma = rho / rho0

        # calculating available power based on Gagg and Ferrar model (ref: S. Gudmundsson, 2014 - eq. 7-16)
        Pavailable                    = PSLS * (sigma - 0.117) / 0.883        
        Pavailable[h_flat > altitude] = PSLS


        # applying throttle setting
        throttle = output_power/Pavailable 
        output_power[output_power<0.] = 0. 
        SFC                           = power_specific_fuel_consumption * Units['lb/hp/hr']

        #fuel flow rate
        a               = np.zeros_like(altitude)
        fuel_flow_rate  = np.fmax(output_power*SFC,a)

        
        # store to outputs
        self.outputs.power_specific_fuel_consumption = power_specific_fuel_consumption
        self.outputs.fuel_flow_rate                  = fuel_flow_rate
        self.outputs.throttle                        = throttle

        return self.outputs    

