# Internal_Combustion_Engine.py
#
# Created:  Aug, 2016: D. Bianchi

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Internal Combustion Engine Class
# ----------------------------------------------------------------------

class Internal_Combustion_Engine(Energy_Component):

    def __defaults__(self):

        self.sea_level_power    = 0.0
        self.flat_rate_altitude = 0.0

    def power(self,conditions):
        """ The internal combustion engine output power and specific power consumption

        Inputs:
            Engine sea-level power
            Freestream conditions: altitude and delta_isa
        Outputs:
            Available power
            Power specific fuel consumption
            Fuel flow

        """

        # Unpack
        altitude = conditions.altitude
        delta_isa = conditions.delta_isa
        atmo = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        atmo_values.compute_values(altitude,delta_isa)
        p   = atmo_values.pressure
        T   = atmo_values.temperature
        rho = atmo_values.density
        a   = atmo_values.speed_of_sound
        mu  = atmo_values.dynamic_viscosity
        PSLS     = self.sea_level_power
        h_flat   = self.flat_rate_altitude

        # computing the sea-level ISA atmosphere conditions
        atmo_values = atmo.compute_values(0,0)
        p0   = atmo_values.pressure
        T0   = atmo_values.temperature
        rho0 = atmo_values.density
        a0   = atmo_values.speed_of_sound
        mu0  = atmo_values.dynamic_viscosity

        # calculating the density ratio:
        sigma = rho / rho0

        if h_flat > altitude:
            Pavailable = PSLS
        else:
            # calculating available power based on Gagg and Ferrar model (ref: S. Gudmundsson, 2014 - eq. 7-16)
            Pavailable = PSLS * (sigma - 0.117) / 0.883

        # brake-specific fuel consumption <--- now considering it as a constant typical value
        BSFC = 0.36 # lb/hr/hp :: Ref: Table 5.1, Modern diesel engines, Saeed Farokhi, Aircraft Propulsion (2014)

        #fuel flow rate
        a = np.array([0.])
        fuel_flow_rate   = np.fmax(Pavailable*BSFC/(3600 * 9.80665),a)

        # store to outputs
        self.outputs.power = Pavailable
        self.outputs.power_specific_fuel_consumption = BSFC
        self.outputs.fuel_flow_rate = fuel_flow_rate

        return outputs
