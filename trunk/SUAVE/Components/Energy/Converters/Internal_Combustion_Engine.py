# Internal_Combustion_Engine.py
#
# Created:  Aug, 2016: D. Bianchi

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE
from SUAVE.Core import Data

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
        atmo_values = atmo.compute_values(altitude,delta_isa)
        p   = atmo_values.pressure[0,0]
        T   = atmo_values.temperature[0,0]
        rho = atmo_values.density[0,0]
        a   = atmo_values.speed_of_sound[0,0]
        mu  = atmo_values.dynamic_viscosity[0,0]
        PSLS     = self.sea_level_power
        h_flat   = self.flat_rate_altitude

        # computing the sea-level ISA atmosphere conditions
        atmo_values = atmo.compute_values(0,0)
        p0   = atmo_values.pressure[0,0]
        T0   = atmo_values.temperature[0,0]
        rho0 = atmo_values.density[0,0]
        a0   = atmo_values.speed_of_sound[0,0]
        mu0  = atmo_values.dynamic_viscosity[0,0]

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
        outputs = Data()
        outputs.power = Pavailable
        outputs.power_specific_fuel_consumption = BSFC
        outputs.fuel_flow_rate = fuel_flow_rate

        return outputs

if __name__ == '__main__':

    import numpy as np
    import pylab as plt
    import SUAVE
    from SUAVE.Core import Units, Data
    conditions = Data()
    atmo = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    ICE = SUAVE.Components.Energy.Converters.Internal_Combustion_Engine()
    PSLS = 1.0
    delta_isa = 0.0
    i = 0
    altitude = list()
    rho = list()
    sigma = list()
    Pavailable = list()
    for h in range(0,25000,500):
        altitude.append(h * 0.3048)
        atmo_values = atmo.compute_values(altitude[i],delta_isa)
        rho.append(atmo_values.density[0,0])
        sigma.append(rho[i] / 1.225)
##        Pavailable.append(PSLS * (sigma[i] - 0.117) / 0.883)
        conditions.altitude = altitude[i]
        conditions.delta_isa = delta_isa
        ICE.sea_level_power = 1.0
        out = ICE.power(conditions)
        Pavailable.append(out.power)
        i += 1
    line_style='bo-'
    fig = plt.figure("Pavailable vs altitude")
    axes = plt.gca()
    axes.plot(np.multiply(altitude,1./Units.ft), Pavailable, line_style)
    axes.set_xlabel('Altitude [ft]')
    axes.set_ylabel('Output power [bhp]')
    axes.grid(True)
    plt.show()

