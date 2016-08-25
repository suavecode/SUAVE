# Internal_Combustion_Engine.py
#
# Created:  Aug, 2016: D. Bianchi

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

class Internal_Combustion_Engine(Energy_Component):

    def __defaults__(self):

        self.sea_level_power    = 0.0
        self.flat_rate_altitude = 0.0
        self.speed              = 0.0

    def power(self,conditions):
        """ The internal combustion engine output power and specific power consumption

        Inputs:
            Engine:
                sea-level power
                flat rate altitude
                speed (RPM)
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
        altitude  = conditions.altitude
        delta_isa = conditions.delta_isa
        PSLS      = self.sea_level_power
        h_flat    = self.flat_rate_altitude
        rpm       = self.speed

        altitude_virtual = altitude - h_flat # shift in power lapse due to flat rate
        atmo = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        atmo_values = atmo.compute_values(altitude_virtual,delta_isa)
        p   = atmo_values.pressure[0,0]
        T   = atmo_values.temperature[0,0]
        rho = atmo_values.density[0,0]
        a   = atmo_values.speed_of_sound[0,0]
        mu  = atmo_values.dynamic_viscosity[0,0]

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

        #torque
        ## SHP = torque * 2*pi * RPM / 33000        (UK units)
        torque = ( 5252. * Pavailable / rpm ) / (Units.ft * Units.lbf)

        # store to outputs
        outputs = Data()
        outputs.power                           = Pavailable
        outputs.power_specific_fuel_consumption = BSFC
        outputs.fuel_flow_rate                  = fuel_flow_rate
        outputs.torque                          = torque

        return outputs

if __name__ == '__main__':

    import numpy as np
    import pylab as plt
    import SUAVE
    from SUAVE.Core import Units, Data
    conditions = Data()
    atmo = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    ICE = SUAVE.Components.Energy.Converters.Internal_Combustion_Engine()
    ICE.sea_level_power = 250.0 * Units.horsepower
    ICE.flat_rate_altitude = 5000. * Units.ft
    ICE.speed = 2200. # rpm
    PSLS = 1.0
    delta_isa = 0.0
    i = 0
    altitude = list()
    rho = list()
    sigma = list()
    Pavailable = list()
    torque = list()
    for h in range(0,25000,500):
        altitude.append(h * 0.3048)
        atmo_values = atmo.compute_values(altitude[i],delta_isa)
        rho.append(atmo_values.density[0,0])
        sigma.append(rho[i] / 1.225)
##        Pavailable.append(PSLS * (sigma[i] - 0.117) / 0.883)
        conditions.altitude = altitude[i]
        conditions.delta_isa = delta_isa
        out = ICE.power(conditions)
        Pavailable.append(out.power)
        torque.append(out.torque)
        i += 1
    fig = plt.figure("Power and Torque vs altitude")
    axes = fig.add_subplot(2,1,1)
    axes.plot(np.multiply(altitude,1./Units.ft), np.multiply(Pavailable,1./Units.horsepower), 'bo-')
    axes.set_xlabel('Altitude [ft]')
    axes.set_ylabel('Output power [bhp]')
    axes.grid(True)

    axes = fig.add_subplot(2,1,2)
    axes.plot(np.multiply(altitude,1./Units.ft), np.multiply(torque,1./(Units.ft * Units.lbf)), 'rs-')
    axes.set_xlabel('Altitude [ft]')
    axes.set_ylabel('Torque [lbf*ft]')
    axes.grid(True)
    plt.show()

