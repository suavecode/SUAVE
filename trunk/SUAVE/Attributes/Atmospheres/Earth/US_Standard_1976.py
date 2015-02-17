""" US_Standard_1976.py: U.S. Standard Atmosphere (1976) """
#
#
# Modified by Tim MacDonald 2/16/15
# Converted to vector form and changed output structure

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Attributes.Gases import Air
from SUAVE.Attributes.Atmospheres import Atmosphere
from SUAVE.Attributes.Planets import Earth
from SUAVE.Core import Data
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Classes
# ----------------------------------------------------------------------

class US_Standard_1976(Atmosphere):

    """ Implements the U.S. Standard Atmosphere (1976 version)
    """
    
    def __defaults__(self):
        self.tag = ' U.S. Standard Atmosphere (1976)'

        # break point data: 
        self.fluid_properties = Air()
        self.planet = Earth()
        self.breaks = Data()
        self.breaks.altitude    = np.array(   [-2.00    , 0.00,     11.00,      20.00,      32.00,      47.00,      51.00,      71.00,      84.852]) * Units.km     # m, geopotential altitude
        self.breaks.temperature = np.array(   [301.15   , 288.15,   216.65,     216.65,     228.65,     270.65,     270.65,     214.65,     186.95])      # K
        self.breaks.pressure    = np.array(   [127774.0 , 101325.0, 22632.1,    5474.89,    868.019,    110.906,    66.9389,    3.95642,    0.3734])      # Pa
        self.breaks.density     = np.array( [1.47808e0, 1.2250e0, 3.63918e-1, 8.80349e-2, 1.32250e-2, 1.42753e-3, 8.61606e-4, 6.42099e-5, 6.95792e-6])  # kg/m^3
    
    def compute_values(self,altitude,type="all"):

        """ Computes values from the International Standard Atmosphere

        Inputs:
            altitude     : geometric altitude (elevation) (m)
                           can be a float, list or 1D array of floats
         
        Outputs:
            list of conditions -
                pressure       : static pressure (Pa)
                temperature    : static temperature (K)
                density        : density (kg/m^3)
                speed_of_sound : speed of sound (m/s)
                viscosity      : viscosity (kg/m-s)
            
        Example:
            atmosphere = SUAVE.Attributes.Atmospheres.Earth.USStandard1976()
            atmosphere.ComputeValues(1000).pressure
          
        """

        # unpack
        zs   = altitude
        gas_base  = self.fluid_properties
        grav_base = self.planet.sea_level_gravity
        Rad_base  = self.planet.mean_radius
        self.fluid_properties = Air()
        self.planet = Earth()     
        gas  = self.fluid_properties
        grav = self.planet.sea_level_gravity        
        Rad  = self.planet.mean_radius
        if (gas_base != self.fluid_properties) or (grav_base != self.planet.sea_level_gravity) or (Rad_base != Rad):
            print 'Warning: US Standard Atmosphere being used outside expected conditions'

        # convert input if necessary
        if isinstance(zs, int): 
            zs = np.array([float(zs)])
        elif isinstance(zs, float):
            zs = np.array([zs])

        # convert geometric to geopotential altitude
        zs = zs/(1 + zs/Rad)

        # initialize return data
        p = np.zeros(np.size(zs))
        T = np.zeros(np.size(zs))
        rho = np.zeros(np.size(zs))
        a = np.zeros(np.size(zs))
        mew = np.zeros(np.size(zs))
        z0 = np.zeros(np.size(zs))
        T0 = np.zeros(np.size(zs))
        p0 = np.zeros(np.size(zs))
        alpha = np.zeros(np.size(zs))
        
        # evaluate at each altitude
        zmin = self.breaks.altitude[0]
        zmax = self.breaks.altitude[-1]
        
        if np.amin(zs) < zmin:
            print "Warning: altitude requested below minimum for this atmospheric model; returning values for h = -2.0 km"
            zs[zs < zmin] = zmin
        if np.amax(zs) > zmax:
            print "Warning: altitude requested above maximum for this atmospheric model; returning values for h = 86.0 km"   
            zs[zs > zmax] = zmax
        for i in range(len(self.breaks.altitude)-1): # this uses >= and <= to capture both edges and because values should be the same at the edges
            z0[(zs>=self.breaks.altitude[i]) & (zs <= self.breaks.altitude[i+1])] = self.breaks.altitude[i]
            T0[(zs>=self.breaks.altitude[i]) & (zs <= self.breaks.altitude[i+1])] = self.breaks.temperature[i]
            p0[(zs>=self.breaks.altitude[i]) & (zs <= self.breaks.altitude[i+1])] = self.breaks.pressure[i]
            alpha[(zs>=self.breaks.altitude[i]) & (zs <= self.breaks.altitude[i+1])] = -(self.breaks.temperature[i+1] - self.breaks.temperature[i])/ \
                                                                                        (self.breaks.altitude[i+1] - self.breaks.altitude[i])
        dz = zs-z0
        p[alpha == 0.0] = p0[alpha == 0.0]*np.exp(-1*dz[alpha == 0.0]*grav/(gas.gas_specific_constant*T0[alpha == 0.0]))
        p[alpha != 0.0] = p0[alpha != 0.0]*((1 - alpha[alpha != 0.0]*dz[alpha != 0.0]/T0[alpha != 0.0])**(1*grav/(alpha[alpha != 0.0]*gas.gas_specific_constant)))
        T = T0 - dz*alpha
        rho = self.fluid_properties.compute_density(T,p)
        a = self.fluid_properties.compute_speed_of_sound(T)
        mew = self.fluid_properties.compute_absolute_viscosity(T)
        
        #p, T, rho, a, mew
        
        conditions = Data()
        conditions.freestream = Data()
        conditions.freestream.pressure = p
        conditions.freestream.temperature = T
        conditions.freestream.density = rho
        conditions.freestream.speed_of_sound = a
        conditions.freestream.kinematic_viscosity = mew
        
        return conditions


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
if __name__ == '__main__':
    
    import pylab as plt
    
    h = np.linspace(-1.,60.,200) * Units.km
    
    atmosphere = US_Standard_1976()
    
    p, T, rho, a, mew = atmosphere.compute_values(h)
    
    plt.figure(1)
    plt.plot(p,h)
    plt.xlabel('Pressure (Pa)')
    plt.ylabel('Altitude (km)')
    
    plt.figure(2)
    plt.plot(T,h)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Altitude (km)')    
    
    plt.figure(3)
    plt.plot(rho,h)
    plt.xlabel('Density (kg/m^3)')
    plt.ylabel('Altitude (km)')       
    
    plt.figure(4)
    plt.plot(a,h)
    plt.xlabel('Speed of Sound (m/s)')
    plt.ylabel('Altitude (km)') 
    
    plt.figure(6)
    plt.plot(mew,h)
    plt.xlabel('Viscosity (kg/m-s)')
    plt.ylabel('Altitude (km)')   

    plt.show()