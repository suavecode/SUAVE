""" US_Standard_1976.py: U.S. Standard Atmosphere (1976) """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Attributes.Gases import Air
from SUAVE.Attributes.Atmospheres import Atmosphere
from SUAVE.Attributes.Planets import Earth
from SUAVE.Structure import Data

# ----------------------------------------------------------------------
#  Classes
# ----------------------------------------------------------------------

class US_Standard_1976(Atmosphere):

    """ Implements the U.S. Standard Atmosphere (1976 version)
    """
    
    def __defaults__(self):
        self.tag = ' U.S. Standard Atmosphere (1976)'

        # break point data: 
        self.gas = Air()
        self.planet = Earth()
        self.z_breaks = np.array(   [-2.00,     0.00,     11.00,      20.00,      32.00,      47.00,      51.00,      71.00,      84.852])      # km, geopotential altitude
        self.T_breaks = np.array(   [301.15,    288.15,   216.65,     216.65,     228.65,     270.65,     270.65,     214.65,     186.95])      # K
        self.p_breaks = np.array(   [127774.0,  101325.0, 22632.1,    5474.89,    868.019,    110.906,    66.9389,    3.95642,    0.3734])      # Pa
        self.rho_breaks = np.array( [1.47808e0, 1.2250e0, 3.63918e-1, 8.80349e-2, 1.32250e-2, 1.42753e-3, 8.61606e-4, 6.42099e-5, 6.95792e-6])  # kg/m^3
    
    def compute_values(self,altitude,type="all"):

        """ Computes values from the International Standard Atmosphere

        Inputs:
            altitude     : geometric altitude (elevation) (km)
                           can be a float, list or 1D array of floats
         
        Outputs:
            Data() with fields -
            pressure     : static pressure (Pa)
            temperature  : static temperature (K)
            density      : density (kg/m^3)
            speedofsound : speed of sound (m/s)
            viscosity    : viscosity (kg/m-s)
            
        Example:
            atmosphere = SUAVE.Attributes.Atmospheres.Earth.USStandard1976()
            atmosphere.ComputeValues(1000).pressure
          
        """

        # unpack
        zs   = altitude
        gas  = self.gas
        grav = self.planet.sea_level_gravity
        Rad  = self.planet.mean_radius

        # return options
        all_vars = ["all", "everything"]
        pressure = ["p", "pressure"]
        temp = ["t", "temp", "temperature"]
        density = ["rho", "density"]
        speedofsound = ["speedofsound", "a"]
        viscosity = ["viscosity", "mew"]

        # some up-front logic for outputs based on thermo
        need_p = True; need_T = True; need_rho = True; need_a = True; need_mew = True;
        if type.lower() in pressure:
            need_T = False; need_rho = False; need_a = False; need_mew = False
        elif type.lower() in temp:
            need_p = False; need_rho = False; need_a = False; need_mew = False
        elif type.lower() in density:
            need_a = False; need_mew = False
        elif type.lower() in speedofsound:
            need_p = False; need_rho = False; need_mew = False
        elif type.lower() in viscosity:
            need_p = False; need_rho = False; need_a = False

        # convert input if necessary
        if isinstance(zs, int): 
            zs = np.array([float(zs)])
        elif isinstance(zs, float):
            zs = np.array([zs])

        # convert geometric to geopotential altitude
        zs = zs/(1 + zs/Rad)

        # initialize return data
        p = np.array([]); T = np.array([]); rho = np.array([]); a = np.array([]); mew = np.array([]);
        
        # evaluate at each altitude
        for z in zs:
            
            # check altitude range
            too_low = False; too_high = False
            if (z < self.z_breaks[0]):
                too_low = True
                print "Warning: altitude requested below minimum for this atmospheric model; returning values for h = -2.0 km"
            elif (z > self.z_breaks[-1]):
                too_high = True
                print "Warning: altitude requested above maximum for this atmospheric model; returning values for h = 86.0 km"
            else:
                # loop through breaks
                for i in range(len(self.z_breaks)):
                    if z >= self.z_breaks[i] and z < self.z_breaks[i+1]:
                        z0 = self.z_breaks[i]; T0 = self.T_breaks[i]; p0 = self.p_breaks[i]
                        alpha = -(self.T_breaks[i+1] - self.T_breaks[i])/ \
                            (self.z_breaks[i+1] - self.z_breaks[i])     # lapse rate K/km
                        dz = z - z0
                        break

            # pressure
            if need_p:
                if too_low:
                    pz = self.p_breaks[0]
                elif too_high:
                    pz = self.p_breaks[-1]
                else:
                    if alpha == 0.0:
                        pz = p0*np.exp(-1000*dz*grav/(gas.R*T0))
                    else:
                        pz = p0*((1 - alpha*dz/T0)**(1000*grav/(alpha*gas.R)))
                p = np.append(p,pz)

            # temperature
            if need_T:
                if too_low:
                    Tz = self.T_breaks[0]
                elif too_high:
                    Tz = self.T_breaks[-1]
                else:
                    Tz = T0 - dz*alpha      # note: alpha = lapse rate (negative)
                T = np.append(T,Tz)

            if need_rho:
                if too_low:
                    rho = np.append(rho,self.rho_breaks[0])
                elif too_high:
                    rho = np.append(rho,self.rho_breaks[-1])
                else:
                    rho = np.append(rho,self.gas.compute_density(Tz,pz))
            
            if need_a:
                a = np.append(a,self.gas.compute_speed_of_sound(Tz))

            if need_mew:
                mew = np.append(mew,self.gas.compute_absolute_viscosity(Tz))

        # for each altitude

        # return requested data
        if type.lower() in all_vars:
            return (p, T, rho, a, mew)

        if type.lower() in pressure:
            return p
                      
        elif type.lower() in temp:
            return T

        elif type.lower() in density:
            return rho
            
        elif type.lower() in speedofsound:
            return a

        elif type.lower() in viscosity:
            return mew

        else:
            print "Unknown data, " + type + ", request; no data returned."
            return []

# ----------------------------------------------------------------------
