#Solar_Flux.py
# 
# Created:  Emilio Botero, Jun 2014
# Modified:  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import scipy as sp
import datetime
import time
from SUAVE.Attributes import Units
from SUAVE.Components.Energy.Energy_Component import Energy_Component

from SUAVE.Structure import (
Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#  Solar Class
# ----------------------------------------------------------------------
class Solar_Radiation(Energy_Component):

    def solar_radiation(self,conditions):  
        
        """ Computes the adjusted solar flux in watts per square meter.
              
              Inputs:
                  day - day of the year from Jan 1st
                  TUTC - time in seconds in UTC
                  longitude- in degrees
                  latitude - in degrees
                  altitude - in meters                  
                  bank angle - in radians
                  pitch attitude - in radians
                  heading angle - in radians
                  
              Outputs:
                  sflux - adjusted solar flux
                  
              Assumptions:
                  Solar intensity =1305 W/m^2
                  Includes a diffuse component of 0% of the direct component
                  Altitudes are not excessive 
        """        
        
        # Unpack
        timedate  = conditions.frames.planet.time_date
        latitude  = conditions.frames.planet.latitude
        longitude = conditions.frames.planet.longitude
        phip      = conditions.frames.body.inertial_rotations[:,0]
        thetap    = conditions.frames.body.inertial_rotations[:,1]
        psip      = conditions.frames.body.inertial_rotations[:,2]
        altitude  = conditions.freestream.altitude
        times     = conditions.frames.inertial.time
        
        # Figure out the date and time
        day       = timedate.tm_yday + np.floor_divide(times, 24.*60.*60.)
        TUTC      = timedate.tm_sec + 60.*timedate.tm_min+ 60.*60.*timedate.tm_hour + np.mod(times,24.*60.*60.)
        
        # Gamma is defined to be due south, so
        gamma = np.reshape(psip-np.pi,np.shape(latitude))
        
        # Solar intensity external to the Earths atmosphere
        Io = 1305.0
        
        # Indirect component adjustment
        Ind = 1.0
        
        # B
        B = (360./365.0)*(day-81.)*np.pi/180.0
        
        # Equation of Time
        EoT = 9.87*np.sin(2*B)-7.53*np.cos(B)-1.5*np.sin(B)
        
        # Time Correction factor
        TC = 4*longitude+EoT
        
        # Local Solar Time
        LST = TUTC/3600.0+TC/60.0
        
        # Hour Angle   
        HRA = (15.0*(LST-12.0))*np.pi/180.0
        
        # Declination angle (rad)
        delta = -23.44*np.cos((360./365.)*(day+10.)*np.pi/180.)*np.pi/180.
        
        # Zenith angle (rad)
        psi = np.arccos(np.sin(delta)*np.sin(latitude*np.pi/180.0)+np.cos(delta)*np.cos(latitude*np.pi/180.0)*np.cos(HRA))
        
        # Solar Azimuth angle, Duffie/Beckman 1.6.6
        gammas = np.sign(HRA)*np.abs((np.cos(psi)*np.sin(latitude*np.pi/180.)-np.sin(delta))/(np.sin(psi)*np.cos(latitude*np.pi/180)))
        
        # Slope of the solar panel, Bower AIAA 2011-7072 EQN 15
        beta = np.reshape(np.arccos(np.cos(thetap)*np.cos(phip)),np.shape(gammas))
        
        # Angle of incidence, Duffie/Beckman 1.6.3
        theta = np.arccos(np.cos(psi)*np.cos(beta)+np.sin(psi)*np.sin(beta)*np.cos(gammas-gamma))
        
        flux = np.zeros_like(psi)

        for ii in range(len(psi[:,0])):
        
            # Within the lower atmosphere
            if (psi[ii,0]>=-np.pi/2.)&(psi[ii,0]<96.70995*np.pi/180.)&(altitude[ii,0]<9000.):
                 
                # Using a homogeneous spherical model
                earthstuff = SUAVE.Attributes.Planets.Earth()
                Re = earthstuff.mean_radius
                 
                Yatm = 9. # The atmospheres thickness in km
                r = Re/Yatm
                c = altitude[ii,0]/9000. # Converted from m to km
                 
                AM = (((r+c)**2)*(np.cos(psi[ii,0])**2)+2.*r*(1.-c)-c**2 +1.)**(0.5)-(r+c)*np.cos(psi[ii,0])
                 
                Id = Ind*Io*(0.7**(AM**0.678))
                
                # Horizontal Solar Flux on the panel
                Ih = Id*(np.cos(latitude[ii]*np.pi/180.)*np.cos(delta[ii,0])*np.cos(HRA[ii,0])+np.sin(latitude[ii]*np.pi/180.)*np.sin(delta[ii,0]))              
                 
                # Solar flux on the inclined panel, Duffie/Beckman 1.8.1
                I = Ih*np.cos(theta[ii,0])/np.cos(psi[ii,0])
                 
            # Outside the atmospheric effects
            elif (psi[ii,0]>=-np.pi/2.)&(psi[ii,0]<96.70995*np.pi/180.)&(altitude[ii,0]>=9000.):
                 
                Id = Ind*Io
                
                # Horizontal Solar Flux on the panel
                Ih = Id*(np.cos(latitude[ii]*np.pi/180.)*np.cos(delta[ii,0])*np.cos(HRA[ii,0])+np.sin(latitude[ii]*np.pi/180.)*np.sin(delta[ii,0]))           
       
                # Solar flux on the inclined panel, Duffie/Beckman 1.8.1
                I = Ih*np.cos(theta[ii,0])/np.cos(psi[ii,0])
                 
            else:
                I = 0.
             
            # Adjusted Solar flux on a horizontal panel
            flux[ii,0] = max(0.,I)
        
        # Store to outputs
        self.outputs.flux = flux      
        
        # Return result for fun/convenience
        return flux