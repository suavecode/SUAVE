#solar_flux.py
# 
# Created:  Emilio Botero, Feb 2014
# Modified:  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import scipy as sp

# ----------------------------------------------------------------------
#  The Function
# ----------------------------------------------------------------------

def solar_flux(day,TUTC,latitude,longitude,altitude):
     """" SUAVE.Methods.Power.test_solar(day,TUTC,latitude,longitude,altitude)
              Computes the adjusted solar flux in watts per square meter.
              
              Inputs:
                  day - day of the year from Jan 1st
                  
                  TUTC - time in seconds in UTC
                  
                  longitude- in degrees
                  
                  latitude - in degrees
                  
                  altitude - in meters
              
              Outputs:
                  sflux - adjusted solar flux
                  
              Assumptions:
              light intensity constant =0.14
              
              Solar intensity =1353 W/m^2
              
              Includes a diffuse component of 10% of the direct component
              
              Altitudes are not excessive
     """
     #

     #Light intensity constant   
     a=0.14
     
     #Solar intensity external to the Earths atmosphere
     Io=1353.0
     
     #B
     B=(360./365.0)*(day-81.)*np.pi/180.0
     
     #Equation of Time
     EoT=9.87*np.sin(2*B)-7.53*np.cos(B)-1.5*np.sin(B)
     
     #Time Correction factor
     TC=4*(longitude)+EoT
     
     #Local Solar Time
     LST=TUTC/3600.0+TC/60.0
     
     #Hour Angle   
     HRA=(15.0*(LST-12.0))*np.pi/180.0
     
     #Declination angle (rad)
     delta=-23.44*np.cos((360./365.)*(day+10.)*np.pi/180.)*np.pi/180.
     
     #Zenith angle (rad)
     psi=np.arccos(np.sin(delta)*np.sin(latitude*np.pi/180.0)+np.cos(delta)*np.cos(latitude*np.pi/180.0)*np.cos(HRA))
     
     if (psi>=-np.pi/2.)&(psi<96.70995*np.pi/180.)&(altitude<9000.):
          
          #Using a homogeneous spherical model
          earthstuff=SUAVE.Attributes.Planets.Earth()
          Re=earthstuff.mean_radius
          
          Yatm=9. #Atmospheres thickness in km
          r=Re/Yatm
          c=altitude/9000. #Converted from m to km
          
          AM=(((r+c)**2)*(np.cos(psi)**2)+2.*r*(1.-c)-c**2 +1.)**(0.5)-(r+c)*np.cos(psi)
          
          Id=1.1*Io*(0.7**(AM**0.678))
          
          #Horizontal Solar Flux
          Ihorizontal=Id/(np.sin(np.pi*0.5-latitude*np.pi/180.+delta))
     elif (psi>=-np.pi/2.)&(psi<96.70995*np.pi/180.)&(altitude>=9000):
          
          Id=1.1*Io
                    
          #Horizontal Solar Flux
          Ihorizontal=Id/(np.sin(np.pi*0.5-latitude*np.pi/180.+delta))

     else:
          Ihorizontal=0
          
     #Adjusted Solar flux on a horizontal panel
     sflux=max(0,Ihorizontal)

     return sflux