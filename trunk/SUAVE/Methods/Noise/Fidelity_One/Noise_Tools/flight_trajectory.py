## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
#flight_trajectory.py
#
# Created:  Jul 2015, C. Ilario
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------        
#   Imports
# ---------------------------------------------------------------------- 

import SUAVE
import numpy as np
from SUAVE.Core import Units

from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing

# ----------------------------------------------------------------------        
#   Flight Trajectory
# ---------------------------------------------------------------------- 

## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
def flight_trajectory(configs,turbofan,analyses):
    """ SUAVE.Methods.Noise.Fidelity_One.flight_trajectory(configs,turbofan):
            This routine calculates a simplified flight trajectory of an aircraft for the noise calculation procedure. 
            It is possible to simulate the three certification points (sideline, flyover, approach) and also a constant altitute flight path. 

            Inputs:
                    configs
                    turbofan

            Outputs: 
                time                              - Array with the discrete time of the data history [s]
                altitute                          - Array with the discrete altitute of the data history [m]
                dist                              - Array with the distance vector from the aircraft to the desired microphone position [m]
                theta                             - Array with the polar angles related to the desired microphone position [rad]
                phi                               - Array with the azimuthal angles related to the desired microphone position [rad]
                engine_data                       - Information regarding the engine performance data for the engine noise calculation

            Assumptions:
                Assume the aircraft as a point source."""
    
    
     #unpack
    s0 = configs.flight.initial_position   
    t0 = configs.flight.initial_time
    velocity = configs.flight.velocity
    altitute = configs.flight.altitute
    gama     = configs.flight.angle_of_climb  
    slope    = configs.flight.glide_slope   
    
    approach = configs.flight.approach
    flyover  = configs.flight.flyover
    sideline = configs.flight.sideline
    constant_flight = configs.flight.constant_flight
       
    #Necessary input for determination of noise trajectory    
    dt = 0.5*Units.s  #time step for noise calculation - Certification requirement
    
    #----------------------------------------
    # CONSTANT ALTITUDE NOISE TRAJECTORY
    #----------------------------------------
        
    if constant_flight==1:
        
        x0 = 0 #microphone reference position
        total_time = 60*Units.s #total time for noise calculation  
        n_steps    = np.int(total_time/dt +1)  #number of time steps (space discretization)
        
        #Defining the necessary arrays for the flight trajectory procedure
        theta = np.zeros(n_steps)
        s     = np.zeros(n_steps)
        dist  = np.zeros(n_steps)
        time  = np.zeros(n_steps)
        phi   = np.zeros(n_steps)
        
        #initial conditions
        time[0]          = t0
        s[0]             = s0    
        altitute_flyover = altitute
                
        dist[0]          = np.sqrt(altitute**2+(s[0]-x0)**2)
        theta[0]         = np.arctan(np.abs(altitute/s0))
        phi[0]           = 0
        altitute         = np.ones(n_steps)*altitute_flyover
        
    #Calculate flight path
        for i in range(1,n_steps):           
            s[i]         = s[i-1]+velocity*dt
            dist[i]      = np.sqrt(altitute[i]**2+(s[i]-x0)**2)
            theta[i]     = np.arctan(np.abs(altitute[i]/(s[i]-x0)))
            phi[i]       = 0
            time[i]      = time[i-1]+dt
            
            
         #Determine the engine performance parameter for the velocity and altitute    
        engine_data = engine_performnace(altitute,velocity,turbofan,analyses)  
        
    #----------------------------------------
    # APPROACH NOISE TRAJECTORY
    #----------------------------------------
    if approach==1:
        
        velocity_x = velocity*np.cos(slope*np.pi/180)
        velocity_y = velocity*np.sin(slope*np.pi/180)
        
        total_time = np.int(4000/velocity_x)    
        n_steps    = np.int(total_time/dt +1)  #number of time steps (space discretization)
        
        #Defining the necessary arrays for the flight trajectory procedure
        theta = np.zeros(n_steps)
        s     = np.zeros(n_steps)
        dist  = np.zeros(n_steps)
        time  = np.zeros(n_steps)
        phi   = np.zeros(n_steps)
        
        #initial conditions
        time[0] = t0
        s[0]    = s0
        
        x0 = 0 #microphone reference position
        
        altitute    = np.zeros(n_steps)
        altitute[0] = 120+2000*np.tan(slope*np.pi/180) 
        dist[0]     = np.sqrt(altitute[0]**2+(s[0]-x0)**2)
        theta[0]    = np.arctan(np.abs(altitute[0]/s0))
        phi[0]      = 0
        
      #Calculate flight path
        for i in range(1,n_steps):
            s[i] = s[i-1]+velocity_x*dt
            altitute[i] = altitute[i-1]-velocity_y*dt
            dist[i]     = np.sqrt(altitute[i]**2+(s[i]-x0)**2)
            theta[i]    = np.arctan(np.abs(altitute[i]/(s[i]-x0)))
            phi[i]      = 0
                           
            time[i] = time[i-1]+dt
            
        #Determine the engine performance parameter for the velocity and altitute    
        engine_data = engine_performnace(altitute,velocity,turbofan,analyses)
        
    #----------------------------------------
    # FLYOVER NOISE TRAJECTORY
    #----------------------------------------     
            
    if flyover==1:
        x0 = 6500*Units.m #Position of the Flyover microphone
                
        velocity_x = velocity*np.cos(gama*np.pi/180)
        velocity_y = velocity*np.sin(gama*np.pi/180)
        
        total_time = np.int((x0+500)/velocity_x)    
        n_steps    = np.int(total_time/dt +1)  #number of time steps (space discretization)
        
        #Defining the necessary arrays for the flight trajectory procedure
        theta    = np.zeros(n_steps)
        s        = np.zeros(n_steps)
        dist     = np.zeros(n_steps)
        time     = np.zeros(n_steps)
        altitute = np.zeros(n_steps)
        phi      = np.zeros(n_steps)
        
        #initial conditions
        time[0]     = t0
        s[0]        = 1061*Units.m #Lift-off position from the brake release
        altitute[0] = 35*Units.ft       
                
        dist[0]  = np.sqrt(altitute[0]**2+(x0-s[0])**2)
        theta[0] = np.arctan(np.abs(altitute[0]/(x0-s[0])))
        phi[0]   = 0
        
        #Calculate flight path
        for i in range(1,n_steps):
            s[i]        = s[i-1]+velocity_x*dt
            altitute[i] = altitute[i-1]+velocity_y*dt
            dist[i]     = np.sqrt(altitute[i]**2+(s[i]-x0)**2)
            theta[i]    = np.arctan(np.abs(altitute[i]/(s[i]-x0)))
            phi[i]      = 0
            time[i]     = time[i-1]+dt
            
        #Determine the engine performance parameter for the velocity and altitute    
        engine_data = engine_performnace(altitute,velocity,turbofan,analyses)
        
    #----------------------------------------
    # SIDELINE NOISE TRAJECTORY
    #----------------------------------------
            
    if sideline==1:
        z0 = 450  #position on the z-direction of the sideline microphone
        y0 = 0    #position on the y-direction of the sideline microphone
                    
        velocity_x = velocity*np.cos(gama*np.pi/180)
        velocity_y = velocity*np.sin(gama*np.pi/180)
        
        total_time = np.int((6500+500)/velocity_x)    
        n_steps    = np.int(total_time/dt +1)  #number of time steps (space discretization)
        
        #Defining the necessary arrays for the flight trajectory procedure
        theta    = np.zeros(n_steps)
        phi      = np.zeros(n_steps)
        s        = np.zeros(n_steps)
        dist     = np.zeros(n_steps)
        time     = np.zeros(n_steps)
        altitute = np.zeros(n_steps)
        
        #initial conditions        
        time[0]     = t0
        s[0]        = 1061*Units.m #Lift-off position from the brake release
        altitute[0] = 35*Units.ft   
        phi[0]      = np.arctan(z0/altitute[0])    
        
        x0 = s[0]+(1000-altitute[0])/np.tan(gama) #Position of the sideline microphone for the maximum take-off noise assumed to be at 1000ft of altitute
                
        dist[0]  = np.sqrt((450/np.sin(phi[0]))**2+(x0-s[0])**2)
        theta[0] = np.arccos(np.abs((x0-s[0]))/dist[0])
        
        
        
        #Calculate flight path
        for i in range(1,n_steps):
            s[i]        = s[i-1]+velocity_x*dt
            altitute[i] = altitute[i-1]+velocity_y*dt
            phi[i]      = np.arctan(z0/altitute[i])
            dist[i]     = np.sqrt((450/np.sin(phi[i]))**2+(s[i]-x0)**2)
            theta[i]    = np.arccos(np.abs((x0-s[i])/dist[i]))
            
            time[i] = time[i-1]+dt

        #Determine the engine performance parameter for the velocity and altitute
        engine_data = engine_performnace(altitute,velocity,turbofan,analyses)
    
    return(time,altitute,dist,theta,phi,engine_data)

# ----------------------------------------------------------------------        
#   Engine Performance
# ---------------------------------------------------------------------- 

def engine_performnace(altitude,velocity,turbofan,analyses):
    """ SUAVE.Methods.Noise.Fidelity_One.engine_performnace(altitude,velocity,turbofan):
            This routine generates the engine performance parameter for each point on the noise trajectory. 

            Inputs:
                    altitute
                    velocity
                    turbofan

            Outputs: 
                velocity_primary        -        Core nozzle jet velocity [m/s]
                temperature_primary     -        Core nozzle jet stagnation temperature [K]
                pressure_primary        -        Core nozzle jet stagnation pressure [Pa]
    
                velocity_secondary      -        Fan nozzle jet velocity [m/s]
                temperature_secondary     -      Fan nozzle jet stagnation temperature [K]
                pressure_secondary      -        Core nozzle jet stagnation pressure [Pa]

            Assumptions:
                               ."""
    
    
    #Calculation of the Aircraft Mach number
    mach_number = velocity/340.3
    
    #Number of discrete points on the flight trajectory
    n_steps = np.size(altitude)
    
    #Defining the necessary arrays for the engine performance procedure
    velocity_primary      = np.zeros(n_steps)
    temperature_primary   = np.zeros(n_steps)
    pressure_primary      = np.zeros(n_steps)
    velocity_secondary    = np.zeros(n_steps)
    temperature_secondary = np.zeros(n_steps)
    pressure_secondary    = np.zeros(n_steps)

    #size the turbofan

    for i in range (0,n_steps):
        
        atmo_data = analyses.atmosphere.compute_values(altitude[i])
         
         #call the atmospheric model to get the conditions at the specified altitude
        atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        conditions = atmosphere.compute_values(altitude[i])
        
        #setup conditions
        conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
        
        turbofan(conditions)
      
        velocity_primary[i]        = np.float(turbofan.core_nozzle.outputs.velocity)
        temperature_primary[i]     = np.float(turbofan.core_nozzle.outputs.stagnation_temperature)
        pressure_primary[i]        = np.float(turbofan.core_nozzle.outputs.stagnation_pressure)
    
        velocity_secondary[i]      = np.float(turbofan.fan_nozzle.outputs.velocity)
        temperature_secondary[i]   = np.float(turbofan.fan_nozzle.outputs.stagnation_temperature)
        pressure_secondary[i]      = np.float(turbofan.fan_nozzle.outputs.stagnation_pressure) 
        
    return (velocity_primary,temperature_primary,pressure_primary,velocity_secondary,temperature_secondary,pressure_secondary)
