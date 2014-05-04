# tut_mission_Embraer_E190.py
# 
# Created:  Michael Colonno, Apr 2013
# Modified: Michael Vegh   , Sep 2013
#           Trent Lukaczyk , Jan 2014


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import sys
sys.path.append('../trunk')


import SUAVE
from SUAVE.Attributes import Units

import numpy as np
import pylab as plt

import matplotlib
matplotlib.interactive(True)

import copy

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    
    # build the vehicle
    vehicle = define_vehicle()
    
    # define the mission
    mission = define_mission(vehicle)
    
    # evaluate the mission
    results = evaluate_mission(vehicle,mission)
    
    # plot results
    post_process(vehicle,mission,results)
    
    return


# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------

def define_vehicle():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'EMBRAER E190AR'

    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.Mass_Props.m_full       = 51800.    # kg
    vehicle.Mass_Props.m_empty      = 30100.    # kg
    vehicle.Mass_Props.m_takeoff    = 51800.    # kg
    vehicle.Mass_Props.m_flight_min = 30100.    # kg

    # basic parameters
    vehicle.delta    = 22.                      # deg
    vehicle.S        = 92.                      # m^2
    vehicle.A_engine = np.pi*( 57*0.0254 /2. )**2. 
    
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Main Wing'
    
    wing.sref      = vehicle.S     #
    wing.ar        = 8.3           #
    wing.span      = 27.8          #
    wing.sweep     = vehicle.delta * Units.deg #
    wing.symmetric = True          #
    wing.t_c       = 0.11          #
    wing.taper     = 0.28          #

    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chord_mac   = 12.0                  #
    wing.S_exposed   = 0.8*wing.area_wetted  # might not be needed as input
    wing.S_affected  = 0.6*wing.area_wetted  # part of high lift system
    wing.e           = 1.0                   #
    wing.alpha_rc    = 2.0                   #
    wing.alpha_tc    = 0.0                   #
    wing.highlift    = False                 
    #wing.hl          = 1                     #
    #wing.flaps_chord = 20                    #
    #wing.flaps_angle = 20                    #
    #wing.slats_angle = 10                    #
    
    # add to vehicle
    vehicle.append_component(wing)
    
    
    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Horizontal Stabilizer'
    
    wing.sref      = 26.         #
    wing.ar        = 5.5         #
    #wing.span      = 100            #
    wing.sweep     = 34.5 * Units.deg #
    wing.symmetric = True          
    wing.t_c       = 0.11          #
    wing.taper     = 0.11           #
    
    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chord_mac  = 8.0                   #
    wing.S_exposed  = 0.8*wing.area_wetted  #
    wing.S_affected = 0.6*wing.area_wetted  #  
    #wing.Cl         = 0.2                   #
    wing.e          = 0.9                   #
    wing.alpha_rc   = 2.0                   #
    wing.alpha_tc   = 2.0                   #
  
    # add to vehicle
    vehicle.append_component(wing)
    
    
    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Vertcal Stabilizer'    
    
    wing.sref      = 16.0        #
    wing.ar        = 1.7          #
    #wing.span      = 100           #
    wing.sweep     = 35. * Units.deg  #
    wing.symmetric = False    
    wing.t_c       = 0.12          #
    wing.taper     = 0.10          #
    
    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chord_mac  = 11.0                  #
    wing.S_exposed  = 0.8*wing.area_wetted  #
    wing.S_affected = 0.6*wing.area_wetted  #  
    #wing.Cl        = 0.002                  #
    wing.e          = 0.9                   #
    wing.alpha_rc   = 0.0                   #
    wing.alpha_tc   = 0.0                   #
        
    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'Fuselage'
    
    fuselage.num_coach_seats = 114  #
    fuselage.seat_pitch      = 0.7455    # m
    fuselage.seats_abreast   = 4    #
    fuselage.fineness_nose   = 2.0  #
    fuselage.fineness_tail   = 3.0  #
    fuselage.fwdspace        = 0    #
    fuselage.aftspace        = 0    #
    fuselage.width           = 3.0  #
    fuselage.height          = 3.4  #
    
    # size fuselage planform
    SUAVE.Geometry.Two_Dimensional.Planform.fuselage_planform(fuselage)
    
    # add to vehicle
    vehicle.append_component(fuselage)
    
    
    # ------------------------------------------------------------------
    #  Turbofan
    # ------------------------------------------------------------------    
    
    turbofan = SUAVE.Components.Propulsors.TurboFanPASS()
    turbofan.tag = 'Turbo Fan'
    
    turbofan.propellant = SUAVE.Attributes.Propellants.Jet_A()
    
    turbofan.analysis_type                 = '1D'     #
    turbofan.diffuser_pressure_ratio       = 0.99     #
    turbofan.fan_pressure_ratio            = 1.7      #
    turbofan.fan_nozzle_pressure_ratio     = 0.98     #
    turbofan.lpc_pressure_ratio            = 1.9      #
    turbofan.hpc_pressure_ratio            = 10.0     #
    turbofan.burner_pressure_ratio         = 0.95     #
    turbofan.turbine_nozzle_pressure_ratio = 0.99     #
    turbofan.Tt4                           = 1500.0   #
    turbofan.bypass_ratio                  = 5.4      #
    turbofan.design_thrust                 = 20300.0  #
    turbofan.no_of_engines                 = 2.0      #
    
    # turbofan sizing conditions
    sizing_segment = SUAVE.Components.Propulsors.Segments.Segment()
    
    sizing_segment.M   = 0.78          #
    sizing_segment.alt = 10.668         #
    sizing_segment.T   = 223.0        #
    sizing_segment.p   = 0.265*10**5  # 
    
    # size the turbofan
    turbofan.engine_sizing_1d(sizing_segment)     
    
    # add to vehicle
    vehicle.append_component(turbofan)


    # ------------------------------------------------------------------
    #   Simple Aerodynamics Model
    # ------------------------------------------------------------------ 
    
    aerodynamics = SUAVE.Attributes.Aerodynamics.PASS_Aero()
    aerodynamics.initialize(vehicle)
    vehicle.aerodynamics_model = aerodynamics
    
    # ------------------------------------------------------------------
    #   Simple Propulsion Model
    # ------------------------------------------------------------------     
    
    vehicle.propulsion_model = vehicle.Propulsors    
    

    # ------------------------------------------------------------------
    #   Define Configurations
    # ------------------------------------------------------------------

    # --- Takeoff Configuration ---
    config = vehicle.new_configuration("takeoff")
    # this configuration is derived from the baseline vehicle

    # --- Cruise Configuration ---
    config = vehicle.new_configuration("cruise")
    # this configuration is derived from vehicle.Configs.takeoff
    

    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------
    
    return vehicle

#: def define_vehicle()


# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------
def define_mission(vehicle):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Attributes.Missions.Mission()
    mission.tag = 'EMBRAER_E190AR test mission'

    # atmospheric model
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet = SUAVE.Attributes.Planets.Earth()
    
    
    # ------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Climb Rate
    # ------------------------------------------------------------------
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "CLIMB_250KCAS"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.takeoff
    
    # define segment attributes
    segment.atmosphere     = atmosphere
    segment.planet         = planet    
    
    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 3.048 * Units.km
    segment.air_speed      = 138.0 * Units['m/s']
    segment.climb_rate     = 3000. * Units['ft/min']
    
    # add to misison
    mission.append_segment(segment)

    
    # ------------------------------------------------------------------
    #   Second Climb Segment: Constant Speed, Constant Climb Rate
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "CLIMB_TRANSITION"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere     = atmosphere
    segment.planet         = planet    
    
    segment.altitude_end   = 3.657 * Units.km
    segment.air_speed      = 168.0 * Units['m/s']
    segment.climb_rate     = 2500. * Units['ft/min']

    # add to mission
    mission.append_segment(segment)

    
    # ------------------------------------------------------------------
    #   Third Climb Segment: Constant Speed, Constant Climb Rate
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "CLIMB_280KCAS"

    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        
    
    segment.altitude_end = 25000. * Units.ft
    segment.air_speed    = 200.0  * Units['m/s']
    segment.climb_rate   = 1800.  * Units['ft/min']
    
    
    # add to mission
    mission.append_segment(segment)

    
    # ------------------------------------------------------------------
    #   Fourth Climb Segment: Constant Speed, Constant Climb Rate
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "CLIMB_M0.78"

    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        
    
    segment.altitude_end = 32000. * Units.ft
    segment.air_speed    = 230.0  * Units['m/s']
    segment.climb_rate   = 900.   * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment)      

   
    # ------------------------------------------------------------------
    #   Fifth Climb Segment: Constant Speed, Constant Climb Rate
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "CLIMB_Final"

    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        
    
    segment.altitude_end = 37000. * Units.ft
    segment.air_speed    = 230.0  * Units['m/s']
    segment.climb_rate   = 200.   * Units['ft/min']

    # add to mission
    mission.append_segment(segment)    
    
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Cruise.Constant_Speed_Constant_Altitude()
    segment.tag = "Cruise"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet        
    
    segment.air_speed  = 230.  * Units['m/s']
    segment.distance   = 2050. * Units.nmi
    
    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------    
    #   First Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "DESCENT_M0.77"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet   
    
    segment.altitude_end = 9.31  * Units.km
    segment.air_speed    = 230.0 * Units['m/s']
    segment.descent_rate = 2600. * Units['ft/min']

    # add to mission
    mission.append_segment(segment)
    

    # ------------------------------------------------------------------    
    #   Second Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "DESCENT_290KCAS"

    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet   
    
    segment.altitude_end = 3.657 * Units.km
    segment.air_speed    = 200.0 * Units['m/s']
    segment.descent_rate = 2300. * Units['ft/min']    

    # append to mission
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------    
    #   Third Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "DESCENT_250KCAS"

    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet   
    
    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 140.0 * Units['m/s']
    segment.descent_rate = 1500. * Units['ft/min']    
    
    # append to mission
    mission.append_segment(segment)

    
    # ------------------------------------------------------------------    
    #   Mission definition complete    
    # ------------------------------------------------------------------
    
    return mission

#: def define_mission()


# ----------------------------------------------------------------------
#   Evaluate the Mission
# ----------------------------------------------------------------------
def evaluate_mission(vehicle,mission):
    
    # ------------------------------------------------------------------    
    #   Run Mission
    # ------------------------------------------------------------------
    results = SUAVE.Methods.Performance.evaluate_mission(mission)
    
    
    ## ------------------------------------------------------------------    
    ##   Compute Useful Results
    ## ------------------------------------------------------------------
    #SUAVE.Methods.Results.compute_energies(results,summary=True)
    #SUAVE.Methods.Results.compute_efficiencies(results)
    #SUAVE.Methods.Results.compute_velocity_increments(results)
    #SUAVE.Methods.Results.compute_alpha(results)    
    
    return results

# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def post_process(vehicle,mission,results):
    
    #output the results first
    #outputMission(results,'output.dat')
    
    ## ------------------------------------------------------------------    
    ##   Thrust Angle
    ## ------------------------------------------------------------------
    #title = "Thrust Angle History"
    #plt.figure(0)
    #for i in range(len(results.Segments)):
        #plt.plot(results.Segments[i].t/60,np.degrees(results.Segments[i].gamma),'bo-')
    #plt.xlabel('Time (mins)'); plt.ylabel('Thrust Angle (deg)'); plt.title(title)
    #plt.grid(True)

    # ------------------------------------------------------------------    
    #   Throttle
    # ------------------------------------------------------------------
    plt.figure("Throttle History")
    axes = plt.gca()
    for i in range(len(results.Segments)):
        time = results.Segments[i].conditions.frames.inertial.time[:,0] / Units.min
        eta  = results.Segments[i].conditions.propulsion.throttle[:,0]
        axes.plot(time, eta, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Throttle')
    axes.grid(True)

    # ------------------------------------------------------------------    
    #   Angle of Attack
    # ------------------------------------------------------------------
    plt.figure("Angle of Attack History")
    axes = plt.gca()    
    for i in range(len(results.Segments)):     
        time = results.Segments[i].conditions.frames.inertial.time[:,0] / Units.min
        aoa = results.Segments[i].conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        axes.plot(time, aoa, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Angle of Attack (deg)')
    axes.grid(True)        


    ## ------------------------------------------------------------------    
    ##   Fuel Burn
    ## ------------------------------------------------------------------
    #title = "Fuel Burn"
    #plt.figure(3)
    #for i in range(len(results.Segments)):
        #plt.plot(results.Segments[i].t/60,mission.m0 - results.Segments[i].m,'bo-')
    #plt.xlabel('Time (mins)'); plt.ylabel('Fuel Burn (kg)'); plt.title(title)
    #plt.grid(True)

    # ------------------------------------------------------------------    
    #   Fuel Burn Rate
    # ------------------------------------------------------------------
    plt.figure("Fuel Burn Rate")
    axes = plt.gca()    
    for i in range(len(results.Segments)):     
        time = results.Segments[i].conditions.frames.inertial.time[:,0] / Units.min
        mdot = results.Segments[i].conditions.propulsion.fuel_mass_rate[:,0]
        axes.plot(time, mdot, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Fuel Burn Rate (kg/s)')
    axes.grid(True)    
    
    # ------------------------------------------------------------------    
    #   Altitude
    # ------------------------------------------------------------------
    plt.figure("Altitude")
    axes = plt.gca()    
    for i in range(len(results.Segments)):     
        time     = results.Segments[i].conditions.frames.inertial.time[:,0] / Units.min
        altitude = results.Segments[i].conditions.freestream.altitude[:,0] / Units.km
        axes.plot(time, altitude, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Altitude (m)')
    axes.grid(True)
    
    # ------------------------------------------------------------------    
    #   Vehicle Mass
    # ------------------------------------------------------------------    
    plt.figure("Vehicle Mass")
    axes = plt.gca()
    for i in range(len(results.Segments)):
        time = results.Segments[i].conditions.frames.inertial.time[:,0] / Units.min
        mass = results.Segments[i].conditions.weights.total_mass[:,0]
        axes.plot(time, mass, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Vehicle Mass (kg)')
    axes.grid(True)
    

    ## ------------------------------------------------------------------    
    ##   Atmosphere
    ## ------------------------------------------------------------------
    #title = "Atmosphere"
    #plt.figure(title)    
    #plt.title(title)
    #for segment in results.Segments.values():

        #plt.subplot(3,1,1)
        #plt.plot( segment.t / Units.minute , 
                  #segment.rho * np.ones_like(segment.t),
                  #'bo-' )
        #plt.xlabel('Time (min)')
        #plt.ylabel('Density (kg/m^3)')
        #plt.grid(True)
        
        #plt.subplot(3,1,2)
        #plt.plot( segment.t / Units.minute , 
                  #segment.p * np.ones_like(segment.t) ,
                  #'bo-' )
        #plt.xlabel('Time (min)')
        #plt.ylabel('Pressure (Pa)')
        #plt.grid(True)
        
        #plt.subplot(3,1,3)
        #plt.plot( segment.t / Units.minute , 
                  #segment.T * np.ones_like(segment.t) ,
                  #'bo-' )
        #plt.xlabel('Time (min)')
        #plt.ylabel('Temperature (K)')
        #plt.grid(True)
    
    
    # ------------------------------------------------------------------    
    #   Aerodynamics
    # ------------------------------------------------------------------
    fig = plt.figure("Aerodynamic Forces")
    for segment in results.Segments.values():
        
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Lift   = -segment.conditions.frames.wind.lift_force_vector[:,2]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]

        axes = fig.add_subplot(3,1,1)
        axes.plot( time , Lift , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Lift (N)')
        axes.grid(True)
        
        axes = fig.add_subplot(3,1,2)
        axes.plot( time , Drag , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Drag (N)')
        axes.grid(True)
        
        axes = fig.add_subplot(3,1,3)
        axes.plot( time , Thrust , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Thrust (N)')
        axes.grid(True)
        
    # ------------------------------------------------------------------    
    #   Aerodynamics 2
    # ------------------------------------------------------------------
    fig = plt.figure("Aerodynamic Coefficients")
    for segment in results.Segments.values():
        
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        CLift  = segment.conditions.aerodynamics.lift_coefficient[:,0]
        CDrag  = segment.conditions.aerodynamics.drag_coefficient[:,0]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]

        axes = fig.add_subplot(3,1,1)
        axes.plot( time , CLift , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('CL')
        axes.grid(True)
        
        axes = fig.add_subplot(3,1,2)
        axes.plot( time , CDrag , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('CD')
        axes.grid(True)
        
        axes = fig.add_subplot(3,1,3)
        axes.plot( time , Drag   , 'bo-' )
        axes.plot( time , Thrust , 'ro-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Drag and Thrust (N)')
        axes.grid(True)
        
    
    ## ------------------------------------------------------------------    
    ##   TSFC
    ## ------------------------------------------------------------------    
    #title = "TSFC"
    #plt.figure(10)  
    #plt.title(title)
    #for segment in results.Segments.values():
        #plt.plot( segment.t / Units.minute , 
                  #3600.*segment.mdot/segment.F ,
                  #'ro-' )
        
    plt.show(block=True)
    
    

    return     

def outputMission(results,filename):
    
    import time                     # importing library
    import datetime                 # importing library
    
    fid = open(filename,'w')   # Open output file
    fid.write('Output file with mission profile breakdown\n\n') #Start output printing

    k1 = 1.727133242E-06                            # constant to airspeed conversion
    k2 = 0.2857142857                               # constant to airspeed conversion

    TotalRange = 0
    for i in range(len(results.Segments)):          #loop for all segments
        segment = results.Segments[i]
        
        if not isinstance(segment.Minf,np.ndarray) or len(segment.Minf)==1:#if mach const. then Mi = Mf
            Mf =  segment.Minf                      #Final mach number
            Mi =  segment.Minf                      #Initial mach number
        else:
            Mf =  segment.Minf[-1]                  #Final mach number
            Mi =  segment.Minf[0]                   #Initial mach number

        HPf = segment.vectors.r[-1,2]*3.28083          #Final segment Altitude   [ft]
        HPi = segment.vectors.r[0,2]*3.28083          #Initial segment Altitude  [ft]

        CLf = segment.CL[-1]                        #Final Segment CL [-]
        CLi = segment.CL[0]                         #Initial Segment CL [-]
        Tf =  segment.t[-1]/60.                     #Final Segment Time [min]
        Ti =  segment.t[0]/60.                      #Initial Segment Time [min]
        Wf =  segment.m[-1]                         #Final Segment weight [kg]
        Wi =  segment.m[0]                          #Initial Segment weight [kg]
        Dist = (segment.vectors.r[-1,0] - segment.vectors.r[0,0] )*0.0005399568 #Distance [nm]
        TotalRange = TotalRange + Dist


#       Aispeed conversion: KTAS to  KCAS
        p0 = np.interp(0,segment.atmosphere.z_breaks,segment.atmosphere.p_breaks )
        deltai = segment.p[0] / p0
        deltaf = segment.p[-1]/ p0

        VEi = Mi*(340.294*np.sqrt(deltai))          #Equivalent airspeed [m/s]
        QCPOi = deltai*((1.+ k1*VEi**2/deltai)**3.5-1.) #
        VCi = np.sqrt(((QCPOi+1.)**k2-1.)/k1)       #Calibrated airspeed [m/s]
        KCASi = VCi * 1.943844                      #Calibrated airspeed [knots]

        VEf = Mf*(340.294*np.sqrt(deltaf))          #Equivalent airspeed [m/s]
        QCPOf = deltaf*((1.+ k1*VEf**2/deltaf)**3.5-1.)
        VCf = np.sqrt(((QCPOf+1.)**k2-1.)/k1) #m/s  #Calibrated airspeed [m/s]
        KCASf = VCf * 1.943844                      #Calibrated airspeed [knots]

#       String formatting
        CLf_str =   str('%15.3f'   % CLf)     + '|'
        CLi_str =   str('%15.3f'   % CLi)     + '|'
        HPf_str =   str('%7.0f'    % HPf)     + '|'
        HPi_str =   str('%7.0f'    % HPi)     + '|'
        Dist_str =  str('%9.0f'    % Dist)    + '|'
        Wf_str =    str('%8.0f'    % Wf)      + '|'
        Wi_str =    str('%8.0f'    % Wi)      + '|'
        T_str =     str('%7.1f'   % (Tf-Ti))  + '|'
        Fuel_str=   str('%8.0f'   % (Wi-Wf))  + '|'
        Mi_str =    str('%7.3f'   % Mi)       + '|'
        Mf_str =    str('%7.3f'   % Mf)       + '|'
        KCASi_str = str('%7.1f'   % KCASi)    + '|'
        KCASf_str = str('%7.1f'   % KCASf)    + '|'

        Segment_str = ' ' + segment.tag[0:31] + (31-len(segment.tag))*' ' + '|'

        if i == 0:  #Write header
            fid.write( '         FLIGHT PHASE           |   ALTITUDE    |     WEIGHT      |  DIST.  | TIME  |  FUEL  |            SPEED              |\n')
            fid.write( '                                | From  |   To  |Inicial | Final  |         |       |        |Inicial| Final |Inicial| Final |\n')
            fid.write( '                                |   ft  |   ft  |   kg   |   kg   |    nm   |  min  |   kg   | KCAS  | KCAS  |  Mach |  Mach |\n')
            fid.write( '                                |       |       |        |        |         |       |        |       |       |       |       |\n')
        # Print segment data
        fid.write( Segment_str+HPi_str+HPf_str+Wi_str+Wf_str+Dist_str+T_str+Fuel_str+KCASi_str+KCASf_str+Mi_str+Mf_str+'\n')

    #Summary of results [nm]
    TotalFuel = results.Segments[0].m[0] - results.Segments[-1].m[-1]   #[kg]
    TotalTime = results.Segments[-1].t[-1] - results.Segments[0].t[0]   #[min]

    fid.write(2*'\n')
    fid.write(' Total Range (nm) ........... '+ str('%9.0f'   % TotalRange)+'\n')
    fid.write(' Total Fuel  (kg) ........... '+ str('%9.0f'   % TotalFuel)+'\n')
    fid.write(' Total Time  (hh:mm) ........ '+ time.strftime('    %H:%M', time.gmtime(TotalTime))+'\n')
    # Print timestamp
    fid.write(2*'\n'+ 43*'-'+ '\n' + datetime.datetime.now().strftime(" %A, %d. %B %Y %I:%M:%S %p"))

# ---------------------------------------------------------------------- 
#   Call Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()


