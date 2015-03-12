# test_takeoff_segment.py
# 
# Created:  Tim Momose, May 2014


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import sys
sys.path.append('../trunk')

import SUAVE
from SUAVE.Core import Units
from SUAVE.Core  import Data

import numpy as np
import pylab as plt

import matplotlib
matplotlib.interactive(True)

import copy

from SUAVE.Analyses.Mission.Segments.Ground import Ground_Segment, Taxi, Takeoff, Landing

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
    
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    
    wing.sref      = vehicle.S     #
    wing.ar        = 8.3           #
    wing.span      = 27.8          #
    wing.sweep     = vehicle.delta * Units.deg #
    wing.symmetric = True          #
    wing.t_c       = 0.11          #
    wing.taper     = 0.28          #

    # size the wing planform
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chord_mac   = 12.0                  #
    wing.S_exposed   = 0.8*wing.area_wetted  # might not be needed as input
    wing.S_affected  = 0.6*wing.area_wetted  # part of high lift system
    wing.e           = 1.0                   #
    wing.twist_rc    = 2.0                   #
    wing.twist_tc    = 0.0                   #
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
    wing.tag = 'horizontal_stabilizer'
    
    wing.sref      = 26.         #
    wing.ar        = 5.5         #
    #wing.span      = 100            #
    wing.sweep     = 34.5 * Units.deg #
    wing.symmetric = True          
    wing.t_c       = 0.11          #
    wing.taper     = 0.11           #
    
    # size the wing planform
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chord_mac  = 8.0                   #
    wing.S_exposed  = 0.8*wing.area_wetted  #
    wing.S_affected = 0.6*wing.area_wetted  #  
    #wing.Cl         = 0.2                   #
    wing.e          = 0.9                   #
    wing.twist_rc   = 2.0                   #
    wing.twist_tc   = 2.0                   #
  
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
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chord_mac  = 11.0                  #
    wing.S_exposed  = 0.8*wing.area_wetted  #
    wing.S_affected = 0.6*wing.area_wetted  #  
    #wing.Cl        = 0.002                  #
    wing.e          = 0.9                   #
    wing.twist_rc   = 0.0                   #
    wing.twist_tc   = 0.0                   #
        
    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    
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
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.fuselage_planform(fuselage)
    
    # add to vehicle
    vehicle.append_component(fuselage)
    
    
    # ------------------------------------------------------------------
    #  Turbofan
    # ------------------------------------------------------------------    
    
    turbofan = SUAVE.Components.Propulsors.TurboFanPASS()
    turbofan.tag = 'turbo_fan'
    
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
    
    aerodynamics = SUAVE.Attributes.Aerodynamics.Fidelity_Zero()
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

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'EMBRAER_E190AR test mission'

    # atmospheric model
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet = SUAVE.Attributes.Planets.Earth()



    # ------------------------------------------------------------------
    #   Takeoff Segment
    # ------------------------------------------------------------------
    
    segment = Takeoff()
    segment.tag = "Takeoff Roll"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.takeoff
    
    # define segment attributes
    airport_altitude       = 0.0 * Units.km
    segment.atmosphere     = atmosphere
    segment.planet         = planet
    segment.altitude       = airport_altitude

    #Determine liftoff speed (which will be the segment end velocity)
    g_to      = planet.sea_level_gravity
    p_to, T_to, rho_to, a_to, mew_to = atmosphere.compute_values(segment.altitude)
    v_rot_assumed  = 175 * Units.mile / Units.hour
    rot_conditions = Data()
    rot_conditions.V = v_rot_assumed
    rot_conditions.mew = mew_to
    rot_conditions.rho = rho_to
    CL_max = SUAVE.Methods.Aerodynamics.Lift.High_lift_correlations.compute_max_lift_coeff(vehicle,rot_conditions)[0][0]
    v_mu   = (vehicle.Mass_Props.m_takeoff*g_to / (0.5*rho_to*CL_max*vehicle.S))**0.5
    v_rot  = 1.1 * v_mu
    print v_rot

    segment.velocity_start = 0.0   * Units['m/s']
    segment.velocity_end   = v_rot

    # add to misison
    mission.append_segment(segment)

    
    
    # ------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Climb Rate
    # ------------------------------------------------------------------
    
    segment =  SUAVE.Analyses.Mission.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb to 35'"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.takeoff
    
    # define segment attributes
    segment.atmosphere     = atmosphere
    segment.planet         = planet  
  
    segment.altitude_start = airport_altitude
    segment.altitude_end   = airport_altitude + 35. * Units.feet
    segment.climb_rate     = 3000. * Units['ft/min']
    segment.air_speed      = 1.2 * v_mu * Units['m/s']
    print segment.air_speed, segment.climb_rate
    
    # add to misison
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
    
    #Read the takeoff distance results
    climb_35ft_position    = results.Segments[1].conditions.frames.inertial.position_vector
    takeoff_ground_roll    = results.Segments[0].conditions.frames.inertial.position_vector[-1,0]
    rotation_distance      = results.Segments[0].conditions.frames.inertial.rotation_distance[0,0]
    takeoff_climb_35ft     = climb_35ft_position[-1,0] - climb_35ft_position[0,0]
    total_takeoff_distance = takeoff_ground_roll + rotation_distance + takeoff_climb_35ft
    print "Distance from brake release to rotation:  {0:.2f}m".format(takeoff_ground_roll)
    print "Distance to complete rotation:            {0:.2f}m".format(rotation_distance)
    print "Distance to clear 35 feet of altitude:    {0:.2f}m".format(takeoff_climb_35ft)
    print "Total FAR25 Takeoff Distance:             {0:.2f}m".format(total_takeoff_distance)
    print "Expected Takeoff Distance for E-190:      1870m"
    
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
        altitude = results.Segments[i].conditions.freestream.altitude[:,0] / Units.ft
        axes.plot(time, altitude, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Altitude (ft)')
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
    
    # ------------------------------------------------------------------    
    #   Position and velocity
    # ------------------------------------------------------------------     
    fig  = plt.figure("Position and Velocity")
    axes = plt.gca()
    for i in range(len(results.Segments)):
        time = results.Segments[i].conditions.frames.inertial.time[:,0] / Units.min
        vel  = results.Segments[i].conditions.frames.inertial.velocity_vector[:,0]
        pos  = results.Segments[i].conditions.frames.inertial.position_vector[:,0]
        
        axes = fig.add_subplot(2,1,1)
        axes.plot( time , pos , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Position (m)')
        axes.grid(True)
        
        axes = fig.add_subplot(2,1,2)
        axes.plot( time , vel , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Velocity (m/s)')
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

