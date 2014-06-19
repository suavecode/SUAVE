# tut_mission_Boeing_737.py
# 
# Created:  Michael Colonno, Apr 2013
# Modified: Michael Vegh   , Sep 2013
#           Trent Lukaczyk , Jan 2014

""" evaluate a mission with a Boeing 737-800
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE

import numpy as np
import pylab as plt

import copy
from SUAVE.Attributes import Units as Units
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
    vehicle.tag = 'Boeing 737-800'

    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.Mass_Props.m_full       = 79015.8   # kg
    vehicle.Mass_Props.m_empty      = 62746.4   # kg
    vehicle.Mass_Props.m_takeoff    = 79015.8   # kg
    vehicle.Mass_Props.m_flight_min = 66721.59  # kg

    # basic parameters
    vehicle.delta    = 25.0                     # deg
    vehicle.S        = 124.862                  # 
    vehicle.A_engine = np.pi*(0.9525)**2   
    Nult_wing=3.75
    Nult_h_stab=3.75
    Nult_v_stab=3.75
    Nult_fus=3.75
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Main Wing'
    
    wing.sref      = vehicle.S       #
    wing.ar        = 8             #
    #wing.span      = 35.66         #
    wing.span=(wing.ar*wing.sref)**.5
    wing.sweep     = 25*np.pi/180  #
    wing.symmetric = True          #
    wing.t_c       = 0.1           #
    wing.taper     = 0.16          #

    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chord_mac   = 12.5                  #
    wing.S_exposed   = 0.8*wing.area_wetted  #
    wing.S_affected  = 0.6*wing.area_wetted  #
    #wing.Cl          = 0.3                   #
    wing.e           = 0.9                   #
    #wing.alpha_rc    = 3.0                   #
    wing.alpha_rc    =3.0
    wing.alpha_tc    = -1.0                   #
    wing.highlift    = False                 
    wing.root_chord= 7.88 
    #wing.hl          = 1                     #
    #wing.flaps_chord = 20                    #
    #wing.flaps_angle = 20                    #
    #wing.slats_angle = 10                    #
    
    # add to vehicle
    vehicle.append_component(wing)
    main_wing=wing
    
    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Horizontal Stabilizer'
    
    wing.sref      =32.488         #
    wing.ar        = 6.16          #
    #wing.span      = 100           #
    wing.sweep     = 30*np.pi/180  #
    wing.symmetric = True          
    wing.t_c       = 0.08          #
    wing.taper     = 0.4           #
    
    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chord_mac  = 8.0                   #
    wing.S_exposed  = 0.8*wing.area_wetted  #
    wing.S_affected = 0.6*wing.area_wetted  #  
    #wing.Cl         = 0.2                   #
    wing.e          = 0.9                   #
    wing.alpha_rc   = 3.0                   #
    wing.alpha_tc   = 3.0                   #
    l_w2h          =16.764  #meters; estimated of length from MAC of wing to h_stabilizer
    h_stab=wing
    # add to vehicle
    vehicle.append_component(wing)
    
    
    # ------------------------------------------------------------------
    #   Vertcal Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Vertical Stabilizer'    
    
    wing.sref      = 32.488        #
    wing.ar        = 1.91          #
    #wing.span      = 100           #
    wing.sweep     = 25*np.pi/180  #
    wing.symmetric = False    
    wing.t_c       = 0.08          #
    wing.taper     = 0.25          #
    
    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chord_mac  = 12.5                  #
    wing.S_exposed  = 0.8*wing.area_wetted  #
    wing.S_affected = 0.6*wing.area_wetted  #  
    #wing.Cl        = 0.002                  #
    wing.e          = 0.9                   #
    wing.alpha_rc   = 0.0                   #
    wing.alpha_tc   = 0.0                   #
    v_stab=wing
    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'Fuselage'
    
    fuselage.num_coach_seats = 200  #
    fuselage.seat_pitch      = 1    #
    fuselage.fineness_nose   = 1.6  #
    fuselage.fineness_tail   = 2    #
    fuselage.fwdspace        = 6    #
    fuselage.aftspace        = 5    #
    fuselage.width           = 4.    #
    fuselage.height          = 4.    #
    fuselage.length          =39.4716 #meters
    S_fus=(np.pi*(fuselage.height))*fuselage.length
   
    atm = SUAVE.Attributes.Atmospheres.Earth.International_Standard()
    p1, T1, rho1, a1, mew1 = atm.compute_values(0.)
    p2, T2, rho2, a2, mew2 = atm.compute_values(10.64)
    diff_p_fus=max(abs(p2-p1),0)   
    
    # size fuselage planform
    SUAVE.Geometry.Two_Dimensional.Planform.fuselage_planform(fuselage)
    
    # add to vehicle
    vehicle.append_component(fuselage)
    
    
    # ------------------------------------------------------------------
    #  Turbofan
    # ------------------------------------------------------------------    
    
    turbofan = SUAVE.Components.Propulsors.TurboFanPASS()
    turbofan.tag = 'Turbo Fan'
    
    turbofan.propellant = SUAVE.Attributes.Propellants.Aviation_Gasoline()
    
    turbofan.analysis_type                 = '1D'     #
    turbofan.diffuser_pressure_ratio       = 0.98     #
    turbofan.fan_pressure_ratio            = 1.7      #
    turbofan.fan_nozzle_pressure_ratio     = 0.99     #
    turbofan.lpc_pressure_ratio            = 1.14     #
    turbofan.hpc_pressure_ratio            = 13.415   #
    turbofan.burner_pressure_ratio         = 0.95     #
    turbofan.turbine_nozzle_pressure_ratio = 0.99     #
    turbofan.Tt4                           = 1450.0   #
    turbofan.bypass_ratio                  = 5.4      #
    turbofan.design_thrust                 = 25000.0  #
    turbofan.no_of_engines                 = 2.0      #
    
    # turbofan sizing conditions
    sizing_segment = SUAVE.Components.Propulsors.Segments.Segment()
    
    sizing_segment.M   = 0.8          #
    sizing_segment.alt = 10.0         #
    sizing_segment.T   = 218.0        #
    sizing_segment.p   = 0.239*10**5  # 
    
    # size the turbofan
    turbofan.engine_sizing_1d(sizing_segment)     
    
    # add to vehicle
    vehicle.append_component(turbofan)


    # ------------------------------------------------------------------
    #   Simple Aerodynamics Model
    # ------------------------------------------------------------------ 
    
    aerodynamics = SUAVE.Attributes.Aerodynamics.PASS_Aero()
    aerodynamics.initialize(vehicle)
    vehicle.Aerodynamics = aerodynamics
    

    # ------------------------------------------------------------------
    #   Define Configurations
    # ------------------------------------------------------------------

    # --- Takeoff Configuration ---
    config = vehicle.new_configuration("takeoff")
    # this configuration is derived from the baseline vehicle

    # --- Cruise Configuration ---
    config = vehicle.new_configuration("cruise")
    # this configuration is derived from vehicle.Configs.takeoff
    
    Mguess=vehicle.Mass_Props.m_full
    #weights
    m_wing=SUAVE.Methods.Weights.Correlations.Tube_Wing.wing_main(wing.area_wetted*(Units.meter**2),wing.span*Units.meter,
    wing.taper,wing.t_c,wing.sweep*Units.rad,Nult_wing,vehicle.Mass_Props.m_empty*Units.kg,Mguess*Units.kg)
    wing.Mass_Props.mass=m_wing
    
    m_h_stab=SUAVE.Methods.Weights.Correlations.Tube_Wing.tail_horizontal(wing.span*Units.meter,wing.sweep*Units.rad,Nult_h_stab,wing.area_wetted*(Units.meter**2)
    ,Mguess*Units.kg,main_wing.chord_mac*Units.meter, wing.chord_mac*Units.meter,l_w2h*Units.meter,wing.t_c,wing.S_exposed)
    h_stab.Mass_Props.mass=m_h_stab
    
    
    m_v_stab=SUAVE.Methods.Weights.Correlations.Tube_Wing.tail_vertical((wing.area_wetted)*Units.meter**2,Nult_v_stab,
    wing.span*Units.meter,Mguess*Units.kg,wing.t_c, wing.sweep*Units.rad,wing.S_exposed*(Units.meter**2),"no")
    
    v_stab.Mass_Props.mass=m_v_stab.wt_tail_vertical+m_v_stab.wt_rudder
    systems=SUAVE.Methods.Weights.Correlations.Tube_Wing.systems(fuselage.num_coach_seats, 'fully powered', h_stab.area_wetted*(Units.meter**2), v_stab.area_wetted*(Units.meter**2),
    main_wing.area_wetted*(Units.meter**2), "medium-range")
    m_systems=systems.wt_systems
   
    m_landing_gear=SUAVE.Methods.Weights.Correlations.Tube_Wing.landing_gear(Mguess*Units.kg)
    m_pl=10454.54545*2.
    m_fuel=16000.
    engine_mass=SUAVE.Methods.Weights.Correlations.Propulsion.engine_jet(turbofan.design_thrust*Units.N)
    propulsion_mass=SUAVE.Methods.Weights.Correlations.Propulsion.integrated_propulsion(engine_mass,turbofan.no_of_engines)
    turbofan.Mass_Props.mass=propulsion_mass
    fus_weight=SUAVE.Methods.Weights.Correlations.Tube_Wing.tube(S_fus*(Units.meter**2), 
    diff_p_fus*Units.pascal,fuselage.width*Units.meter,fuselage.height*Units.meter, fuselage.length*Units.meter,
    Nult_fus, vehicle.Mass_Props.m_empty*Units.kg, main_wing.Mass_Props.mass*Units.kg, (turbofan.Mass_Props.mass)*Units.kg, main_wing.root_chord*Units.meter)
    fuselage.Mass_Props.mass=fus_weight
   
    #print S_fus*(Units.meter**2), diff_p_fus*Units.pascal,fuselage.width*Units.meter,fuselage.height*Units.meter, fuselage.length*Units.meter,   Nult_fus, vehicle.Mass_Props.m_empty*Units.kg, main_wing.Mass_Props.mass*Units.kg, (turbofan.Mass_Props.mass)*Units.kg, main_wing.root_chord*Units.meter
    fus_weight=SUAVE.Methods.Weights.Correlations.Tube_Wing.tube(S_fus*(Units.meter**2), 
    diff_p_fus*Units.pascal,fuselage.width*Units.meter,fuselage.height*Units.meter, fuselage.length*Units.meter,
    Nult_fus, vehicle.Mass_Props.m_empty*Units.kg, main_wing.Mass_Props.mass*Units.kg, (turbofan.Mass_Props.mass)*Units.kg, main_wing.root_chord*Units.meter)
    fuselage.Mass_Props.mass=fus_weight
    m_full=fuselage.Mass_Props.mass+ v_stab.Mass_Props.mass+h_stab.Mass_Props.mass+main_wing.Mass_Props.mass+m_pl+\
    turbofan.Mass_Props.mass*turbofan.no_of_engines+m_fuel+m_systems+m_landing_gear
    print m_wing, fuselage.Mass_Props.mass,  v_stab.Mass_Props.mass, turbofan.Mass_Props.mass
    print m_full, vehicle.Mass_Props.m_full
    
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
    mission.tag = 'The Test Mission'

    # initial mass
    mission.m0 = vehicle.Mass_Props.linked_copy('m_full') # linked copy updates if parent changes
    
    # atmospheric model
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet = SUAVE.Attributes.Planets.Earth()
    
    
    # ------------------------------------------------------------------
    #   First Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed()
    segment.tag = "Climb - 1"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.takeoff
    
    # define segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet    
    segment.altitude   = [0.0, 3.0]   # km
    
    # pick two:
    segment.Vinf       = 125.0        # m/s
    segment.rate       = 6.0          # m/s
    #segment.psi        = 8.5          # deg
    
    # add to misison
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------
    #   Second Climb Segment: constant Speed, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed()
    segment.tag = "Climb - 2"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet    
    segment.altitude   = [3.0, 8.0]  # km
    
    # pick two:
    segment.Vinf       = 190.0       # m/s
    segment.rate       = 6.0         # m/s
    #segment.psi        = 15.0        # deg
    
    # add to mission
    mission.append_segment(segment)

    
    # ------------------------------------------------------------------
    #   Third Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed()
    segment.tag = "Climb - 3"

    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet        
    segment.altitude   = [8.0, 10.668] # km
    
    # pick two:
    segment.Vinf        = 226.0        # m/s   
    segment.rate        = 3.0          # m/s
    #segment.psi         = 15.0         # deg
    
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
    segment.altitude   = 10.668    # km
    segment.Vinf       = 230.412   # m/s
    segment.range      = 3933.65   # km
    mission.append_segment(segment)

    # ------------------------------------------------------------------    
    #   First Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed()
    segment.tag = "Descent - 1"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # sergment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet    
    segment.altitude   = [10.668, 5.0]  # km
    segment.Vinf       = 170.0          # m/s
    segment.rate       = 5.0            # m/s
    
    # add to mission
    mission.append_segment(segment)
    

    # ------------------------------------------------------------------    
    #   Second Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed()
    segment.tag = "Descent - 2"

    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet   
    segment.altitude   = [5.0, 0.0]  # km
    segment.Vinf       = 145.0       # m/s
    segment.rate       = 5.0         # m/s

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
    
    
    # ------------------------------------------------------------------    
    #   Compute Useful Results
    # ------------------------------------------------------------------
    SUAVE.Methods.Results.compute_energies(results,summary=True)
    SUAVE.Methods.Results.compute_efficiencies(results)
    SUAVE.Methods.Results.compute_velocity_increments(results)
    SUAVE.Methods.Results.compute_alpha(results)    
    
    return results

# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def post_process(vehicle,mission,results):
    
    # ------------------------------------------------------------------    
    #   Thrust Angle
    # ------------------------------------------------------------------
    title = "Thrust Angle History"
    plt.figure(0)
    for i in range(len(results.Segments)):
        plt.plot(results.Segments[i].t/60,np.degrees(results.Segments[i].gamma),'bo-')
    plt.xlabel('Time (mins)'); plt.ylabel('Thrust Angle (deg)'); plt.title(title)
    plt.grid(True)

    # ------------------------------------------------------------------    
    #   Throttle
    # ------------------------------------------------------------------
    title = "Throttle History"
    plt.figure(1)
    for i in range(len(results.Segments)):
        plt.plot(results.Segments[i].t/60,results.Segments[i].eta,'bo-')
    plt.xlabel('Time (mins)'); plt.ylabel('Throttle'); plt.title(title)
    plt.grid(True)

    # ------------------------------------------------------------------    
    #   Angle of Attack
    # ------------------------------------------------------------------
    title = "Angle of Attack History"
    plt.figure(2)
    for i in range(len(results.Segments)):
        plt.plot(results.Segments[i].t/60,np.degrees(results.Segments[i].alpha),'bo-')
    plt.xlabel('Time (mins)'); plt.ylabel('Angle of Attack (deg)'); plt.title(title)
    plt.grid(True)

    # ------------------------------------------------------------------    
    #   Fuel Burn
    # ------------------------------------------------------------------
    title = "Fuel Burn"
    plt.figure(3)
    for i in range(len(results.Segments)):
        plt.plot(results.Segments[i].t/60,mission.m0 - results.Segments[i].m,'bo-')
    plt.xlabel('Time (mins)'); plt.ylabel('Fuel Burn (kg)'); plt.title(title)
    plt.grid(True)

    # ------------------------------------------------------------------    
    #   Fuel Burn Rate
    # ------------------------------------------------------------------
    title = "Fuel Burn Rate"
    plt.figure(4)
    for i in range(len(results.Segments)):
        plt.plot(results.Segments[i].t/60,results.Segments[i].mdot,'bo-')
    plt.xlabel('Time (mins)'); plt.ylabel('Fuel Burn Rate (kg/s)'); plt.title(title)
    plt.grid(True)

    
    
    # ------------------------------------------------------------------    
    #   Altitude
    # ------------------------------------------------------------------
    plt.figure(5)
    title = "Altitude"
    
    for i in range(len(results.Segments)):
     
        plt.plot(results.Segments[i].t/60,results.Segments[i].vectors.r[:,2],'bo-')
        
    plt.xlabel('Time (mins)'); plt.ylabel('Altitude (m)'); plt.title(title)
    plt.grid(True)
    
    
   

    
       
    # ------------------------------------------------------------------    
    #   Drag
    # ------------------------------------------------------------------
        
    plt.figure(6)
    title = "Drag"
    for i in range(len(results.Segments)):
        plt.plot(results.Segments[i].t/60,results.Segments[i].vectors.D[:,0],'bo-')
        
    plt.xlabel('Time (mins)'); plt.ylabel('Drag(N)'); plt.title(title)
    plt.grid(True)
    
    
    
    
    # ------------------------------------------------------------------    
    #   Lift
    # ------------------------------------------------------------------
        
    plt.figure(7)
    title = "Lift"
    for i in range(len(results.Segments)):
        plt.plot(results.Segments[i].t/60,results.Segments[i].vectors.L[:,2],'bo-')
        
    plt.xlabel('Time (mins)'); plt.ylabel('Drag(N)'); plt.title(title)
    plt.grid(True)
    plt.show()
    
    return     



# ---------------------------------------------------------------------- 
#   Call Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()


