""" test_mission.py: evaluate a simple mission with a Cessna 172 Skyhawk """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import sys
sys.path.append('../trunk')
import SUAVE
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    
    # create simple vehicle
    Cessna172 = SUAVE.Vehicle()
    Cessna172.tag = "Cessna 172"
    Cessna172.Mass_Props.m_full = 1110.0             # kg
    Cessna172.Mass_Props.m_empty = 743.0             # kg
    Cessna172.Mass_Props.m_takeoff = 1110.0          # kg
    Cessna172.Mass_Props.m_flight_min = 750.0        # kg
    Cessna172.delta = 0.0                           # deg  

    # add the engine
    Lycoming_IO_360_L2AFC = SUAVE.Components.Propulsors.Motor_FC()
    Lycoming_IO_360_L2AFC.D = 1.905                      # m
    Lycoming_IO_360_L2AFC.F_min_static = 343.20          # N
    Lycoming_IO_360_L2AFC.F_max_static = 2085.9          # N
    Lycoming_IO_360_L2AFC.mdot_min_static = 0.004928     # kg/s 
    Lycoming_IO_360_L2AFC.mdot_max_static = 0.01213      # kg/s
    Cessna172.append_component(Lycoming_IO_360_L2AFC)

   
    
    
    
    # add a simple aerodynamic model
    Finite_Wing = SUAVE.Attributes.Aerodynamics.Finite_Wing()
    Finite_Wing.S = 16.2                            # reference area (m^2)
    Finite_Wing.AR = 7.32                           # aspect ratio
    Finite_Wing.e = 0.80                            # Oswald efficiency factor
    Finite_Wing.CD0 = 0.0341                        # CD at zero lift (from wind tunnel data)
    Finite_Wing.CL0 = 0.30                          # CL at alpha = 0.0 (from wind tunnel data)  
    Cessna172.Aerodynamics = Finite_Wing

    
    #create fuel cell
    fuel_cell = SUAVE.Components.Energy.Converters.Fuel_Cell()
    fuel_cell.active=1
    fuel     = SUAVE.Attributes.Propellants.Gaseous_H2()
    fuel_cell.propellant = fuel
    Cessna172.Energy.Converters['Fuel_Cell'] = fuel_cell
    Lycoming_IO_360_L2AFC.propellant = fuel #make fuel from fuel cell the aircraft propellant
    print fuel
    # create an Airport
    LAX = SUAVE.Attributes.Airports.Airport()

    # create takeoff configuration
    takeoff_config = Cessna172.new_configuration("takeoff")

    # create cruise configuration
    cruise_config = Cessna172.new_configuration("cruise")

    # create a climb segment: constant Mach, constant climb angle 
    
    climb = SUAVE.Analyses.Missions.Segments.Climb.Constant_Mach()
    climb.tag = "Cessena 172 Climb"
    climb.altitude = [0.0, 10.0]                     # km
    climb.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    climb.planet = SUAVE.Attributes.Planets.Earth()
    climb.config = cruise_config
    climb.Minf = 0.15                            
    climb.psi = 15.0                             # deg
    climb.options.N = 16
    
    
    

    # create a cruise segment: constant speed, constant altitude
    cruise = SUAVE.Analyses.Missions.Segments.Cruise.Constant_Speed_Constant_Altitude()
    cruise.tag = "Cessena 172 Cruise"
    cruise.altitude = 10.0                         # km
    cruise.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    cruise.planet = SUAVE.Attributes.Planets.Earth()
    cruise.config = cruise_config
    cruise.Vinf = 62.0                              # m/s
    cruise.range = 1000                             # km
    cruise.options.N = 16

    # create a descent segment: consant speed, constant descent rate
    descent = SUAVE.Analyses.Missions.Segments.Climb.Constant_Speed()
    descent.tag = "Cessena 172 Descent"
    descent.altitude = [10.0, 0.0]                     # km
    descent.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    descent.planet = SUAVE.Attributes.Planets.Earth()
    descent.config = cruise_config
    descent.Vinf = 45.0                              # m/s
    descent.rate = 5.0                               # m/s
    descent.options.N = 16

    
    # INITIAL SIZING
    # ------------------------------------------------------------------
    # (mike v's magical methods)
    
    #size a fuel cell and a battery that will theoretically run the mission
    W = 9.81*Cessna172.Mass_Props.m_full
    a = (1.4*287.*293.)**.5
    T = 1./(np.sin(climb.psi*np.pi/180.)) #direction of thrust vector
    factor = 5 #maximum power requirement of mission/5 so fuel cell doesnt get too big
    Preq = climb.Minf * W * a * T / factor 
    
    SUAVE.Methods.Power.size_fuel_cell(fuel_cell, Preq)
    
    #add fuel cell mass to aircraft
    Cessna172.Mass_Props.m_full += fuel_cell.Mass_Props.mass 
    Cessna172.Mass_Props.m_empty += fuel_cell.Mass_Props.mass 
    
    
    # create a Mission
    flight = SUAVE.Analyses.Missions.Mission()
    flight.tag = "Cessna 172 test mission"
    flight.m0 = Cessna172.Mass_Props.m_full          # kg
    flight.append_segment(climb)
    flight.append_segment(cruise)
    flight.append_segment(descent)

    # run mission
    results = SUAVE.Methods.Performance.evaluate_mission(flight)

    # compute energies
    SUAVE.Methods.Results.compute_energies(results,True)
    SUAVE.Methods.Results.compute_efficiencies(results)
    SUAVE.Methods.Results.compute_velocity_increments(results)
    SUAVE.Methods.Results.compute_alpha(results)


   # plot solution
    title = "Thrust Angle History"
    plt.figure(0)
    for i in range(len(results.Segments)):
        plt.plot(results.Segments[i].t/60,np.degrees(results.Segments[i].gamma),'bo-')
    plt.xlabel('Time (mins)'); plt.ylabel('Thrust Angle (deg)'); plt.title(title)
    plt.grid(True)

    title = "Throttle History"
    plt.figure(1)
    for i in range(len(results.Segments)):
        plt.plot(results.Segments[i].t/60,results.Segments[i].eta,'bo-')
    plt.xlabel('Time (mins)'); plt.ylabel('Throttle'); plt.title(title)
    plt.grid(True)

    title = "Angle of Attack History"
    plt.figure(2)
    for i in range(len(results.Segments)):
        plt.plot(results.Segments[i].t/60,np.degrees(results.Segments[i].alpha),'bo-')
    plt.xlabel('Time (mins)'); plt.ylabel('Angle of Attack (deg)'); plt.title(title)
    plt.grid(True)

    title = "Fuel Burn"
    plt.figure(3)
    for i in range(len(results.Segments)):
        plt.plot(results.Segments[i].t/60,flight.m0 - results.Segments[i].m,'bo-')
    plt.xlabel('Time (mins)'); plt.ylabel('Fuel Burn (kg)'); plt.title(title)
    plt.grid(True)

    title = "Fuel Burn Rate"
    plt.figure(4)
    for i in range(len(results.Segments)):
        plt.plot(results.Segments[i].t/60,results.Segments[i].mdot,'bo-')
    plt.xlabel('Time (mins)'); plt.ylabel('Fuel Burn Rate (kg/s)'); plt.title(title)
    plt.grid(True)

    #plt.subplot(234)
    #for i in range(len(SUAVE.Methods.Results.Segments)):
    #plt.plot(results.Segments[0].t/60,results.Segments[0].eta,'o-')
    #plt.xlabel('Time (mins)'); plt.ylabel('Throttle'); plt.title(title)
    #plt.grid(True)

    #title = "Cruise"
    #plt.subplot(232)
    #plt.plot(results.Segments[1].t/60,np.degrees(results.Segments[1].gamma),'o-')
    #plt.xlabel('Time (mins)'); plt.ylabel('Thrust Angle (deg)'); plt.title(title)
    #plt.grid(True)

    #plt.subplot(235)
    #plt.plot(results.Segments[1].t/60,results.Segments[1].eta,'o-')
    #plt.xlabel('Time (mins)'); plt.ylabel('Throttle'); plt.title(title)
    #plt.grid(True)

    #title = "Descent"
    #plt.subplot(233)
    #plt.plot(results.Segments[2].t/60,np.degrees(results.Segments[2].gamma),'o-')
    #plt.xlabel('Time (mins)'); plt.ylabel('Thrust Angle (deg)'); plt.title(title)
    #plt.grid(True)

    #plt.subplot(236)
    #plt.plot(results.Segments[2].t/60,results.Segments[2].eta,'o-')
    #plt.xlabel('Time (mins)'); plt.ylabel('Throttle'); plt.title(title)
    #plt.grid(True)


    plt.show()
    
    return

# call main
if __name__ == '__main__':
    main()

# Notes:
# Avgas power density = 43.71 MJ/kg
# Avgas mass density = 0.721 kg/l = 721 kg/m^3
