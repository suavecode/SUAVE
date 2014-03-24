# tut_mission_Cessna_172_fc.py
# 
# Created:  Michael Colonno, Apr 2013
# Modified: Michael Vegh   , Sep 2013
#           Trent Lukaczyk , Jan 2014

""" evaluate a simple mission with a Cessna 172 Skyhawk 
    powered by fuel cells 
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE

import numpy as np
import pylab as plt

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
    vehicle.tag = 'Cessna 172'
    
    # vehicle-level properties
    vehicle.Mass_Props.m_full       = 1110.0 # kg
    vehicle.Mass_Props.m_empty      = 743.0  # kg
    vehicle.Mass_Props.m_takeoff    = 1110.0 # kg
    vehicle.Mass_Props.m_flight_min = 750.0  # kg
    vehicle.delta                   = 0.0    # deg  
    
    
    # ------------------------------------------------------------------
    #   Propulsor
    # ------------------------------------------------------------------        
    
    # create a motor
    propulsor = SUAVE.Components.Propulsors.Motor_FC()
    propulsor.tag = 'Lycoming_IO_360_L2A_FuelCell'
    
    propulsor.propellant = SUAVE.Attributes.Propellants.Aviation_Gasoline()
    propulsor.D               = 1.905    # m
    propulsor.F_min_static    = 343.20   # N
    propulsor.F_max_static    = 2085.9   # N
    propulsor.mdot_min_static = 0.004928 # kg/s 
    propulsor.mdot_max_static = 0.01213  # kg/s
    
    # add component to vehicle
    vehicle.append_component(propulsor)     
    
    
    # ------------------------------------------------------------------
    #   Fuel Cell
    # ------------------------------------------------------------------        
    
    #create fuel cell
    fuelcell = SUAVE.Components.Energy.Converters.Fuel_Cell()
    fuelcell.tag = 'Fuel_Cell'
    
    # assign fuel
    fuelcell.Fuel = SUAVE.Attributes.Propellants.Gaseous_H2()
    fuelcell.active = 1
    propulsor.propellant = fuelcell.Fuel 
    vehicle.append_component( fuelcell )
    
    
    # ------------------------------------------------------------------
    #   Aerodynamic Model
    # ------------------------------------------------------------------        
    
    # a simple aerodynamic model
    aerodynamics = SUAVE.Attributes.Aerodynamics.Finite_Wing()
    
    aerodynamics.S   = 16.2    # reference area (m^2)
    aerodynamics.AR  = 7.32    # aspect ratio
    aerodynamics.e   = 0.80    # Oswald efficiency factor
    aerodynamics.CD0 = 0.0341  # CD at zero lift (from wind tunnel data)
    aerodynamics.CL0 = 0.30    # CL at alpha = 0.0 (from wind tunnel data)  
    
    # add model to vehicle aerodynamics
    vehicle.Aerodynamics = aerodynamics
    
    
    # ------------------------------------------------------------------
    #   Configurations
    # ------------------------------------------------------------------        
    
    # Takeoff configuration
    config = vehicle.new_configuration("takeoff")

    # Cruise Configuration
    config = vehicle.new_configuration("cruise")
    
    # these are available as -
    # vehicle.Configs.takeoff
    # vehicle.Configs.cruise
    
    
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
    mission.tag = 'Cessna 172 Test Mission'
    
    # initial mass
    mission.m0 = vehicle.Mass_Props.linked_copy('m_full') # linked copy updates if parent changes
    
    # atmospheric model
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet = SUAVE.Attributes.Planets.Earth()

    # ------------------------------------------------------------------
    #   Climb Segment: constant Mach, constant climb angle 
    # ------------------------------------------------------------------
    
    climb = SUAVE.Attributes.Missions.Segments.Climb.Constant_Mach()
    climb.tag = "Climb"
    
    # connect vehicle configuration
    climb.config = vehicle.Configs.cruise
    
    # segment attributes
    climb.atmosphere = atmosphere
    climb.planet     = planet    
    climb.altitude   = [0.0, 10.0] # km
    climb.Minf       = 0.15        # m/s
    climb.psi        = 15.0        # deg

    # add to mission
    mission.append_segment(climb)


    # ------------------------------------------------------------------
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------
    
    cruise = SUAVE.Attributes.Missions.Segments.Cruise.Constant_Speed_Constant_Altitude()
    cruise.tag = "Cruise"
    
    # connect vehicle configuration
    cruise.config = vehicle.Configs.cruise
    
    # config attributes
    cruise.atmosphere = atmosphere
    cruise.planet     = planet        
    cruise.altitude  = 10.0    # km
    cruise.Vinf      = 62.0    # m/s
    cruise.range     = 1000    # km
    
    # add to mission
    mission.append_segment(cruise)


    # ------------------------------------------------------------------
    #   Descent Segment: consant speed, constant descent rate
    # ------------------------------------------------------------------
    
    descent = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed()
    descent.tag = "Descent"
    
    # connect vehicle configuration
    descent.config = vehicle.Configs.cruise

    # segment attributes
    descent.atmosphere  = atmosphere
    descent.planet      = planet        
    descent.altitude  = [10.0, 0.0] # km
    descent.Vinf      = 45.0        # m/s
    descent.rate      = 5.0         # m/s
    
    # add to mission
    mission.append_segment(descent)

    
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
    #   Fuel Cell Initial Sizing
    # ------------------------------------------------------------------
    
    # component to size
    fuelcell = vehicle.Energy.Converters['Fuel_Cell']
    
    # sizing on the climb segment
    climb = mission.Segments['Climb']
    
    # estimate required power of the fuel cell
    W = 9.81*vehicle.Mass_Props.m_full
    a = (1.4*287.*293.)**.5
    T = 1./(np.sin(climb.psi*np.pi/180.))  #direction of thrust vector
    factor = 5                             #maximum power requirement of mission/5 so fuel cell doesnt get too big
    Preq = climb.Minf * W * a * T / factor # the required power
    
    # size the fuel cell
    SUAVE.Methods.Power.size_fuel_cell(fuelcell, Preq)
    
    #add fuel cell mass to aircraft
    vehicle.Mass_Props.m_full  += fuelcell.Mass_Props.mass 
    vehicle.Mass_Props.m_empty += fuelcell.Mass_Props.mass 
    
    
    # ------------------------------------------------------------------    
    #   Run Mission
    # ------------------------------------------------------------------
    results = SUAVE.Methods.Performance.evaluate_mission(mission)
    
    
    # ------------------------------------------------------------------    
    #   Compute Useful Results
    # ------------------------------------------------------------------
    SUAVE.Methods.Results.compute_energies(results, summary=True)
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

    plt.show()

    return     


# ---------------------------------------------------------------------- 
#   Call Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()


