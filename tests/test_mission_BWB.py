""" test_mission_BWB.py: evaluate a mission with a Blended Wing Body Concept """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
import numpy as np
import matplotlib.pyplot as plt


from SUAVE.Methods.Aerodynamics import aero
from SUAVE.Methods.Propulsion import engine_sizing_1d  
from SUAVE.Methods.Propulsion import engine_analysis_1d
from SUAVE.Components.Wings import Wing
from SUAVE.Components.Fuselages import Fuselage
from SUAVE.Components.Propulsors import Ducted_Fan
from SUAVE.Methods.Geometry import wing_planform
from SUAVE.Methods.Geometry import fuselage_planform
from SUAVE.Attributes.Missions.Segments import Segment

# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    
    # create simple vehicle
    BWB = SUAVE.Vehicle()
    BWB.tag = "Blended Wing Body"
    

#----------------------mass property definition------------------------------------------    
    
    BWB.Mass_Props.m_full = 82029.2#67746.4             # kg
    BWB.Mass_Props.m_empty =  78319.3  #60848.6 #62746.4             # kg (Actually Zero-Fuel Weight)
    BWB.Mass_Props.m_takeoff = 82029.2#67746.4             # kg
    BWB.Mass_Props.m_flight_min = 0.1*(BWB.Mass_Props.m_full -BWB.Mass_Props.m_empty)+BWB.Mass_Props.m_empty #63246.4        # kg (10 percent fuel as reserve)

#-----------------------basic parameters---------------------------------------------------

    BWB.delta = 25.0                           # deg
    BWB.S = 124.862 # (value used in CFD calculations)
    BWB.A_engine = np.pi*(0.9525)**2   

    
    
    
    
    #--create necessary aerodynamic parts(wing,fuselage,engine)----------------------------------
    
    Wing1=Wing(tag = 'Wing1')
    Wing2=Wing(tag = 'Wing2')
    Wing3=Wing(tag = 'Wing3')    
    
       
    #--------------Main wing---------------
    Wing1.sref=124.862
    Wing1.ar = 8.0
    Wing1.span =35.66
    Wing1.sweep=25.0*np.pi/180
    Wing1.symmetric = True
    Wing1.t_c = 0.1
    Wing1.taper= 0.16    

    wing_planform(Wing1)
    Wing1.chord_mac= 12.5
    Wing1.S_exposed=0.8*Wing1.area_wetted
    Wing1.S_affected=0.6*Wing1.area_wetted 
    #Wing1.Cl = 0.3
    Wing1.e = 0.9
    Wing1.alpha_rc =3.0
    Wing1.alpha_tc =3.0  
    Wing1.highlift == False
  
    
    
    BWB.add_component(Wing1)
    
    
    #--Horizontal stabalizer------------------------------

    Wing2.sref=32.488
    Wing2.ar = 6.16
    #Wing2.span =100
    Wing2.sweep=30*np.pi/180
    Wing2.symmetric = True
    Wing2.t_c = 0.08
    Wing2.taper= 0.4    

    wing_planform(Wing2)
    Wing2.chord_mac= 8.0
    Wing2.S_exposed=0.8*Wing2.area_wetted
    Wing2.S_affected=0.6*Wing2.area_wetted     
    #Wing2.Cl = 0.2
    Wing2.e = 0.9
    Wing2.alpha_rc =3.0
    Wing2.alpha_tc =3.0    
  
    BWB.add_component(Wing2)
    
    
#----------------------Vertcal stabalizer----------    
    
    Wing3.sref=32.488
    Wing3.ar = 1.91
    #Wing3.span =100
    Wing3.sweep=25*np.pi/180
    Wing3.symmetric = False
    Wing3.t_c = 0.08
    Wing3.taper= 0.25
       
    wing_planform(Wing3)
    Wing3.chord_mac= 12.5
    Wing3.S_exposed=0.8*Wing3.area_wetted
    Wing3.S_affected=0.6*Wing3.area_wetted     
    #Wing3.Cl = 0.002  
    Wing3.e = 0.9
    Wing3.alpha_rc =0.0
    Wing3.alpha_tc =0.0       
        
    BWB.add_component(Wing3)


#---------------fuselage------------------------------------------------------------------------------
    fus=Fuselage(tag = 'fuselage1')
    
    fus.num_coach_seats = 200
    fus.seat_pitch = 1
    fus.fineness_nose = 1.6
    fus.fineness_tail =  2
    fus.fwdspace = 6
    fus.aftspace = 5
    fus.width  =4
    fus.height =4   
    
    fuselage_planform(fus)
    #print fus.length_cabin
    BWB.add_component(fus)
    
#-----------propulsion------------------------------------------------------------------------    
    """
    Turbofan=Turbo_Fan()
    Turbofan.analysis_type == '1D'
    Turbofan.diffuser_pressure_ratio = 0.98
    Turbofan.fan_pressure_ratio = 1.7
    Turbofan.fan_nozzle_pressure_ratio = 0.99
    Turbofan.lpc_pressure_ratio = 1.14
    Turbofan.hpc_pressure_ratio = 13.415
    Turbofan.burner_pressure_ratio = 0.95
    Turbofan.turbine_nozzle_pressure_ratio = 0.99
    Turbofan.Tt4 = 1350.0
    Turbofan.bypass_ratio = 5.4
    Turbofan.design_thrust = 25000.0
    Turbofan.no_of_engines=2.0
    """
    
    #SUAVE.Methods.Power.SizeFuelCell(fuelcell, Preq)
    
    Ductedfan=Ducted_Fan()
    Ductedfan.analysis_type == '1D'
    fuelcell = SUAVE.Components.Energy.Converters.Fuel_Cell()
    fuel     = SUAVE.Attributes.Propellants.H2()
    fuelcell.Fuel = fuel
    BWB.Energy.Converters['Fuel_Cell'] = fuelcell
    Ductedfan.propellant = fuel #make fuel from fuel cell the aircraft propellant
    
    Ductedfan.diffuser_pressure_ratio = 0.98
    Ductedfan.fan_pressure_ratio = 1.65
    Ductedfan.fan_nozzle_pressure_ratio = 0.99
    Ductedfan.design_thrust = 25000.0  
    Ductedfan.no_of_engines=2.0     
    Ductedfan.fuelcell=fuelcell
    

    #--turbofan sizing conditions--------------------------------------------------------
    st=Segment()
    
    st.M=0.8
    st.alt=10.0
    st.T=218.0
    st.p=0.239*10**5    
        
    print st.p   
    #Turbofan.engine_sizing_1d(st)   #calling the engine sizing method 
    Ductedfan.engine_sizing_ductedfan(st,fuelcell)   #calling the engine sizing method 
    
    
    #--turbofan sizing conditions--------------------------------------------------------
    st=Segment()
    
    st.M=0.8
    st.alt=10.0
    st.T=218.0
    st.p=0.239*10**5    
        
    print st.p   
    Ductedfan.engine_sizing_ductedfan(st,fuelcell)   #calling the engine sizing method 
    
    #engine_sizing_1d(Turbofan,st)     
    
    #----------------------------------------------------------------------------------------

    
    
    
    #---reading in cfd aero data-----------
    Cl_var=np.empty([6,2])
    
    #---explicitly define the CL and aoa values
    
    Cl_var=np.array([[0,-0.3423111], 
    [1,0.1792845],                  
    [2,0.7122031],                  
    [3,1.247309],            
    [4,1.776392],           
    [5,2.294759],        
    [7,3.278907]])    
    
    
    
    BWB.add_component(Ductedfan)
# add a simple aerodynamic model
    wing_aero = SUAVE.Attributes.Aerodynamics.CFD_Aero()
    wing_aero.initialize(BWB,Cl_var)
    BWB.Aerodynamics = wing_aero   

    
    
   

    # create an Airport (optional)
    LAX = SUAVE.Attributes.Airports.Airport()
    
    # create takeoff configuration
    takeoff_config =  BWB.new_configuration("takeoff")

    # create cruise configuration
    cruise_config =  BWB.new_configuration("cruise")    


#-----------create configurations---------------------------------------------------------

   
   
       


    ##-------------------define the necessary segments---------------------------------------------
    
    
    ##-----------------------climb segments------------------------------
    ##------------------------------------------------------------
    ## create a climb segment: constant Mach, constant climb angle 
  
    
    climb = SUAVE.Attributes.Missions.Segments.ClimbDescentConstantSpeed()
    climb.tag = "Climb"
    climb.altitude = [0.0, 3.0]                     # km
    climb.atmosphere = SUAVE.Attributes.Atmospheres.Earth.USStandard1976()
    climb.planet = SUAVE.Attributes.Planets.Earth()
    climb.config = cruise_config
    climb.Vinf =125.0   
    climb.rate = 6.0      # m/s
    climb.angle = 8.5  # m/s                          # deg
    #climb.m0 = mass_eos_cl1_2        # kg
    #climb.t0 = 0.0                                 # initial time (s)
    climb.options.N = 4
    
    
   
    climb1 = SUAVE.Attributes.Missions.Segments.ClimbDescentConstantSpeed()
    climb1.tag = "Climb1"
    climb1.altitude = [3.0, 8.0]                     # km
    climb1.atmosphere = SUAVE.Attributes.Atmospheres.Earth.USStandard1976()
    climb1.planet = SUAVE.Attributes.Planets.Earth()
    climb1.config = cruise_config
    climb1.Vinf =181.0    
    climb1.rate = 6     # m/s
    #climb.psi = 15.0                             # deg
    #climb.m0 = mass_eos_cl1_2        # kg
    #climb.t0 = 0.0                                 # initial time (s)
    climb.options.N = 4
    
    
    climb2 = SUAVE.Attributes.Missions.Segments.ClimbDescentConstantMach()
    climb2.tag = "Climb2"
    climb2.altitude = [8.0, 10.0]                     # km
    climb2.atmosphere = SUAVE.Attributes.Atmospheres.Earth.USStandard1976()
    climb2.planet = SUAVE.Attributes.Planets.Earth()
    climb2.config = cruise_config
    climb2.Minf = 0.8
    #climb2.Vinf =181    
    climb2.rate = 1.0    # m/s
    #climb.psi = 15.0                             # deg
    #climb.m0 = mass_eos_cl1_2        # kg
    #climb.t0 = 0.0                                 # initial time (s)
    climb2.options.N = 4    
    
    
    #---------------cruise segment----------------------------------------------
    # create a cruise segment: constant speed, constant altitude
    

    #cruise = SUAVE.Attributes.Missions.Segments.ClimbDescentConstantSpeed()    
    cruise = SUAVE.Attributes.Missions.Segments.CruiseConstantSpeedConstantAltitude()
    cruise.tag = "Cruise"
    cruise.altitude = 10.0                         # km
    cruise.atmosphere = SUAVE.Attributes.Atmospheres.Earth.USStandard1976()
    cruise.planet = SUAVE.Attributes.Planets.Earth()
    cruise.config = cruise_config
    cruise.Vinf = 236.76                              # m/s
    cruise.range = 3933.65                            # km
    #cruise.m0 = mass_eos_cl  #Boeing737_800.Mass_Props.m_full          # kg
    #cruise.t0 = 0.0                                 # initial time (s)
    cruise.options.N = 4
    
    
    #sol2=SUAVE.Methods.Utilities.pseudospectral(cruise)
    #mass_eos_cr=cruise.m[cruise.options.N -1]
    #print 'fuelburn cruise ', mass_eos_cl-mass_eos_cr




    #-------------------------------descent segments------------------------

    descent1 = SUAVE.Attributes.Missions.Segments.ClimbDescentConstantSpeed()
    descent1.tag = "Descent"
    descent1.altitude = [10.0, 5.0]                     # km
    descent1.atmosphere = SUAVE.Attributes.Atmospheres.Earth.USStandard1976()
    descent1.planet = SUAVE.Attributes.Planets.Earth()
    descent1.config = cruise_config
    #descent1.Minf = 0.6   
    descent1.Vinf = 170.0 # m/s
    descent1.rate = 3.0                               # m/s
    #descent1.m0 = mass_eos_cr  #Boeing737_800.Mass_Props.m_full          # kg
    #descent1.t0 = 0.0                                 # initial time (s)
    descent1.options.N = 4
    



    ## create a descent segment: consant speed, constant descent rate
    descent = SUAVE.Attributes.Missions.Segments.ClimbDescentConstantSpeed()
    descent.tag = "Descent2"
    descent.altitude = [5.0, 0.0]                     # km
    descent.atmosphere = SUAVE.Attributes.Atmospheres.Earth.USStandard1976()
    descent.planet = SUAVE.Attributes.Planets.Earth()
    descent.config = cruise_config
    descent.Vinf = 145.0                              # m/s
    descent.rate = 3.0                               # m/s
    #descent.m0 = mass_eos_des31  #Boeing737_800.Mass_Props.m_full          # kg
    #descent.t0 = 0.0                                 # initial time (s)
    descent.options.N = 4

    


    # create a Mission
    flight = SUAVE.Attributes.Missions.Mission()
    flight.tag = "BWB test mission"
    flight.m0 = BWB.Mass_Props.m_full          # kg
    flight.add_segment(climb)
    flight.add_segment(climb1)
    flight.add_segment(climb2)
    flight.add_segment(cruise)
    flight.add_segment(descent1) 
    flight.add_segment(descent) 



    # run mission
    results = SUAVE.Methods.Performance.evaluate_mission(flight)    


    # compute energies
    SUAVE.Methods.Results.compute_energies(results,True)
    SUAVE.Methods.Results.compute_efficiencies(results)
    SUAVE.Methods.Results.compute_velocity_increments(results)
    SUAVE.Methods.Results.compute_alpha(results)
    
    print 'No of segments: ',len(results.Segments)
    print 'Climb Fuel burn ', flight.m0 - results.Segments[2].m
    print 'Cruise Fuel burn ', results.Segments[2].m - results.Segments[3].m
    print 'Descent Fuel burn ', results.Segments[3].m-results.Segments[len(results.Segments)-1].m 
    
    print 'Fuel burn of the mission', flight.m0 - results.Segments[len(results.Segments)-1].m

###---------------------Display the results---------------------------------------------

###------------------------------------------------------------

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

    plt.show()

    
    
       
    return

 

 
 
 
# call main
if __name__ == '__main__':
    main()
