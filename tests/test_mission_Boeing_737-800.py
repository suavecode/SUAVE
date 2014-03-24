""" test_mission_Boeing_737-800.py: evaluate a mission with a Boeing 737-800 """

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
from SUAVE.Components.Propulsors import Turbofan
from SUAVE.Geometry.Two_Dimensional.Planform import wing_planform, fuselage_planform
from SUAVE.Attributes.Missions.Segments import Segment

# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    
    # create simple vehicle
    Boeing737_800 = SUAVE.Vehicle()
    Boeing737_800.tag = "Boeing 737-800"
    

#----------------------mass property definition------------------------------------------    
    
    Boeing737_800.Mass_Props.m_full = 79015.8             # kg
    Boeing737_800.Mass_Props.m_empty = 62746.4             # kg
    Boeing737_800.Mass_Props.m_takeoff = 79015.8         # kg
    Boeing737_800.Mass_Props.m_flight_min = 66721.59        # kg

#-----------------------basic parameters---------------------------------------------------

    Boeing737_800.delta = 25.0                           # deg
    Boeing737_800.S = 124.862 
    Boeing737_800.A_engine = np.pi*(0.9525)**2   

    
    
    
    
    #--create necessary aerodynamic parts(wing,fuselage,engine)----------------------------------
    
    Wing1=Wing(tag = 'Wing1')
    Wing2=Wing(tag = 'Wing2')
    Wing3=Wing(tag = 'Wing3')    
    
       
    #--------------Main wing---------------
    Wing1.sref=124.862
    Wing1.ar = 8
    Wing1.span =35.66
    Wing1.sweep=25*np.pi/180
    Wing1.symmetric = True
    Wing1.t_c = 0.1
    Wing1.taper= 0.16    

    wing_planform(Wing1)
    #Wing1.chord_mac= 12.5
    Wing1.S_exposed=0.8*Wing1.area_wetted
    Wing1.S_affected=0.6*Wing1.area_wetted 
    #Wing1.Cl = 0.3
    Wing1.e = 0.9
    Wing1.alpha_rc =3.0
    Wing1.alpha_tc =3.0  
    Wing1.highlift == False
    #Wing1.hl     = 1
    #Wing1.flaps_chord=20
    #Wing1.flaps_angle=20
    #Wing1.slats_angle= 10   
    
    
    Boeing737_800.append_component(Wing1)
    
    
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
  
    Boeing737_800.append_component(Wing2)
    
    
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
        
    Boeing737_800.append_component(Wing3)


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
    Boeing737_800.append_component(fus)
    
#-----------propulsion------------------------------------------------------------------------    
    
    engine=Turbofan()
    engine.analysis_type = '1D'
    engine.diffuser_pressure_ratio = 0.98
    engine.fan_pressure_ratio = 1.7
    engine.fan_nozzle_pressure_ratio = 0.99
    engine.lpc_pressure_ratio = 1.14
    engine.hpc_pressure_ratio = 13.415
    engine.burner_pressure_ratio = 0.95
    engine.turbine_nozzle_pressure_ratio = 0.99
    engine.Tt4 = 1350.0
    engine.bypass_ratio = 5.4
    engine.design_thrust = 25000.0
    engine.no_of_engines=2.0
    
    #--turbofan sizing conditions--------------------------------------------------------
    st=Segment()
    
    st.M=0.8
    st.alt=10.0
    st.T=218.0
    st.p=0.239*10**5    
        
    engine_sizing_1d(engine,st)     
    
    #----------------------------------------------------------------------------------------
    
    
    Boeing737_800.append_component(engine)


    
   

    # create an Airport (optional)
    LAX = SUAVE.Attributes.Airports.Airport()


#-----------create configurations---------------------------------------------------------

    # create cruise configuration
    cruise_config = Boeing737_800.new_configuration("cruise")
    cruise_config.Functions.Propulsion = engine_analysis_1d #Lycoming_IO_360_L2A #engine_analysis_1d
    cruise_config.Functions.Aero = aero #Cessna172_finite_wing #aero
    cruise_config.throttle = 1.0   
   
   
   
    # create takeoff configuration
    takeoff_config = Boeing737_800.new_configuration("takeoff")
    takeoff_config.Functions.Propulsion = engine_analysis_1d
    takeoff_config.Functions.Aero = aero
    takeoff_config.throttle = 1.0
    takeoff_config.Wings[0].hl     = 1
    takeoff_config.Wings[0].flaps_chord=20
    takeoff_config.Wings[0].flaps_angle=20
    takeoff_config.Wings[0].slats_angle= 10       
    
    
    


    #-------------------define the necessary segments---------------------------------------------
    
    
    #-----------------------climb segments------------------------------
    #------------------------------------------------------------
    # create a climb segment: constant Mach, constant climb angle 
    climb1 = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed()
    climb1.tag = "Climb"
    climb1.altitude = [0.0, 3.0]                     # km
    climb1.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    climb1.planet = SUAVE.Attributes.Planets.Earth()
    climb1.config = takeoff_config
    climb1.Vinf = 125.0                              # m/s
    #climb1.rate = 5.0     
    climb1.angle = 8.5  # m/s
    climb1.m0 = Boeing737_800.Mass_Props.m_full          # kg          # kg
    climb1.t0 = 0.0                                 # initial time (s)
    climb1.options.N = 4
    
   
    sol=SUAVE.Methods.Solvers.pseudospectral(climb1)    
    mass_eos_cl1=climb1.m[climb1.options.N -1]
    print 'fuelburn climb1' , Boeing737_800.Mass_Props.m_full -mass_eos_cl1
    
    
    
    
    climb2 = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed()
    climb2.tag = "Climb"
    climb2.altitude = [3.0, 7.7]                     # km
    climb2.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    climb2.planet = SUAVE.Attributes.Planets.Earth()
    climb2.config = cruise_config
    climb2.Vinf = 181.0 
    climb2.rate = 7.8     # m/s
    #climb.angle = 10.0                             # deg
    climb2.m0 = mass_eos_cl1        # kg
    climb2.t0 = 0.0                                 # initial time (s)
    climb2.options.N = 4
    
    
    sol1_2=SUAVE.Methods.Utilities.pseudospectral(climb2)    
    mass_eos_cl1_2=climb2.m[climb2.options.N -1]    
    
    print 'fuelburn climb' , mass_eos_cl1 -mass_eos_cl1_2    

    
    
    
    
    climb = SUAVE.Attributes.Missions.Segments.Climb.Constant_Mach()
    climb.tag = "Climb"
    climb.altitude = [7.7, 10.0]                     # km
    climb.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    climb.planet = SUAVE.Attributes.Planets.Earth()
    climb.config = cruise_config
    climb.Minf = 0.8     
    climb.rate = 3.0      # m/s
    #climb.angle = 10.0                             # deg
    climb.m0 = mass_eos_cl1_2        # kg
    climb.t0 = 0.0                                 # initial time (s)
    climb.options.N = 4
    
    
    sol1=SUAVE.Methods.Utilities.pseudospectral(climb)    
    mass_eos_cl=climb.m[climb.options.N -1]    
    
    print 'fuelburn climb' , Boeing737_800.Mass_Props.m_full -mass_eos_cl
    
    
    
    
    
    
    #---------------cruise segment----------------------------------------------
    # create a cruise segment: constant speed, constant altitude
    cruise = SUAVE.Attributes.Missions.Segments.Cruise.Constant_Speed_Constant_Altitude()
    cruise.tag = "Cruise"
    cruise.altitude = 10.0                         # km
    cruise.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    cruise.planet = SUAVE.Attributes.Planets.Earth()
    cruise.config = cruise_config
    cruise.Vinf = 236.76                              # m/s
    cruise.range = 3933.65                            # km
    cruise.m0 = mass_eos_cl  #Boeing737_800.Mass_Props.m_full          # kg
    cruise.t0 = 0.0                                 # initial time (s)
    cruise.options.N = 8
    
    
    sol2=SUAVE.Methods.Utilities.pseudospectral(cruise)
    mass_eos_cr=cruise.m[cruise.options.N -1]
    print 'fuelburn cruise ', mass_eos_cl-mass_eos_cr




    #-------------------------------descent segments------------------------

    descent1 = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed()
    descent1.tag = "Descent"
    descent1.altitude = [10.0, 5.0]                     # km
    descent1.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    descent1.planet = SUAVE.Attributes.Planets.Earth()
    descent1.config = cruise_config
    #descent1.Minf = 0.6   
    descent1.Vinf = 170.0 # m/s
    descent1.rate = 10.0                               # m/s
    descent1.m0 = mass_eos_cr  #Boeing737_800.Mass_Props.m_full          # kg
    descent1.t0 = 0.0                                 # initial time (s)
    descent1.options.N = 4
    

    sol31=SUAVE.Methods.Utilities.pseudospectral(descent1)
    mass_eos_des31=descent1.m[descent1.options.N -1]    
    print 'fuelburn descent1' , mass_eos_des31 -mass_eos_cr



    # create a descent segment: consant speed, constant descent rate
    descent = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed()
    descent.tag = "Descent"
    descent.altitude = [5.0, 0.0]                     # km
    descent.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    descent.planet = SUAVE.Attributes.Planets.Earth()
    descent.config = cruise_config
    descent.Vinf = 145.0                              # m/s
    descent.rate = 8.0                               # m/s
    descent.m0 = mass_eos_des31  #Boeing737_800.Mass_Props.m_full          # kg
    descent.t0 = 0.0                                 # initial time (s)
    descent.options.N = 4

    
    sol3=SUAVE.Methods.Utilities.pseudospectral(descent)
    mass_eos_des=descent.m[descent.options.N -1]
    
    
    print 'fuelburn descent' , mass_eos_des31 -mass_eos_des
    



#---------------------Display the results---------------------------------------------

#------------------------------------------------------------

    # run segments
    #sol1=SUAVE.Methods.Utilities.pseudospectral(climb)
    #sol2=SUAVE.Methods.Utilities.pseudospectral(cruise)
    #sol3=SUAVE.Methods.Utilities.pseudospectral(descent)

    #print 'cruise mdot' , cruise.mdot
    print 'descent weight' , descent.m
    print 'cruise cl ' ,cruise.CL
    print 'fuel burnt' , (Boeing737_800.Mass_Props.m_full-mass_eos_des)
    
    




#------------------Useful plots -------------------------------------------------------------
#----Climb---------------------------------------------------------
     ######plot solution
    title = "Climb"
    plt.subplot(5,6,1)
    plt.plot(climb1.t/60,climb1.mdot,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('mdot'); plt.title(title)
    plt.grid(True)

    plt.subplot(5,6,7)
    plt.plot(climb1.t/60,climb1.CL,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('Cl'); plt.title(title)
    plt.grid(True)
    
    plt.subplot(5,6,13)
    plt.plot(climb1.t/60,climb1.m,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('Weight'); plt.title(title)
    plt.grid(True)  
    
    title = "Climb"
    plt.subplot(5,6,19)
    plt.plot(climb1.t/60,climb1.eta,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('throttle'); plt.title(title)
    plt.grid(True)    
    
    plt.subplot(5,6,25)
    plt.plot(climb1.t/60,np.degrees(climb1.gamma),'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('Thrust Angle (deg)'); plt.title(title)
    plt.grid(True)
    
    #---------------------------------------------------
    
    title = "Climb2"
    plt.subplot(5,6,2)
    plt.plot(climb2.t/60,climb2.mdot,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('mdot'); plt.title(title)
    plt.grid(True)

    plt.subplot(5,6,8)
    plt.plot(climb2.t/60,climb2.CL,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('Cl'); plt.title(title)
    plt.grid(True)
    
    plt.subplot(5,6,14)
    plt.plot(climb2.t/60,climb2.m,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('Weight'); plt.title(title)
    plt.grid(True)  
    
    title = "Climb"
    plt.subplot(5,6,20)
    plt.plot(climb2.t/60,climb2.eta,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('throttle'); plt.title(title)
    plt.grid(True)    
    
    plt.subplot(5,6,26)
    plt.plot(climb2.t/60,np.degrees(climb2.gamma),'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('Thrust Angle (deg)'); plt.title(title)
    plt.grid(True)
    
    
    #----------------climb3--------------
    title = "Climb2"
    plt.subplot(5,6,3)
    plt.plot(climb.t/60,climb.mdot,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('mdot'); plt.title(title)
    plt.grid(True)

    plt.subplot(5,6,9)
    plt.plot(climb.t/60,climb.CL,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('Cl'); plt.title(title)
    plt.grid(True)
    
    plt.subplot(5,6,15)
    plt.plot(climb.t/60,climb.m,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('Weight'); plt.title(title)
    plt.grid(True)  
    
    title = "Climb"
    plt.subplot(5,6,21)
    plt.plot(climb.t/60,climb.eta,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('throttle'); plt.title(title)
    plt.grid(True)    
    
    plt.subplot(5,6,27)
    plt.plot(climb.t/60,np.degrees(climb.gamma),'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('Thrust Angle (deg)'); plt.title(title)
    plt.grid(True)    
    
    
    
    
    #-cruise-------------------
    

    title = "Cruise"
    plt.subplot(5,6,4)
    plt.plot(cruise.t/60,cruise.mdot,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('Thrust Angle (deg)'); plt.title(title)
    plt.grid(True)

    ###plt.subplot(5,6,10)
    ###plt.plot(cruise.t/60,cruise.CL,'o-')
    ###plt.xlabel('Time (mins)'); plt.ylabel('Cl'); plt.title(title)
    ###plt.grid(True)
    
    plt.subplot(5,6,16)
    plt.plot(cruise.t/60,cruise.m,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('Weight'); plt.title(title)
    plt.grid(True)
    
    plt.subplot(5,6,22)
    plt.plot(cruise.t/60,cruise.eta,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('throttle'); plt.title(title)
    plt.grid(True)    
    
    plt.subplot(5,6,28)
    plt.plot(cruise.t/60,np.degrees(cruise.gamma),'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('Thrust Angle (deg)'); plt.title(title)
    plt.grid(True)
    
    
    #----------descent1--------------------------

    title = "Descent"
    plt.subplot(5,6,5)
    plt.plot(descent1.t/60,descent1.mdot,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('mdot'); plt.title(title)
    plt.grid(True)

    plt.subplot(5,6,11)
    plt.plot(descent1.t/60,descent1.CL,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('Cl'); plt.title(title)
    plt.grid(True)
    
    plt.subplot(5,6,17)
    plt.plot(descent1.t/60,descent1.m,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('Weigth'); plt.title(title)
    plt.grid(True)     
    
    plt.subplot(5,6,23)
    plt.plot(descent1.t/60,descent1.eta,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('throttle'); plt.title(title)
    plt.grid(True)     
    
    plt.subplot(5,6,29)
    plt.plot(descent1.t/60,np.degrees(descent1.gamma),'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('Thrust Angle (deg)'); plt.title(title)
    plt.grid(True)    
    
    #--------------------descent 2---------------------------
    title = "Descent2"
    plt.subplot(5,6,6)
    plt.plot(descent.t/60,descent.mdot,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('mdot'); plt.title(title)
    plt.grid(True)

    plt.subplot(5,6,12)
    plt.plot(descent.t/60,descent.CL,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('Cl'); plt.title(title)
    plt.grid(True)
    
    plt.subplot(5,6,18)
    plt.plot(descent.t/60,descent.m,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('Weigth'); plt.title(title)
    plt.grid(True)     
    
    plt.subplot(5,6,24)
    plt.plot(descent.t/60,descent.eta,'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('throttle'); plt.title(title)
    plt.grid(True)     
    
    plt.subplot(5,6,30)
    plt.plot(descent.t/60,np.degrees(descent.gamma),'o-')
    plt.xlabel('Time (mins)'); plt.ylabel('Thrust Angle (deg)'); plt.title(title)
    plt.grid(True)      
 
    plt.show()
    
    
       
    return

 

 
 
 
# call main
if __name__ == '__main__':
    main()
