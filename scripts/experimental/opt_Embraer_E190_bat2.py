#by M. Vegh

import SUAVE
import numpy as np
import scipy as sp
import pylab as plt
import copy
import time
import matplotlib
matplotlib.interactive(True)
from pint import UnitRegistry
from SUAVE.Attributes import Units as Units
# ----------------------------------------------------------------------
#   Inputs
# ----------------------------------------------------------------------

def main():
   
    Ereq =1.25137847e+11                      # required energy
    #Preq=  1.00318268e+07                     #estimated power requirements for cruise
    Ereq_lis=  2.7621607e+09                 #remaining power requirement for lithium sulfur battery
    Preq_lis= 3.98197096e+06
    
    
    climb_alt_1=3.
    climb_alt_2=  8. 
    climb_alt_3= 10.866
    alpha_rc   =3.0
    alpha_tc    = -1.0   
    wing_sweep=25.
    vehicle_S        = 124.862
    Vclimb_1      = 125.0   
    Vclimb_2      =150. #m/s
    Vclimb_3      = 190.0 
    V_cruise      =230.2
    range      = 2400            #cruise range [km]
    #range        =4.12781663e+03 preliminary calc for min Weight/range
    npass      =200
    #wing_sweep=25
    """
    climb_alt_1= 7.71367305e-01 
    climb_alt_2= 3.75403122e+00
    climb_alt_3= 3.75413000e+00
    """
    """
    climb_alt_1=3.
    climb_alt_2= 8.
    climb_alt_3= 10.
    """

    #inputs=[  1.09054811e+11 ,  1.57963981e-01  , 2.66041483e-01 ,  1.41709413e+00, 3.08597883e+00 ,  6.21923270e+00 ,  7.59263894e-01 , -1.82784674e-02,   4.14179730e-04 ,  8.98849634e+01 ,  1.21002104e+02,   1.64754015e+02,   2.01648275e+02  , 2.30416367e+02  , 2.44539233e+03]
    #inputs=   [  1.31584398e+11 ,  4.54129682e+00 , -9.15734295e+00  , 2.60026466e+00,   2.63153086e+00 ,  6.02593615e+00 ,  1.19244464e+00 , -3.45626189e+00,   8.87913318e-03 ,  1.01237339e+02  , 1.55975682e+02 ,  1.64117776e+02,   1.83349067e+02  , 2.30419853e+02  , 2.86384406e+03]
   
   
    """
    Li-S
    inputs= [  8.30342833e+10  , 2.99062504e+10,   4.77626013e+06 ,  2.86536336e+00,  5.53443366e+00 ,  5.53620145e+00 ,  5.70223223e+00 ,  5.89191450e+00,
   4.71228083e+00 , -2.28259122e+00 ,  6.78616016e-08 ,  1.19645462e+02,
   1.10636483e+02 ,  1.47459194e+02 ,  1.84638188e+02  , 2.05206786e+02,
   1.85829008e+02 ,  2.30319308e+02 ,  2.22268873e+03]
    """
     
     
   
  
    inputs=  [  1.72969081e+11 ,  8.34289325e+03 ,  1.78401198e+00  , 2.14421657e+00,  5.58288581e+00 ,  5.65821025e+00 ,  5.75191743e+00  , 5.77491276e+00,   2.96603752e+00 , -4.98926921e+00 ,  8.78684362e-05  , 1.14191554e+02,   1.38908850e+02 ,  1.50408709e+02 ,  2.25516393e+02  , 1.74643398e+02,  2.06298598e+02 ,  2.30000004e+02 ,  3.58480325e+03]

   
    
    #out=sp.optimize.basinhopping(run, inpuSts, niter=1E5)
    #out=sp.optimize.fmin_bfgs(run,inputs)
    #bounds for simulated annealing
    lower_bounds=[1E11,.01,.01,0.1,0.1,4., -5., -5.,0.01, 50., 50.,50., 50., 230., 1800.]
    upper_bounds=[2E11,.01, .01, 12., 12., 12., 5., 5., 25., 200., 200., 200., 200., 300., 2300.]
    #out=sp.optimize.anneal(run, inputs, lower=lower_bounds, upper=upper_bounds)
    #bounds for bfgs
    mybounds=[(.01,None),(.01, None), (.01, None),(0., 15.), (0., 15.) ,(0., 15.),(-5.,5), (-5.,5.),(0., 25.),(0.,None),(50.,300.),(50.,300.),(50.,300.),(50.,300.),(1900.,None)]
    
    #out=sp.optimize.fmin_l_bfgs_b(run,x0=inputs, bounds=mybounds, approx_grad=True)
    
    #print mybounds
    
    #out=sp.optimize.fmin(run, inputs)
    if 'out' in locals():
        
        obj_out=disp_results(out)
    else:
        obj_out=disp_results(inputs)


# ----------------------------------------------------------------------
#   Calls
# ----------------------------------------------------------------------
def run(inputs):                #sizing loop to enable optimization
    target_range=4400.           #minimum flight range of the aircraft (constraint)
    i=0 #counter for inputs
    
    
    #mass=[ 100034.162173]
    #mass=[ 113343.414181]                 
    mass=[ 67502.3079434 ]       #mass guess, 2000 W-h/kg
    Ereq=inputs[i];              i+=1
    Ereq_lis=inputs[i];          i+=1
    Preq_lis=inputs[i];          i+=1
    climb_alt_1=inputs[i];       i+=1
    climb_alt_2=inputs[i];       i+=1
    climb_alt_3=inputs[i];       i+=1
    climb_alt_4=inputs[i];       i+=1
    climb_alt_5=inputs[i];       i+=1
    alpha_rc=inputs[i];          i+=1
    alpha_tc=inputs[i];          i+=1
    wing_sweep=inputs[i];        i+=1
    vehicle_S=inputs[i];         i+=1
    Vclimb_1=inputs[i];          i+=1
    Vclimb_2=inputs[i];          i+=1
    Vclimb_3=inputs[i];          i+=1
    Vclimb_4=inputs[i];          i+=1
    Vclimb_5=inputs[i];          i+=1
    V_cruise =inputs[i];         i+=1
    cruise_range=inputs[i];      i+=1
    
    tol=.1 #difference in mass in kg between iterations
    dm=10000.
    max_iter=1000
    j=0
     
   
    while abs(dm)>tol:      #size the vehicle
    
        vehicle = define_vehicle(mass[j],Ereq, Ereq_lis, Preq_lis, climb_alt_5,wing_sweep, alpha_rc, alpha_tc, vehicle_S)
        mass.append(vehicle.Mass_Props.m_end)
        dm=mass[j+1]-mass[j]
        j+=1
        if j>max_iter:
            print "maximum number of iterations exceeded"
    #vehicle sized and defined now
    
    # define the mission
    mission = define_mission(vehicle,climb_alt_1,climb_alt_2,climb_alt_3, climb_alt_4,climb_alt_5, Vclimb_1, Vclimb_2, Vclimb_3, Vclimb_4,Vclimb_5,V_cruise , cruise_range)
    
    # evaluate the mission
    results = evaluate_mission(vehicle,mission)
    
    #post_process(vehicle,mission,results)
    #add penalty function for battery
    battery = vehicle.Energy.Storages['Battery']
    battery_lis = vehicle.Energy.Storages['Battery_Li_S']
    #find minimum energy for each battery
    min_Ebat=np.zeros(len(results.Segments))
    min_Ebat_lis=np.zeros(len(results.Segments))
    max_alpha=np.zeros(len(results.Segments))
    min_alpha=np.zeros(len(results.Segments))
    for i in range(len(results.Segments)):
        min_Ebat[i]=min(results.Segments[i].Ecurrent)
        min_Ebat_lis[i]=min(results.Segments[i].Ecurrent_lis)
        max_alpha[i]=max(np.degrees(results.Segments[i].alpha))
        min_alpha[i]=min(np.degrees(results.Segments[i].alpha))
        
        
    min_Ebat=min(min_Ebat)
    min_Ebat_lis=min(min_Ebat_lis)
    max_alpha=max(max_alpha)
    min_alpha=min(min_alpha)
    
    #results.Segments[i].Ecurrent_lis=battery_lis.CurrentEnergy
    penalty_energy=abs(min(min_Ebat/battery.TotalEnergy, min_Ebat_lis/battery_lis.TotalEnergy,0.))*10.**8.
    penalty_bat_neg=(10.**4.)*abs(min(0.,Ereq_lis, Preq_lis))   #penalty to make sure none of the inputs are negative
    
    
    vehicle.Mass_Props.m_full=results.Segments[-1].m[-1]    #make the vehicle the final mass
    vehicle.Mass_Props.m_full+=penalty_energy+vehicle.penalty_power+10000.*abs(min(0, battery_lis.MaxPower, battery.MaxPower))+penalty_bat_neg
    
    #add penalty functions for twist, ensuring that trailing edge is >-5 degrees
    vehicle.Mass_Props.m_full+=100000.*abs(min(0, alpha_tc+5))
    vehicle.Mass_Props.m_full+=100000.*abs(max(0, alpha_rc-5))
    #now add penalty function if range is not met
    vehicle.Mass_Props.m_full+=10000.*abs(min(results.Segments[-1].vectors.r[-1,0]/1000-target_range,0,))
    #add penalty function for washin
    vehicle.Mass_Props.m_full+=10000.*abs(min(0, alpha_rc-alpha_tc))
    
    #make sure that angle of attack is below 30 degrees but above -30 degrees
    vehicle.Mass_Props.m_full+=10000.*abs(min(0, 30-max_alpha))+10000.*abs(min(0, 30+min_alpha))
    
    #now add penalty function if wing sweep is too high
    vehicle.Mass_Props.m_full+=10000.*abs(min(0, 30.-wing_sweep, wing_sweep))
    
    #penalty function in case altitude segments don't match up
    vehicle.Mass_Props.m_full+=100000*abs(min(climb_alt_5-climb_alt_4, climb_alt_5-climb_alt_3, climb_alt_5-climb_alt_2, climb_alt_5-climb_alt_1,
    climb_alt_4-climb_alt_3, climb_alt_4-climb_alt_2, climb_alt_4-climb_alt_1,
    climb_alt_3-climb_alt_2, climb_alt_2-climb_alt_1, climb_alt_3-climb_alt_1, 0.))
      #penalty function to make sure that cruise velocity >=E-190 cruise
    vehicle.Mass_Props.m_full+=100000*abs(max(0,230.-V_cruise))                                                         
    #print vehicle.Mass_Props.m_full/(results.Segments[-1].vectors.r[-1,0]/1000.),' ', vehicle.Mass_Props.m_full, ' ',results.Segments[-1].vectors.r[-1,0]/1000. , inputs
    print vehicle.Mass_Props.m_full, ' ',results.Segments[-1].vectors.r[-1,0]/1000. , inputs
    return vehicle.Mass_Props.m_full#/(results.Segments[-1].vectors.r[-1,0]/1000.) 

    
def disp_results(inputs):     #run a mission while displaying results
    i=0
    mass=[67502.3079434 ]       #mass guess
    Ereq=inputs[i];              i+=1
  
    Ereq_lis=inputs[i];          i+=1
    Preq_lis=inputs[i];          i+=1
    climb_alt_1=inputs[i];       i+=1
    climb_alt_2=inputs[i];       i+=1
    climb_alt_3=inputs[i];       i+=1
    climb_alt_4=inputs[i];       i+=1
    climb_alt_5=inputs[i];       i+=1
    alpha_rc=inputs[i];          i+=1
    alpha_tc=inputs[i];          i+=1
    wing_sweep=inputs[i];        i+=1
    vehicle_S=inputs[i];         i+=1
    Vclimb_1=inputs[i];          i+=1
    Vclimb_2=inputs[i];          i+=1
    Vclimb_3=inputs[i];          i+=1
    Vclimb_4=inputs[i];          i+=1
    Vclimb_5=inputs[i];          i+=1
    V_cruise =inputs[i];         i+=1
    cruise_range=inputs[i];      i+=1
   
    tol=.1 #difference in mass in kg between iterations
    dm=10000.
    max_iter=1000
    j=0
    #print resulting inputs
    print 'climb_alt1=', climb_alt_1
    print 'Vclimb_1=', Vclimb_1
    print 'climb_alt2=', climb_alt_2
    print 'Vclimb_2=', Vclimb_2
    print 'climb_alt3=', climb_alt_3
    print 'Vclimb_3=', Vclimb_3
    print 'climb_alt4=', climb_alt_4
    print 'Vclimb_4=', Vclimb_4
    print 'climb_alt5=', climb_alt_5
    print 'Vclimb_5=', Vclimb_5
    print 'V_cruise=' ,V_cruise
    print 'alpha_rc=', alpha_rc
    print 'alpha_tc=', alpha_tc
    print 'wing_sweep=', wing_sweep
    print 'Sref=', vehicle_S
    print 'cruise range=', cruise_range
    
    t1=time.time()
    
    while abs(dm)>tol:      #size the vehicle
    
        vehicle = define_vehicle(mass[j],Ereq, Ereq_lis, Preq_lis, climb_alt_4,wing_sweep, alpha_rc, alpha_tc, vehicle_S)
        mass.append(vehicle.Mass_Props.m_end)
        dm=mass[j+1]-mass[j]
        j+=1
        if j>max_iter:
            print "maximum number of iterations exceeded"
    t2=time.time()
    #vehicle sized and defined now
    
    # define the mission
    mission = define_mission(vehicle,climb_alt_1,climb_alt_2,climb_alt_3, climb_alt_4, climb_alt_5,Vclimb_1, Vclimb_2, Vclimb_3, Vclimb_4, Vclimb_5, V_cruise , cruise_range)
    t3=time.time()
    # evaluate the mission
    results = evaluate_mission(vehicle,mission)
    t4=time.time()
    # plot results
    print 'total range=',results.Segments[-1].vectors.r[-1,0]/1000 
    vehicle.Mass_Props.m_full=results.Segments[-1].m[-1]    #make the vehicle the final mass
    print vehicle.Mass_Props.m_full, inputs
    battery = vehicle.Energy.Storages['Battery']
    battery_lis = vehicle.Energy.Storages['Battery_Li_S']
    #print battery_lis
    
    post_process(vehicle,mission,results)
  
   
    """
    print "vehicle definition=", t2-t1
    print "mission definition=", t3-t2
    print "mission evaluation=", t4-t3
    """
   
    

# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------

def define_vehicle(Mguess,Ereq, Ereq_lis, Preq_lis, max_alt,wing_sweep,alpha_rc, alpha_tc, vehicle_S   ):
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Embraer E190'

    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    
    vehicle.Nult      = 1.5 * 2.5                       # Ultimate load
    vehicle.TOW       = Mguess # Maximum takeoff weight in kilograms
    vehicle.zfw       = Mguess # Maximum zero fuel weight in kilograms
    vehicle.Nlim      = 2.5                       # Limit Load
    vehicle.num_eng   = 2.                        # Number of engines on the aircraft
    vehicle.num_pax   = 110.                      # Number of passengers
    vehicle.wt_cargo  = 0.  * Units.kilogram  # Mass of cargo
    vehicle.num_seats = 110.                      # Number of seats on aircraft
    vehicle.ctrl      = "partially powered"       # Specify fully powered, partially powered or anything else is fully aerodynamic
    vehicle.ac        = "medium-range"              # Specify what type of aircraft you have
    vehicle.w2h       = 16.     * Units.meters    # Length from the mean aerodynamic center of wing to mean aerodynamic center of the horizontal tail
    #vehicle.delta     =wing_sweep
    vehicle.S        = vehicle_S 
    # mass properties
    vehicle.Mass_Props.m_full       = Mguess   # kg
    vehicle.Mass_Props.m_empty      = Mguess   # kg
    
    
 

    # basic parameters
    #vehicle.delta    = 25.0                     # deg
                  # 
    #vehicle.A_engine = np.pi*(0.9525)**2   
    vehicle.A_engine = 2.85*2
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Main Wing'
    """
    wing.sref      = vehicle.S       #
    wing.ar        = 8             #
    #wing.span      = 35.66         #
    wing.span      =(wing.ar*wing.sref)**.5
    wing.sweep     =wing_sweep*np.pi/180.
    #wing.sweep     = 25*np.pi/180. #
    wing.symmetric = True          #
    wing.t_c       = 0.1           #
    wing.taper     = 0.16          #
    wing.root_chord= 7.88         #meters
    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    Nult_wing      = 3.75
    """
    wing.gross_area    = vehicle.S    * Units.meter**2  # Wing gross area in square meters
    wing.sref          = vehicle.S
    wing.ar            = 8.3  
    wing.span          = (wing.ar*wing.sref)**.5    * Units.meter     # Span in meters
    wing.taper         = 0.28                       # Taper ratio
    wing.t_c           = 0.105                      # Thickness-to-chord ratio
    wing.sweep         = wing_sweep     * Units.deg       # sweep angle in degrees
    wing.c_r           = 5.4     * Units.meter     # Wing exposed root chord length
    wing.mac           = 12.     * Units.ft    # Length of the mean aerodynamic chord of the wing
    wing.chord_mac     =wing.mac
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    wing.chord_mac   = 12.                 #
    wing.S_exposed   = 0.8*wing.sref  #
    wing.S_affected  = 0.6*wing.sref  #
    wing.e           = 1.0                   #
    
    #wing.alpha_tc    = -1.0                   #
    #wing.alpha_rc   =3.0
    wing.alpha_rc   =alpha_rc
    wing.alpha_tc    =alpha_tc 
 
    wing.highlift    = False                 
    
    #wing.hl          = 1                     #
    #wing.flaps_chord = 20                    #
    #wing.flaps_angle = 20                    #
    #wing.slats_angle = 10                    #
    """
    m_wing=SUAVE.Methods.Weights.Correlations.Tube_Wing.wing_main(wing.area_wetted*(Units.meter**2),wing.span*Units.meter,
    wing.taper,wing.t_c,wing.sweep*Units.rad,Nult_wing,Mguess*Units.kg,Mguess*Units.kg)
    wing.Mass_Props.mass=m_wing
    main_wing=wing  #save variable for later correlation use
    """
    # add to vehicle
    vehicle.append_component(wing)
    
    
    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    horizontal = SUAVE.Components.Wings.Wing()

    horizontal.area    = 26.     * Units.meters**2 # Area of the horizontal tail
    horizontal.span    = 12.08     * Units.meters    # Span of the horizontal tail
    horizontal.sweep   = 34.5 * Units.deg       # Sweep of the horizontal tail
    horizontal.mac     = 2.4      * Units.meters    # Length of the mean aerodynamic chord of the horizontal tail
    horizontal.t_c     = 0.11                      # Thickness-to-chord ratio of the horizontal tail
    horizontal.exposed = 0.9                         # Fraction of horizontal tail area exposed
  
    horizontal.tag = 'Horizontal Stabilizer'
    horizontal.sref      = horizontal.area         #
    horizontal.ar        = 5.5         #
    horizontal.symmetric = True          
    horizontal.t_c       = 0.11          #
    horizontal.taper     = 0.11           #
    
    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(horizontal)
    
    horizontal.chord_mac  = 8.0                   #
    horizontal.S_exposed  = 0.8*horizontal.area_wetted  #
    horizontal.S_affected = 0.6*horizontal.area_wetted  #  
    #wing.Cl         = 0.2                   #
    horizontal.e          = 0.9                   #
    horizontal.alpha_rc   = 2.0                   #
    horizontal.alpha_tc   = 2.0                   #
  
    # add to vehicle
    vehicle.append_component(horizontal)
    """
    wing.sref      =32.488         #
    wing.ar        = 6.16          #
    wing.span=(wing.sref*wing.ar)**.5
    #wing.span      = 100           #
    wing.sweep     = wing_sweep*np.pi/180.  #
    wing.symmetric = True          
    wing.t_c       = 0.08          #
    wing.taper     = 0.4           #
    l_w2h          =16.764         #meters; estimated of length from MAC of wing to h_stabilizer
    Nult_h_stab    =3.75
    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chord_mac  = 8.0                   #
    wing.S_exposed  = 0.8*wing.area_wetted  #
    wing.S_affected = 0.6*wing.area_wetted  #  
    #wing.Cl         = 0.2                   #
    wing.e          = 0.9                   #
    wing.alpha_rc   = 3.0                   #
    wing.alpha_tc   = 3.0                   #
    Nult_h_stab=3.5
    
    
    m_h_stab=SUAVE.Methods.Weights.Correlations.Tube_Wing.tail_horizontal(wing.span*Units.meter,wing.sweep*Units.rad,Nult_h_stab,wing.area_wetted*(Units.meter**2)
    ,Mguess*Units.kg,main_wing.chord_mac*Units.meter, wing.chord_mac*Units.meter,l_w2h*Units.meter,wing.t_c, .8)
    # add to vehicle
  
    vehicle.append_component(wing)
    h_stab=wing
    """
    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    vertical = SUAVE.Components.Wings.Wing()
    vertical.tag = 'Vertical Stabilizer'    
    
    vertical.sref      = 16.0        #
    vertical.ar        = 1.7          #
    #wing.span      = 100           #
    vertical.sweep     = wing_sweep * Units.deg  #
    vertical.symmetric = False    
    vertical.t_c       = 0.12          #
    vertical.taper     = 0.10          #
    
    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(vertical)
    
    vertical.chord_mac  = 11.0                  #
    vertical.S_exposed  = 0.8*vertical.area_wetted  #
    vertical.S_affected = 0.6*vertical.area_wetted  #  
    #wing.Cl        = 0.002                  #
    vertical.e          = 0.9                   #
    vertical.alpha_rc   = 0.0                   #
    vertical.alpha_tc   = 0.0                   #
    
    
    vertical.area      = vertical.sref     * Units.meters**2 # Area of the vertical tail
    vertical.span      = 5.3     * Units.meters    # Span of the vertical tail
    vertical.t_tail    = "no"                      # Set to "yes" for a T-tail

    # add to vehicle
    vehicle.append_component(vertical)
    
    """
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Vertical Stabilizer'    
    
    wing.sref      = 32.488        #
    wing.ar        = 1.91          #
    #wing.span      = 100           #
    wing.span      =(wing.sref*wing.ar)**.5
    wing.sweep     = wing_sweep*np.pi/180.  #
    wing.symmetric = False    
    wing.t_c       = 0.08          #
    wing.taper     = 0.25          #
    Nult_v_stab    = 3.75
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
    m_v_stab=SUAVE.Methods.Weights.Correlations.Tube_Wing.tail_vertical((wing.area_wetted)*Units.meter**2,Nult_v_stab,
    wing.span*Units.meter,Mguess*Units.kg,wing.t_c, wing.sweep*Units.rad,wing.S_exposed*(Units.meter**2),"no")
    v_stab.Mass_Props.mass=m_v_stab.wt_tail_vertical+m_v_stab.wt_rudder
    # add to vehicle
    vehicle.append_component(wing)
    """

    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'Fuselage'
    
    fuselage.num_coach_seats = 114.  #
    fuselage.seat_pitch      = 0.7455    # m
    fuselage.seats_abreast   = 4    #
    fuselage.fineness_nose   = 2.0  #
    fuselage.fineness_tail   = 3.0  #
    fuselage.fwdspace        = 0.    #
    fuselage.aftspace        = 0.    #
    fuselage.width           = 3.0  #
    fuselage.height          = 3.4  #
    fuselage.area            = 320.      * Units.meter**2  
    
    # size fuselage planform
    SUAVE.Geometry.Two_Dimensional.Planform.fuselage_planform(fuselage)
    
    # add to vehicle
    vehicle.append_component(fuselage)
    """
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'Fuselage'
    
    fuselage.num_coach_seats = 110  #
    fuselage.seats_abreast=6.
    fuselage.seat_pitch      = 1.    #meters
    fuselage.fineness_nose   = 1.6  #
    fuselage.fineness_tail   = 2.    #
    fuselage.fwdspace        = 6.    #
    fuselage.aftspace        = 5.    #
    fuselage.width           = 4.    #
    fuselage.height          = 4.    #
    fuselage.length          =39.4716 #meters
    Nult_fus=2.5                    #ultimate load factor
    # size fuselage planform
    SUAVE.Geometry.Two_Dimensional.Planform.fuselage_planform(fuselage)
    """
    ################
    
    # ------------------------------------------------------------------
    #  Propulsion
    # ------------------------------------------------------------------
    
    
    atm = SUAVE.Attributes.Atmospheres.Earth.International_Standard()
    p1, T1, rho1, a1, mew1 = atm.compute_values(0.)
    p2, T2, rho2, a2, mew2 = atm.compute_values(max_alt)
    
    sizing_segment = SUAVE.Components.Propulsors.Segments.Segment()
    sizing_segment.M   = 0.8          
    sizing_segment.alt = max_alt
    sizing_segment.T   = T2           
    
    sizing_segment.p   = p2     
    #create battery
    battery = SUAVE.Components.Energy.Storages.Battery()
    battery_lis = SUAVE.Components.Energy.Storages.Battery()
    battery_lis.type='Li_S'
    battery_lis.tag='Battery_Li_S'
    battery.tag = 'Battery'
    battery.type ='Li-Air'
    # attributes
    battery.SpecificEnergy = 2000.0 #W-h/kW
    battery.SpecificPower  = 0.64   #kW/kg    
    
    #ducted fan
    DuctedFan= SUAVE.Components.Propulsors.Ducted_Fan_Bat()
    DuctedFan.propellant = SUAVE.Attributes.Propellants.Aviation_Gasoline()
    DuctedFan.diffuser_pressure_ratio = 0.98
    DuctedFan.fan_pressure_ratio = 1.65
    DuctedFan.fan_nozzle_pressure_ratio = 0.99
    DuctedFan.design_thrust = 20000.0*3.
    DuctedFan.no_of_engines=2.0   
    DuctedFan.engine_sizing_ductedfan(sizing_segment,battery)   #calling the engine sizing method 
    vehicle.append_component(DuctedFan)
    vehicle.append_component( battery )
    vehicle.append_component( battery_lis )

    SUAVE.Methods.Power.size_opt_battery(battery_lis,Ereq_lis, Preq_lis) #create an optimum battery from these requirements
    
    battery.Mass_Props.mass=Ereq/(battery.SpecificEnergy*3600)
    
    battery.MaxPower=battery.Mass_Props.mass*(battery.SpecificPower*1000) #find max power available from battery
    battery.TotalEnergy=battery.Mass_Props.mass*battery.SpecificEnergy*3600
    battery.CurrentEnergy=battery.TotalEnergy
    
    m_air=battery.find_mass_gain()    #find mass gain of battery throughout mission and size vehicle

    #vehicle.Mass_Props.m_empty+=battery.Mass_Props.mass-16416.4+3600
    
    #now add the electric motor weight
    motor_mass=DuctedFan.no_of_engines*SUAVE.Methods.Weights.Correlations.Propulsion.air_cooled_motor((battery.MaxPower+Preq_lis)*Units.watts/DuctedFan.no_of_engines)
    engine_mass=SUAVE.Methods.Weights.Correlations.Propulsion.engine_jet(DuctedFan.design_thrust*Units.N)
    propulsion_mass=SUAVE.Methods.Weights.Correlations.Propulsion.integrated_propulsion(motor_mass/DuctedFan.no_of_engines,DuctedFan.no_of_engines)
   
    propulsion_mass=motor_mass
    DuctedFan.Mass_Props.mass=propulsion_mass*DuctedFan.no_of_engines
    DuctedFan.battery=battery
    DuctedFan.battery_lis=battery_lis
    fuselage.diff_p=max(abs(p2-p1*2./3.),0)   #assume its pressurized to 2/3 atmospheric pressure
    """
    fus_weight=SUAVE.Methods.Weights.Correlations.Tube_Wing.tube(S_fus*(Units.meter**2), 
    diff_p_fus*Units.pascal,fuselage.width*Units.meter,fuselage.height*Units.meter, fuselage.length*Units.meter,
    Nult_fus, Mguess*Units.kg, main_wing.Mass_Props.mass*Units.kg, (DuctedFan.Mass_Props.mass+motor_mass)*Units.kg, main_wing.root_chord*Units.meter)
  
    fuselage.Mass_Props.mass=fus_weight
    vehicle.append_component(fuselage)
    

    #######Other weight correlations##########
    m_landing_gear=SUAVE.Methods.Weights.Correlations.Tube_Wing.landing_gear(Mguess*Units.kg)*1.5 ##factor of 1.5 is for increased landing weight
    systems=SUAVE.Methods.Weights.Correlations.Tube_Wing.systems(fuselage.num_coach_seats, 'fully powered', h_stab.area_wetted*(Units.meter**2), v_stab.area_wetted*(Units.meter**2),
    main_wing.area_wetted*(Units.meter**2), "medium-range")
    m_pl=10454.54545*2.
    m_systems=systems.wt_systems
    """
   
    
    # ------------------------------------------------------------------
    #   Simple Aerodynamics Model
    # ------------------------------------------------------------------ 
    
    aerodynamics = SUAVE.Attributes.Aerodynamics.PASS_Aero()
    aerodynamics.initialize(vehicle)
    vehicle.Aerodynamics = aerodynamics
    

    # ------------------------------------------------------------------
    #   Define Configurations
    # ------------------------------------------------------------------


    
    
    # ------------------------------------------------------------------
    #   Define New Gross Takeoff Weight
    # ------------------------------------------------------------------
    #now add component weights to the gross takeoff weight of the vehicle
    m_fuel=0.
    #m_air=0.              #mass gain from the lithium air battery 
    engine1 = SUAVE.Structure.Data()
    engine1.thrust_sls  =0.001;  #dummy variable to make sizing easier
    weight =SUAVE.Methods.Weights.Correlations. Tube_Wing.empty(engine1,wing,vehicle,fuselage,horizontal,vertical)
    vehicle.Mass_Props.m_full=weight.empty+battery.Mass_Props.mass+battery_lis.Mass_Props.mass+DuctedFan.Mass_Props.mass+(vehicle.num_pax+4.)*250.*Units.lb
    """
    vehicle.Mass_Props.m_full=fuselage.Mass_Props.mass+main_wing.Mass_Props.mass+battery.Mass_Props.mass+battery_lis.Mass_Props.mass+m_landing_gear+ \
    v_stab.Mass_Props.mass+h_stab.Mass_Props.mass+m_pl+DuctedFan.Mass_Props.mass+m_fuel+m_systems+m_air
    """
    #use penalty function to ensure that battery energy is always positive
    vehicle.Mass_Props.m_full+=10000.*abs(min(0, battery_lis.TotalEnergy, battery.TotalEnergy))
    vehicle.Mass_Props.m_empty=vehicle.Mass_Props.m_full
    vehicle.Mass_Props.m_end=vehicle.Mass_Props.m_full+m_air
 
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

###############################################################################################################################################
# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------
def define_mission(vehicle,climb_alt_1,climb_alt_2,climb_alt_3, climb_alt_4, climb_alt_5,Vclimb_1, Vclimb_2, Vclimb_3, Vclimb_4,Vclimb_5, V_cruise , cruise_range):
   
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
    segment.altitude   = [0.0, climb_alt_1]   # km
    
    # pick two:
    segment.Vinf       = Vclimb_1        # m/s
    #segment.rate       = 6.0          # m/s
    segment.rate       = 2000.*(Units.ft/Units.minute) 
    #segment.psi        = 8.5          # deg
    
    # add to mission
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
    #segment.altitude   = [3.0,8.0] # km  
    segment.altitude   = [mission.Segments["Climb - 1"].altitude[-1], climb_alt_2]  # km
    
    # pick two:
    #segment.Vinf       = 190.0       # m/s
    segment.Vinf       =Vclimb_2
    #segment.rate       = 6.0         # m/s
    segment.rate        =1500.*(Units.ft/Units.minute)
    #segment.psi        = 15.0        # deg
    
    # add to mission
    mission.append_segment(segment)

    
    # ------------------------------------------------------------------
    #   Third Climb Segment: constant velocity, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed()
    segment.tag = "Climb - 3"

    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet   
    #segment.altitude   = [8.0, 10.668] # km    
    segment.altitude   = [mission.Segments["Climb - 2"].altitude[-1], climb_alt_3] # km
    
    # pick two:
    #segment.Vinf        = 226.0        # m/s   
    segment.Vinf        = Vclimb_3
    #segment.rate        = 3.0          # m/s
    segment.rate       =1500.*(Units.ft/Units.minute)
    #segment.psi         = 15.0         # deg
    
    # add to mission
    mission.append_segment(segment)
    
     # ------------------------------------------------------------------
    #   Fourth Climb Segment: constant velocity, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed()
    segment.tag = "Climb - 4"

    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet   
    #segment.altitude   = [8.0, 10.668] # km    
    segment.altitude   = [mission.Segments["Climb - 3"].altitude[-1], climb_alt_4] # km
    
    # pick two:
    #segment.Vinf        = 226.0        # m/s   
    segment.Vinf        = Vclimb_4
    #segment.rate        = 3.0          # m/s
    segment.rate       =900.*(Units.ft/Units.minute)
    #segment.psi         = 15.0         # deg
    
    # add to mission
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------
    #   Fifth Climb Segment: Constant Speed, Constant Climb Rate
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed()
    segment.tag = "Climb - 5"

    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet        
    segment.altitude   = [mission.Segments["Climb - 4"].altitude[-1], climb_alt_5] # km
    climb_alt5=segment.altitude
    # pick two:
    segment.Vinf       = Vclimb_5
    #segment.Minf        = 0.78        # m/s   
    #segment.rate        = 1.0         # m/s
    segment.rate       =200.*(Units.ft/Units.minute)
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
    #segment.altitude   = 10.668   
    segment.altitude   = mission.Segments["Climb - 5"].altitude[-1]    # km
    segment.Vinf       = V_cruise    # m/s
    segment.range      =cruise_range
    #segment.range      = 2400  # km
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   First Descent Segment: constant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed()
    segment.tag = "Descent - 1"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet    
    #segment.altitude   = [10.668, 5.0]  # km
    segment.altitude   = [mission.Segments["Cruise"].altitude , climb_alt_3]  # km
    segment.Vinf       = 230.0          # m/s
    #segment.rate       = 5.0            # m/s
    segment.rate         =1600*(Units.ft/Units.minute)
    
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
    segment.altitude   = [mission.Segments["Descent - 1"].altitude[-1], climb_alt_2]  # km
    #segment.altitude   = [5., 0.0]
    segment.Vinf       = 200.0       # m/s
    #segment.rate       = 5.0         # m/s
    segment.rate         =1500*(Units.ft/Units.minute)
    # append to mission
    mission.append_segment(segment)

       # ------------------------------------------------------------------    
    #   Third Descent Segment: constant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed()
    segment.tag = "Descent -3"

    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet   
    segment.altitude   = [mission.Segments["Descent - 2"].altitude[-1], 0.]  # km
    segment.Vinf       = 140.0       # m/s
    #segment.rate       = 5.0         # m/s
    segment.rate         =1500*(Units.ft/Units.minute)
    # append to mission
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   Mission definition complete   
   
    #vehicle.Mass_Props.m_empty+=motor_mass

    #print vehicle.Mass_Props.m_full
    
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
    SUAVE.Methods.Results.compute_energies(results,summary=False)
    SUAVE.Methods.Results.compute_efficiencies(results)
    SUAVE.Methods.Results.compute_velocity_increments(results)
    SUAVE.Methods.Results.compute_alpha(results)    
    
    battery = vehicle.Energy.Storages['Battery']
    battery_lis = vehicle.Energy.Storages['Battery_Li_S']
    Pbat_loss=np.zeros_like(results.Segments[0].P_e)   #initialize battery losses
    Ecurrent=np.zeros_like(results.Segments[0].t)      #initialize battery energies
    Ecurrent_lis=np.zeros_like(results.Segments[0].t)
    penalty_energy=0.                                  #initialize penalty functions
    penalty_power=0.
    #now run the batteries for the mission
    j=0
    
    for i in range(len(results.Segments)):
       
        results.Segments[i].Ecurrent=np.zeros_like(results.Segments[i].t)
        results.Segments[i].Ecurrent_lis=np.zeros_like(results.Segments[i].t)
        if i==0:
            results.Segments[i].Ecurrent[0]=battery.TotalEnergy
            results.Segments[i].Ecurrent_lis[0]=battery_lis.TotalEnergy
        if i!=0 and j!=0:
            results.Segments[i].Ecurrent[0]=battery.CurrentEnergy #assign energy at end of segment to next segment 
            results.Segments[i].Ecurrent_lis[0]=battery_lis.CurrentEnergy #assign energy at end of segment to next segment 
        for j in range(len(results.Segments[i].P_e)):
            if battery.MaxPower>=results.Segments[i].P_e[j]:
                Ploss_lis=0
                if j!=0:
                    [Ploss,mdot]=battery(results.Segments[i].P_e[j], (results.Segments[i].t[j]-results.Segments[i].t[j-1]))
                    
                elif i!=0 and j==0:
                    [Ploss,mdot]=battery(results.Segments[i].P_e[j], (results.Segments[i].t[j]-results.Segments[i-1].t[-2]))
                elif j==0 and i==0: 
                    [Ploss,mdot]=battery(results.Segments[i].P_e[j], (results.Segments[i].t[j+1]-results.Segments[i].t[j]))
            else: #Li-air battery cannot meet total power requirements
                if j!=0:
                     [Ploss,mdot]=battery(battery.MaxPower, (results.Segments[i].t[j]-results.Segments[i].t[j-1]))
                     Ploss_lis=battery_lis(results.Segments[i].P_e[j]-battery.MaxPower, (results.Segments[i].t[j]-results.Segments[i].t[j-1]))
                     
                elif i!=0 and j==0:
                    [Ploss,mdot]=battery(battery.MaxPower, (results.Segments[i].t[j]-results.Segments[i-1].t[-1]))
                    Ploss_lis=battery_lis(battery.MaxPower-results.Segments[i].P_e[j], (results.Segments[i].t[j]-results.Segments[i-1].t[-1]))
                   
                elif j==0 and i==0: 
                    [Ploss,mdot]=battery(battery.MaxPower, (results.Segments[i].t[j+1]-results.Segments[i].t[j]))
                    Ploss_lis=battery_lis(battery.MaxPower-results.Segments[i].P_e[j], (results.Segments[i].t[j+1]-results.Segments[i].t[j]))
                    
                if results.Segments[i].P_e[j]>battery.MaxPower+battery_lis.MaxPower:
                    penalty_power=results.Segments[i].P_e[j]
                    
            vehicle.penalty_power=penalty_power  #save penalty function to vehicle for outputting
            results.Segments[i].mdot[j]=mdot
            results.Segments[i].Ecurrent[j]=battery.CurrentEnergy
            results.Segments[i].Ecurrent_lis[j]=battery_lis.CurrentEnergy
            results.Segments[i].P_e[j]+=Ploss+Ploss_lis
   
    
    #add lithium air battery mass gain to weight of the vehicle
   
    #vehicle.Mass_Props.m_full+=m_li_air
    
    return results

# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def post_process(vehicle,mission,results):
    battery = vehicle.Energy.Storages['Battery']
    battery_lis = vehicle.Energy.Storages['Battery_Li_S']
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
    #   Vehicle Mass
    # ------------------------------------------------------------------
    title = "Vehicle Mass"
    plt.figure(3)
    for i in range(len(results.Segments)):
        #plt.plot(results.Segments[i].t/60,mission.m0 - results.Segments[i].m,'bo-')
        plt.plot(results.Segments[i].t/60,results.Segments[i].m,'bo-')
        
    plt.xlabel('Time (mins)'); plt.ylabel('Vehicle Mass(kg)'); plt.title(title)
    plt.grid(True)
    
    
    # ------------------------------------------------------------------    
    #   Mass Gain Rate
    # ------------------------------------------------------------------
    plt.figure(4)
    title = "Battery Mass Gain Rate"
    
    for i in range(len(results.Segments)):
        plt.plot(results.Segments[i].t/60,-results.Segments[i].mdot,'bo-')
    plt.xlabel('Time (mins)'); plt.ylabel('Battery Mass Accumulation (kg/s)'); plt.title(title)
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
    #   Horizontal Distance
    # ------------------------------------------------------------------
    
    title = "Ground Distance"
    plt.figure(title)
    for i in range(len(results.Segments)):
        plt.plot(results.Segments[i].t/60,results.Segments[i].vectors.r[:,0]/1000,'bo-')
    plt.xlabel('Time (mins)'); plt.ylabel('ground distance(km)'); plt.title(title)
    plt.grid(True)
    
    
    # ------------------------------------------------------------------    
    #   Energy
    # ------------------------------------------------------------------
    
    title = "Energy"
    plt.figure(title)
    for i in range(len(results.Segments)):
        plt.plot(results.Segments[i].t/60, results.Segments[i].Ecurrent/battery.TotalEnergy,'bo-')
        
      
    for i in range(len(results.Segments)):
        plt.plot(results.Segments[i].t/60, results.Segments[i].Ecurrent_lis/battery_lis.TotalEnergy,'ko-')
      
        results.Segments[i].Ecurrent_lis/battery_lis.TotalEnergy
    plt.xlabel('Time (mins)'); plt.ylabel('State of Charge of the Battery'); plt.title(title)
    plt.grid(True)

    # ------------------------------------------------------------------    
    #   Power
    # ------------------------------------------------------------------
    
    
    plt.figure(8)
    title = "Electrical Power"
    for i in range(len(results.Segments)):
        plt.plot(results.Segments[i].t/60,results.Segments[i].P_e,'bo-')

    plt.xlabel('Time (mins)'); plt.ylabel('Electrical Power (Watts)'); plt.title(title)
    plt.grid(True)
    
    # ------------------------------------------------------------------    
    #   Aerodynamics
    # ------------------------------------------------------------------
    title = "Aerodynamics"
    plt.figure(9)  
    plt.title(title)
    for segment in results.Segments.values():

        plt.subplot(3,1,1)
        plt.plot( segment.t / Units.minute , 
                  segment.L ,
                  'bo-' )
        plt.xlabel('Time (min)')
        plt.ylabel('Lift (N)')
        plt.grid(True)
        
        plt.subplot(3,1,2)
        plt.plot( segment.t / Units.minute , 
                  segment.D ,
                  'bo-' )
        plt.xlabel('Time (min)')
        plt.ylabel('Drag (N)')
        plt.grid(True)
        
        plt.subplot(3,1,3)
        plt.plot( segment.t / Units.minute , 
                  segment.F ,
                  'bo-' )
        plt.xlabel('Time (min)')
        plt.ylabel('Thrust (N)')
        plt.grid(True)
        
    # ------------------------------------------------------------------    
    #   Aerodynamics 2
    # ------------------------------------------------------------------
    title = "Aerodynamics 2"
    plt.figure(10)  
    plt.title(title)
    for segment in results.Segments.values():

        plt.subplot(3,1,1)
        plt.plot( segment.t / Units.minute , 
                  segment.CL ,
                  'bo-' )
        plt.xlabel('Time (min)')
        plt.ylabel('CL')
        plt.grid(True)
        
        plt.subplot(3,1,2)
        plt.plot( segment.t / Units.minute , 
                  segment.CD ,
                  'bo-' )
        plt.xlabel('Time (min)')
        plt.ylabel('CD')
        plt.grid(True)
        
        plt.subplot(3,1,3)
        plt.plot( segment.t / Units.minute , 
                  segment.D ,
                  'bo-' )
        plt.plot( segment.t / Units.minute , 
                  segment.F ,
                  'ro-' )
        plt.xlabel('Time (min)')
        plt.ylabel('Drag and Thrust (N)')
        plt.grid(True)
   
    
    # ------------------------------------------------------------------    
    #   Angle of Attack
    # ------------------------------------------------------------------
    title = "L/D vs. alpha"
    plt.figure(title)
    for i in range(len(results.Segments)):
        plt.plot(np.degrees(results.Segments[i].alpha), results.Segments[i].CL/results.Segments[i].CD,'bo-')
    plt.xlabel('Angle of Attack (deg)'); plt.ylabel('L/D'); plt.title(title)
    plt.grid(True)
    
    plt.show()
    raw_input('Press Enter To Quit')

   
    return     
   
  
  
if __name__ == '__main__':
    main()
