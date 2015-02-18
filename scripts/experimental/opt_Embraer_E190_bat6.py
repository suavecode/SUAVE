#by M. Vegh
#note; this script uses the old SUAVE mission at the moment
import SUAVE
import numpy as np
import scipy as sp
import pylab as plt
import copy
import time
import matplotlib
from SUAVE.Methods.Performance import estimate_take_off_field_length
from SUAVE.Methods.Performance import estimate_landing_field_length 
matplotlib.interactive(True)


from pint import UnitRegistry
from SUAVE.Attributes import Units as Units
# ----------------------------------------------------------------------
#   Inputs
# ----------------------------------------------------------------------

def main():
    #use global variables for guess to make optimization faster
    
    global m_guess
    global Ereq_guess
    global Preq_guess
    global iteration_number
    global disp_results #set to 0 unless you want to display plots
    #use these global variables so both optimizer aircraft have access to constraints
    
    global target_range #target overall range
    
    global wclimb1 #vertical velocity for climb segment 1
    global wclimb2
    global wclimb3
    global wclimb4
    global wclimb5
    global wdesc1 #vertical velocity for descent segment 1
    global wdesc2
    global wdesc3
    
    global Vdesc1 #velocity magnitude of first descent segment
    global Vdesc2
    global Vdesc3
    #initial guess  
    m_guess=  64204.6490117
    Ereq_guess= 117167406053.0
    Preq_guess= 8007935.5158
    disp_results=0                            #1 for displaying results, 0 for optimize    
    target_range=2800
    
    wclimb1=3000.*(Units.ft/Units.minute)
    wclimb2=2500.*(Units.ft/Units.minute)
    wclimb3=1800.*(Units.ft/Units.minute)
    wclimb4=900.*(Units.ft/Units.minute)
    wclimb5=300.*(Units.ft/Units.minute)
    
    wdesc1=2600.*(Units.ft/Units.minute)
    wdesc2=2300.*(Units.ft/Units.minute)
    wdesc3=1500.*(Units.ft/Units.minute)
    
    Vdesc1=230.
    Vdesc2=200.
    Vdesc3=140.0 
    
    iteration_number=1
    
    
    #Preq=  1.00318268e+07                     #estimated power requirements for cruise
    Ereq_lis=  2.7621607e+09                 #remaining power requirement for lithium sulfur battery
    Preq_lis= 3.98197096e+06
    
    
    #############
    #input guesses and indices
    i=0
    P_mot        =2E7;   i_P_mot=copy.copy(i);          i+=1
 
    climb_alt_1  =.01;   i_climb_alt_1=copy.copy(i);    i+=1
    climb_alt_2  =.1;    i_climb_alt_2=copy.copy(i);    i+=1
    climb_alt_3  =1;     i_climb_alt_3=copy.copy(i);    i+=1
    climb_alt_4  =2;     i_climb_alt_4=copy.copy(i);    i+=1
    climb_alt_5  =3;     i_climb_alt_5=copy.copy(i);    i+=1
    alpha_rc     =-1.2;  i_alpha_rc=copy.copy(i);       i+=1
    alpha_tc     =-1.3;  i_alpha_tc=copy.copy(i);       i+=1
    wing_sweep   =0.1;   i_wing_sweep=copy.copy(i);     i+=1
    vehicle_S    =45;    i_vehicle_S=copy.copy(i);      i+=1
    Vclimb_1     =120.;  i_Vclimb_1=copy.copy(i);       i+=1
    Vclimb_2     =130;   i_Vclimb_2=copy.copy(i);       i+=1
    Vclimb_3     =200;   i_Vclimb_3=copy.copy(i);       i+=1
    Vclimb_4     =210;   i_Vclimb_4=copy.copy(i);       i+=1
    Vclimb_5     =230;   i_Vclimb_5=copy.copy(i);       i+=1
    desc_alt_1   =2.;    i_desc_alt_1=copy.copy(i);     i+=1
    desc_alt_2   =1;     i_desc_alt_2=copy.copy(i);     i+=1
    cruise_range=2900;   i_cruise_range=copy.copy(i);   i+=1 #cruise range in km
 

    #esp=1500 W-h/kg, range =3800 km
    #inputs= [  2.08480239e+11 ,  1.56398971e+02  , 2.26671872e+02 ,  2.13920953e+00,  5.26093979e+00 ,  5.70421849e+00 ,  5.76221019e+00 ,  5.76221019e+00,  9.08494888e-01 , -4.97338819e+00 ,  2.03386957e-04 ,  1.56432236e+02,   1.31848163e+02 ,  1.57824691e+02 ,  1.74230984e+02 ,  2.00350719e+02,   2.62705750e+02 ,  2.30000000e+02 ,  3.58529306e+03]
   
    #esp=4000 W-h/kg, range=3800 km
    #inputs= [  2.38659244e+11 ,  2.30908552e+04  , 5.14354632e+03  , 2.39493637e+00,  6.46783420e+00,   6.46783421e+00,   6.50325679e+00,   6.50325679e+00,  5.00000000e+00,  -4.99986122e+00 , -2.42616090e-07,   1.03206563e+02, 1.35560173e+02,   1.67515945e+02 ,  1.82203092e+02,   1.70670293e+02,   2.44657141e+02,   2.30000000e+02 ,  3.55771157e+03]

    
    #esp=2000 W-h/kg, range=2400 km
    #inputs=   [  2.53815497e+00 ,  4.61192540e+00 ,  5.81856327e+00,   5.88529881e+00,   5.95456804e+00 ,  1.76614416e+00,  -4.91528402e+00  , 1.31966981e-04,  7.96927333e+01 ,  1.20086331e+02 ,  1.75800907e+02  , 1.73174135e+02,   1.77817946e+02 ,  2.36432755e+02  , 5.94550201e+00 ,  2.42198671e+00,  2.13912785e+03]
    
    #esp=2000 W-h/kg, range=2800 km
  
    inputs=[ 0.590851962935 , 0.497964229001 , 1.24154654143 , 1.24634155623 , 1.48332185353 , 8.63850066855 , -0.286565986413 , -0.678792281953 , 0.00859176374449 , 1.001435128 , 1.56347255279 , 1.96284439037 , 1.98267863948 , 2.00878705494 , 1.85792015629 , 3.52601815815 , 1.2070791392 , 1.66333974717 ]
    #print mybounds
    #print inputs
    out=sp.optimize.fmin(run, inputs)
    #out=run(inputs)
    


# ----------------------------------------------------------------------
#   Calls
# ----------------------------------------------------------------------
def run(inputs):                #sizing loop to enable optimization
    time1=time.time()
    global target_range           #minimum flight range of the aircraft (constraint)
    i=0 #counter for inputs
    #uncomment these global variables unless doing monte carlo optimization
    
    global m_guess
    global Ereq_guess
    global Preq_guess
    global iteration_number
    global disp_results
    
    #mass=[ 100034.162173]
    #mass=[ 113343.414181]                 
    if np.isnan(m_guess) or m_guess>1E7:   #handle nan's so it doesn't result in weird initial guess
        m_guess=10000.

    if np.isnan(Ereq_guess) or Ereq_guess>1E13:
        Ereq_guess=1E10
        print 'Ereq nan'
    if np.isnan(Preq_guess) or Preq_guess>1E10:
        Preq_guess=1E6
        print 'Preq nan'
    print 'mguess=', m_guess
    mass=[ m_guess ]       #mass guess, 2000 W-h/kg
    Ereq=[Ereq_guess]
    #Preq=[Preq_guess]
    P_mot      =inputs[i];       i+=1
    climb_alt_1=inputs[i];       i+=1
    climb_alt_2=inputs[i];       i+=1
    climb_alt_3=inputs[i];       i+=1
    climb_alt_4=inputs[i];       i+=1
    climb_alt_5=inputs[i];       i+=1
    alpha_rc   =inputs[i];       i+=1
    alpha_tc   =inputs[i];       i+=1
    wing_sweep =inputs[i];       i+=1
    vehicle_S  =inputs[i]*100;   i+=1
    Vclimb_1   =inputs[i]*100;   i+=1
    Vclimb_2   =inputs[i]*100;   i+=1
    Vclimb_3   =inputs[i]*100;   i+=1
    Vclimb_4   =inputs[i]*100;   i+=1
    Vclimb_5   =inputs[i]*100;   i+=1
    desc_alt_1 =inputs[i];       i+=1
    desc_alt_2 =inputs[i];       i+=1
    cruise_range=inputs[i]*1000; i+=1

    Preq=P_mot*10**7   #set power as an input
    Preq_guess=Preq
    
    #V_cruise =inputs[i];         i+=1
    V_cruise=230.
    
    
    tol=.01 #percentage difference in mass and energy between iterations
    dm=10000. #initialize error
    dE=10000.
    if disp_results==0:
        max_iter=20
    else:
        max_iter=50
    j=0
     
   
    while abs(dm)>tol or abs(dE)>tol:      #size the vehicle
        m_guess=mass[j]
        Ereq_guess=Ereq[j]
        #Preq_guess=Preq[j]
        vehicle = define_vehicle(m_guess,Ereq_guess, Preq, climb_alt_5,wing_sweep, alpha_rc, alpha_tc, vehicle_S, V_cruise)
        mission = define_mission(vehicle,climb_alt_1,climb_alt_2,climb_alt_3, climb_alt_4,climb_alt_5, desc_alt_1, desc_alt_2, Vclimb_1, Vclimb_2, Vclimb_3, Vclimb_4,Vclimb_5,V_cruise , cruise_range)
        #initialize battery in mission
        mission.segments[0].battery_energy=vehicle.propulsion_model.battery.max_energy
        results = evaluate_mission(vehicle,mission)
        results = evaluate_field_length(vehicle,mission,results) #now evaluate field length
        mass.append(results.segments[-1].conditions.weights.total_mass[-1,0] )
        Ereq.append(results.e_total)
     
        #Preq.append(results.Pmax)
        dm=(mass[j+1]-mass[j])/mass[j]
        dE=(Ereq[j+1]-Ereq[j])/Ereq[j]
        
        
        #display convergence of aircraft
        print 'mass=', mass[j+1]
        print 'dm=', dm
        print 'dE=', dE
        print 'Ereq_guess=', Ereq_guess 
        print 'Preq=', results.Pmax
        j+=1
        if j>max_iter:
            print "maximum number of iterations exceeded"
            break
    #vehicle sized and defined now
 
    #post_process(vehicle,mission,results)
   
    #find minimum energy for each battery

    
    #results.segments[i].Ecurrent_lis=battery_lis.CurrentEnergy
    #penalty_energy=abs(min(min_Ebat/battery.TotalEnergy, min_Ebat_lis/battery_lis.TotalEnergy,0.))*10.**8.
    #penalty_bat_neg=(10.**4.)*abs(min(0.,Ereq_lis, Preq_lis))   #penalty to make sure none of the inputs are negative
    
    
    vehicle.mass_properties.m_full=results.segments[0].conditions.weights.total_mass[-1,0]    #make the vehicle the final mass
    if  disp_results:
        print 'climb_alt1=', climb_alt_1
        print 'Vclimb_1=', Vclimb_1
        print 'climb_alt2=', climb_alt_2
        print 'Vclimb_2=', Vclimb_2
        print 'climb_alt3=', climb_alt_3
        print 'Vclimb_3=', Vclimb_3
        print 'climb_alt4=', climb_alt_4
        print 'Vclimb_4=', Vclimb_4
        print 'climb_alt_5=', climb_alt_5
        print 'Vclimb_5=', Vclimb_5
        print 'desc_alt_1=', desc_alt_1
        print 'desc_alt_2=', desc_alt_2
  
        #print 'V_cruise=' ,V_cruise
        print 'alpha_rc=', alpha_rc
        print 'alpha_tc=', alpha_tc
        print 'wing_sweep=', wing_sweep
        print 'Sref=', vehicle_S
        print 'cruise range=', cruise_range
        print 'total range=', results.segments[-1].conditions.frames.inertial.position_vector[-1,0]/1000
        print 'takeoff mass=', results.segments[0].conditions.weights.total_mass[-1,0] 
        print 'landing mass=', results.segments[-1].conditions.weights.total_mass[-1,0] 
        print 'takeoff field length=', results.field_length.takeoff
        print 'landing field length=', results.field_length.landing
        post_process(vehicle,mission,results)
        
    else: #include penalty functions if you are not displaying the results
        #penalty function if power not high enough
        results=evaluate_penalty(vehicle, results, inputs)
        
    #print vehicle.mass_properties.m_full/(results.segments[-1].vectors.r[-1,0]/1000.),' ', vehicle.mass_properties.m_full, ' ',results.segments[-1].vectors.r[-1,0]/1000. , inputs
    print 'landing mass=', results.segments[-1].conditions.weights.total_mass[-1,0], ' total range=',results.segments[-1].conditions.frames.inertial.position_vector[-1,0] /1000
    
    print 'Ereq=',Ereq_guess, 'Preq=', results.Pmax
    print 'takeoff field length=', results.field_length.takeoff,'landing field length=', results.field_length.landing
    time2=time.time()
    iteration_number+=1
    print 't=', time2-time1, 'seconds'
    print 'iteration number=', iteration_number
    
    #print inputs of each iteration so they can be directly copied
    
    print '[',
    for j in range(len(inputs)):
        print inputs[j],
        if j!=len(inputs)-1:
            print ',',
    print ']'

    #scale vehicle objective function to be of order 1
    return results.segments[-1].conditions.weights.total_mass[-1,0]/(10.**4.)#/(results.segments[-1].vectors.r[-1,0]/1000.) 
    
def evaluate_penalty(vehicle,results, inputs):
    i=0
    #use penalty functions to constrain problem; unpack inputs
    P_mot      =inputs[i];       i+=1
    climb_alt_1=inputs[i];       i+=1
    climb_alt_2=inputs[i];       i+=1
    climb_alt_3=inputs[i];       i+=1
    climb_alt_4=inputs[i];       i+=1
    climb_alt_5=inputs[i];       i+=1
    alpha_rc   =inputs[i];       i+=1
    alpha_tc   =inputs[i];       i+=1
    wing_sweep =inputs[i];       i+=1
    vehicle_S  =inputs[i]*100;   i+=1
    Vclimb_1   =inputs[i]*100;   i+=1
    Vclimb_2   =inputs[i]*100;   i+=1
    Vclimb_3   =inputs[i]*100;   i+=1
    Vclimb_4   =inputs[i]*100;   i+=1
    Vclimb_5   =inputs[i]*100;   i+=1
    desc_alt_1 =inputs[i];       i+=1
    desc_alt_2 =inputs[i];       i+=1
    cruise_range=inputs[i]*1000; i+=1
    V_cruise=230.
 
    results.segments[-1].conditions.weights.total_mass[-1,0]+=abs(min(0, P_mot*10**7-results.Pmax))
    #add penalty function for takeoff and landing field length
    results.segments[-1].conditions.weights.total_mass[-1,0]+=100.*abs(min(0, 1500-results.field_length.takeoff, 1500-results.field_length.landing))
    #add penalty functions for twist, ensuring that trailing edge is >-5 degrees
    results.segments[-1].conditions.weights.total_mass[-1,0]+=100000.*abs(min(0, alpha_tc+5))
    results.segments[-1].conditions.weights.total_mass[-1,0]+=100000.*abs(max(0, alpha_rc-5))
    #now add penalty function if range is not met
    results.segments[-1].conditions.weights.total_mass[-1,0]+=1000.*abs(min(results.segments[-1].conditions.frames.inertial.position_vector[-1,0]/1000-target_range,0,))
    #add penalty function for washin
    results.segments[-1].conditions.weights.total_mass[-1,0]+=10000.*abs(min(0, alpha_rc-alpha_tc))
    
    #make sure that angle of attack is below 30 degrees but above -30 degrees
    max_alpha=np.zeros(len(results.segments))
    min_alpha=np.zeros(len(results.segments))
    for i in range(len(results.segments)):
    
        aoa=results.segments[i].conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
      
    #Smin_Ebat_lis[i]=min(results.segments[i].Ecurrent_lis)
    
    max_alpha[i]=max(aoa)
    min_alpha[i]=min(aoa)
    max_alpha=max(max_alpha)
    min_alpha=min(min_alpha)

    results.segments[-1].conditions.weights.total_mass[-1,0]+=10000.*abs(min(0, 30-max_alpha))+10000.*abs(min(0, 30+min_alpha))
    
    #now add penalty function if wing sweep is too high
    results.segments[-1].conditions.weights.total_mass[-1,0]+=10000.*abs(min(0, 30.-wing_sweep, wing_sweep))
    
    #penalty function in case altitude segments don't match up
    results.segments[-1].conditions.weights.total_mass[-1,0]+=100000*abs(min(climb_alt_5-climb_alt_4, climb_alt_5-climb_alt_3, climb_alt_5-climb_alt_2, climb_alt_5-climb_alt_1,
    climb_alt_4-climb_alt_3, climb_alt_4-climb_alt_2, climb_alt_4-climb_alt_1,
    climb_alt_3-climb_alt_2, climb_alt_2-climb_alt_1, climb_alt_3-climb_alt_1, 0.))
    
    #penalty function in case descent altitude segments don't match up
    results.segments[-1].conditions.weights.total_mass[-1,0]+=100000*abs(min(0., climb_alt_5-desc_alt_1, desc_alt_1-desc_alt_2))
    
    #penalty function to make sure that cruise velocity >=E-190 cruise
    results.segments[-1].conditions.weights.total_mass[-1,0]+=100000*abs(max(0,230.-V_cruise))    
    return results
        
    

# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------

def define_vehicle(Mguess,Ereq, Preq, max_alt,wing_sweep,alpha_rc, alpha_tc, vehicle_S , V_cruise  ):
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Embraer E190 Electric'

    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
     
    vehicle.envelope.ultimate_load = 3.5
    vehicle.envelope.limit_load    = 1.5
    vehicle.num_eng                                 = 2.                        # Number of engines on the aircraft
    vehicle.passengers                              = 110.                      # Number of passengers
    vehicle.wt_cargo                                = 0.  * Units.kilogram  # Mass of cargo
    vehicle.num_seats = 110.                      # Number of seats on aircraft
    vehicle.systems.control                         = "partially powered"       # Specify fully powered, partially powered or anything else is fully aerodynamic
    vehicle.systems.accessories                     = "medium-range"              # Specify what type of aircraft you have
    vehicle.w2h                                     = 16.     * Units.meters    # Length from the mean aerodynamic center of wing to mean aerodynamic center of the horizontal tail
    vehicle.w2v                                     = 20.     * Units.meters    # Length from the mean aerodynamic center of wing to mean aerodynamic center of the vertical tail
    vehicle.reference_area                          = vehicle_S 

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'Main Wing'
   
    wing.areas.reference           = vehicle.reference_area    * Units.meter**2  # Wing gross area in square meters
    wing.aspect_ratio              = 8.3  
    wing.spans.projected           = (wing.aspect_ratio*wing.areas.reference)**.5    * Units.meter     # Span in meters
    wing.taper                     = 0.28                       # Taper ratio
    wing.thickness_to_chord        = 0.105                      # Thickness-to-chord ratio
    wing.sweep                     = wing_sweep     * Units.deg       # sweep angle in degrees
    wing.chords.root               = 5.4     * Units.meter     # Wing exposed root chord length
    wing.chords.mean_aerodynamic   = 12.     * Units.ft    # Length of the mean aerodynamic chord of the wing
    wing.areas_wetted              =wing.areas.reference*2.
    wing.aerodynamic_center        =[wing.chords.mean_aerodynamic/4.,0,0]
    wing.flaps_chord               = 0.28
    wing.areas_exposed             = 0.8*wing.areas.wetted  #
    wing.areas_affected            = 0.6*wing.areas.wetted  #
    wing.eta                       = 1.0                   #
    wing.flap_type   = 'double_slotted'
    wing.twists.root  =alpha_rc*Units.degrees 
    wing.twists.tip  =alpha_tc*Units.degrees 
    wing.vertical    =False
    wing.highlift    = False                 


    # add to vehicle
    vehicle.append_component(wing)
   
    #compute max CL for takeoff now
    
    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    horizontal = SUAVE.Components.Wings.Wing()
    horizontal.tag = 'Horizontal Stabilizer'
    
    horizontal.spans.projected          = 12.08     * Units.meters    # Span of the horizontal tail
    horizontal.sweep                    = wing.sweep      # Sweep of the horizontal tail
    horizontal.chords.mean_aerodynamic  = 2.4      * Units.meters    # Length of the mean aerodynamic chord of the horizontal tail
    horizontal.thickness_to_chord       = 0.11                      # Thickness-to-chord ratio of the horizontal tail
                         # Fraction of horizontal tail area exposed
  
    
   
    horizontal.aspect_ratio         = 5.5         #
    horizontal.symmetric            = True          
    horizontal.thickness_to_chord   = 0.11          #
    horizontal.taper                = 0.11           #
    c_ht                            = 1.   #horizontal tail sizing coefficient
    
    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.horizontal_tail_planform_raymer(horizontal,wing,vehicle.w2h,c_ht )
    
    horizontal.chords.mean_aerodynamic = 8.0
    horizontal. areas.wetted          = horizontal.areas.reference*2.0
    horizontal.areas.exposed          = 0.8*horizontal.areas.wetted  #
    horizontal.areas.affected         = 0.6*horizontal.areas.wetted  #  

    horizontal.eta          = 0.9                   #
                     #
    horizontal.twists.root   =2.0*Units.degrees 
                #
    horizontal.twists.tip   =2.0*Units.degrees 
    #wing.origin             =[ ]
    horizontal.aerodynamic_center= [horizontal.chords.mean_aerodynamic/4.,0,0]
    horizontal.vertical= False
    # add to vehicle
    vehicle.append_component(horizontal)

    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    vertical = SUAVE.Components.Wings.Wing()
    vertical.tag = 'Vertical Stabilizer'    
    
   
    vertical.aspect_ratio       = 1.7          #
    #wing.span      = 100           #
    vertical.sweep              = wing_sweep * Units.deg  #
    vertical.spans.projected     = 5.3     * Units.meters    # Span of the vertical tail
    #vertical.symmetric = False    
    vertical.thickness_to_chord = 0.12          #
    vertical.taper              = 0.10          #
    c_vt                        =.09
    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.vertical_tail_planform_raymer(vertical, wing, vehicle.w2v, c_vt)
    
    vertical.chords.mean_aerodynamic  = 11.0                  
    vertical.areas.wetted   = 2.*vertical.areas.reference
    vertical.areas.exposed  = 0.8*vertical.areas.wetted  #
    vertical.areas.affected = 0.6*vertical.areas.wetted  #  
    #wing.Cl        = 0.002                  #
    vertical.eta           = 0.9                   #
    
    vertical.twists.root   =0.
    vertical.twists.tip    =0.
    
    vertical.aerodynamic_center=[vertical.chords.mean_aerodynamic/4.,0,0]
    vertical.t_tail    = False      # Set to "yes" for a T-tail
    
    vertical.vertical  =True
    # add to vehicle
    vehicle.append_component(vertical)
    
    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    fuselage                           = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag                       = 'Fuselage'
    
    fuselage.number_coach_seats        = 114.  #
    fuselage.seat_pitch                = 0.7455    # m
    fuselage.seats_abreast             = 4    #
    fuselage.fineness.nose             = 1.5  #
    fuselage.fineness.tail             = 1.8  #
  
    fuselage.lengths.fore_space        = 0.
    fuselage.lengths.aft_space         =0.
    fuselage.width                     = 3.0  #
   
    fuselage.heights.maximum           =3.4
    fuselage.heights.at_quarter_length = fuselage.heights.maximum
    fuselage.heights.at_three_quarters_length = fuselage.heights.maximum
    fuselage.heights.at_wing_root_quarter_chord = fuselage.heights.maximum
    #fuselage.area            = 320.      * Units.meter**2  
    
    # size fuselage planform
    SUAVE.Geometry.Two_Dimensional.Planform.fuselage_planform(fuselage)
    
    # add to vehicle
    vehicle.append_component(fuselage)
 
    ################
    
    # ------------------------------------------------------------------
    #  Propulsion
    # ------------------------------------------------------------------
    
    
    atm = SUAVE.Attributes.Atmospheres.Earth.International_Standard()
    p1, T1, rho1, a1, mew1 = atm.compute_values(0.)
    p2, T2, rho2, a2, mew2 = atm.compute_values(max_alt*Units.km)
  
    
    sizing_segment = SUAVE.Components.Propulsors.Segments.Segment()
    sizing_segment.M   = 230./a2        
    sizing_segment.alt = max_alt
    sizing_segment.T   = T2           
    
    sizing_segment.p   = p2     
    #create battery
    battery = SUAVE.Components.Energy.Storages.Battery_Li_Air()
    #battery_lis = SUAVE.Components.Energy.Storages.Battery()
    #battery_lis.type='Li_S'
    #battery_lis.tag='Battery_Li_S'
    battery.tag = 'Battery'
   
    # attributes
 
    #ducted fan
    DuctedFan= SUAVE.Components.Propulsors.Ducted_Fan_Bat()
    #DuctedFan.propellant = SUAVE.Attributes.Propellants.Aviation_Gasoline()
    DuctedFan.diffuser_pressure_ratio = 0.98
    DuctedFan.fan_pressure_ratio = 1.65
    DuctedFan.fan_nozzle_pressure_ratio = 0.99
    DuctedFan.design_thrust = 2.5*Preq/V_cruise #factor of 2.5 accounts for top of climb
    DuctedFan.number_of_engines=2.0   
    DuctedFan.eta_pe=.95         #electric efficiency of battery
    DuctedFan.engine_sizing_ductedfan(sizing_segment)   #calling the engine sizing method 

    battery.initialize(Ereq, Preq)
    m_air=battery.find_mass_gain()    #find mass gain of battery throughout mission and size vehicle
    
    #now add the electric motor weight
    motor_mass=DuctedFan.number_of_engines*SUAVE.Methods.Weights.Correlations.Propulsion.air_cooled_motor((Preq)*Units.watts/DuctedFan.number_of_engines)
    propulsion_mass=SUAVE.Methods.Weights.Correlations.Propulsion.integrated_propulsion(motor_mass/DuctedFan.number_of_engines,DuctedFan.number_of_engines)
    DuctedFan.mass_properties.mass=propulsion_mass

    # ------------------------------------------------------------------
    #   Simple Aerodynamics Model
    # ------------------------------------------------------------------ 
    
    aerodynamics = SUAVE.Attributes.Aerodynamics.Fidelity_Zero()
    aerodynamics.initialize(vehicle)
    vehicle.aerodynamics_model = aerodynamics
    
    #define the energy network
    net=SUAVE.Components.Energy.Networks.Basic_Battery_Network()
    net.battery=battery
    net.propulsor=DuctedFan
    net.number_of_engines=DuctedFan.number_of_engines
    vehicle.propulsors.propulsor=net #Emilio did this; not sure if necessary
    
    vehicle.propulsion_model=net
    
    # ------------------------------------------------------------------
    #   Define New Gross Takeoff Weight
    # ------------------------------------------------------------------
    #now add component weights to the gross takeoff weight of the vehicle
    m_fuel=0.
    vehicle.mass_properties.max_takeoff             = Mguess 
    vehicle.mass_properties.takeoff                 = Mguess-m_air
    vehicle.mass_properties.max_zero_fuel           = Mguess
    weight =SUAVE.Methods.Weights.Correlations.Tube_Wing.empty_custom_eng(vehicle, DuctedFan)
    m_full=weight.empty+battery.mass_properties.mass+weight.payload
    m_end=m_full+m_air
    vehicle.mass_properties.m_full=m_full
    vehicle.mass_properties.takeoff=m_full

    
    '''
    print 'motor              =', motor_mass
    print 'ducted fan         =', DuctedFan.mass_properties.mass-motor_mass
    print 'battery            =', battery.mass_properties.mass
   
    print 'propulsion_overall =', weight.propulsion
    print 'passenger          =', weight.payload
    print 'air                =', m_air
    print 'wing               =', weight.wing
    print 'vtail              =', weight.vertical_tail+weight.rudder     
    print 'htail              =', weight.horizontal_tail
    print 'fuselage           =', weight.fuselage[0]
    print 'landing gear       =', weight.landing_gear
    print 'systems            =', weight.systems
    print 'furnishing         =', weight.wt_furnish
    print 'takeoff mass       =', vehicle.mass_properties.m_full[0]
    print 'landing mass       =', vehicle.mass_properties.m_full[0]+m_air
    print 'empty mass         =', weight.empty[0]
    '''
    
    #print weight
    
    ##############################################################################
    # ------------------------------------------------------------------
    #   Define Configurations
    # ------------------------------------------------------------------

    # --- Takeoff Configuration ---
    config = vehicle.new_configuration("takeoff")
    # this configuration is derived from the baseline vehicle

    # --- Cruise Configuration ---
    config = vehicle.new_configuration("cruise")
    # this configuration is derived from vehicle.configs.takeoff

    # --- Takeoff Configuration ---
    takeoff_config = vehicle.configs.takeoff
    takeoff_config.wings['Main Wing'].flaps_angle =  15. * Units.deg
    takeoff_config.wings['Main Wing'].slats_angle  = 25. * Units.deg


    # --- Landing Configuration ---
    landing_config = vehicle.new_configuration("landing")
    landing_config.wings['Main Wing'].flaps_angle =  20. * Units.deg
    landing_config.wings['Main Wing'].slats_angle  = 25. * Units.deg
    landing_config.mass_properties.landing = m_end


    
    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------
    
    return vehicle

###############################################################################################################################################
# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------
def define_mission(vehicle,climb_alt_1,climb_alt_2,climb_alt_3, climb_alt_4, climb_alt_5, desc_alt_1, desc_alt_2,Vclimb_1, Vclimb_2, Vclimb_3, Vclimb_4,Vclimb_5,  V_cruise , cruise_range):
   
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
   
    
    mission = SUAVE.Attributes.Missions.Mission()
    mission.tag = 'The Test Mission'

    # initial mass
    mission.m0 = vehicle.mass_properties.m_full
    
    # atmospheric model
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet = SUAVE.Attributes.Planets.Earth()
    
    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    
    mission.airport = airport
    
    
    # ------------------------------------------------------------------
    #   First Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------
    
    segment =  SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb - 1"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.takeoff
    
    # define segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet    
    segment.altitude_start=0.0
    segment.altitude_end  =climb_alt_1* Units.km
    # pick two:
    segment.air_speed    = Vclimb_1        # m/s

    global wclimb1
    segment.rate       = wclimb1

    
    # add to mission
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------
    #   Second Climb Segment: constant Speed, constant segment angle 
    # ------------------------------------------------------------------    
   
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb - 2"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet    
    segment.altitude_end=climb_alt_2* Units.km
    
    # pick two:

    segment.air_speed       =Vclimb_2
    global wclimb2
    segment.rate        =wclimb2
    
    # add to mission
    mission.append_segment(segment)

    
    # ------------------------------------------------------------------
    #   Third Climb Segment: constant velocity, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb - 3"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet   
   
    segment.altitude_end=climb_alt_3* Units.km
    # pick two:
    segment.air_speed        = Vclimb_3
    global wclimb3
    segment.rate       =wclimb3
    
    # add to mission
    mission.append_segment(segment)
    
     # ------------------------------------------------------------------
    #   Fourth Climb Segment: constant velocity, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb - 4"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet   
   
    segment.altitude_end=climb_alt_4* Units.km
    # pick two:
    segment.air_speed        = Vclimb_4
    global wclimb4
    segment.rate       =wclimb4
    
    # add to mission
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------
    #   Fifth Climb Segment: Constant Speed, Constant Climb Rate
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb - 5"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet        

    segment.altitude_end=climb_alt_5* Units.km
    
    # pick two:
    segment.air_speed      = Vclimb_5
    global wclimb5
    segment.rate       =wclimb5
    # add to mission
    mission.append_segment(segment)    
    
    
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Cruise.Constant_Speed_Constant_Altitude()
    segment.tag = "Cruise"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet     

    segment.air_speed       = V_cruise    # m/s
    segment.distance   =cruise_range*Units.km
    segment.numerics.n_control_points =8
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   First Descent Segment: constant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "Descent - 1"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet    
    segment.altitude_end   =  desc_alt_1  * Units.km
    global Vdesc1
    segment.air_speed       = Vdesc1          # m/s
    global wdesc1
    segment.rate         =wdesc1
    
    # add to mission
    mission.append_segment(segment)
    

    # ------------------------------------------------------------------    
    #   Second Descent Segment: constant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "Descent - 2"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise

    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet   
    segment.altitude_end   =  desc_alt_2 * Units.km # km
    #segment.altitude   = [5., 0.0]
    global Vdesc2
    segment.air_speed      = Vdesc2      # m/s
    #segment.rate       = 5.0         # m/s
    global wdesc2
    segment.rate         =wdesc2
    # append to mission
    mission.append_segment(segment)

       # ------------------------------------------------------------------    
    #   Third Descent Segment: constant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "Descent -3"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet   
    segment.altitude_end   =  0.  # km
    global Vdesc3
    segment.air_speed       = Vdesc3      # m/s
    #segment.rate       = 5.0         # m/s
    global wdesc3
    segment.rate         =wdesc3
    # append to mission
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   Mission definition complete   
    
    #vehicle.mass_properties.m_empty+=motor_mass

    #print vehicle.mass_properties.m_full
    
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
    #determine energy characteristiscs
    e_current_min=1E14
    Pmax=0.
    for i in range(len(results.segments)):
            if np.min(results.segments[i].conditions.propulsion.battery_energy[:,0])<e_current_min:
                e_current_min=np.min(results.segments[i].conditions.propulsion.battery_energy[:,0])
            if np.max(np.abs(results.segments[i].conditions.propulsion.battery_draw[:,0]))>Pmax:
                Pmax=np.max(np.abs(results.segments[i].conditions.propulsion.battery_draw[:,0]))
                
    results.e_total=results.segments[0].conditions.propulsion.battery_energy[0,0]-e_current_min
    results.Pmax=Pmax
    #print 'battery_energy=', results.segments[0].conditions.propulsion.battery_energy
    print 'e_current_min=',e_current_min
    print "e_total=", results.e_total
    print "Pmax=", Pmax
    print "e_current_min=", e_current_min
    '''
    #add lithium air battery mass gain to weight of the vehicle
   
    #vehicle.mass_properties.m_full+=m_li_air
    '''
    return results
#########################################################################################################################
def evaluate_field_length(vehicle,mission,results):
    
    # unpack
    airport = mission.airport
    
    takeoff_config = vehicle.configs.takeoff
    landing_config = vehicle.configs.landing
    
     
    
    # evaluate
    TOFL = estimate_take_off_field_length(vehicle,takeoff_config,airport)
    LFL = estimate_landing_field_length(vehicle,landing_config,airport)
    
    # pack
    field_length = SUAVE.Structure.Data()
    field_length.takeoff = TOFL[0]
    field_length.landing = LFL[0]
    
    results.field_length = field_length
 
    
    return results
#########################################################################################################################
# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def post_process(vehicle,mission,results):
    #battery = vehicle.energy.Storages['Battery']
    #battery_lis = vehicle.Energy.Storages['Battery_Li_S']
    # ------------------------------------------------------------------    
    #   Thrust Angle
    # ------------------------------------------------------------------
    '''
    title = "Thrust Angle History"
    plt.figure(0)
    for i in range(len(results.segments)):
        plt.plot(results.segments[i].t/60,np.degrees(results.segments[i].gamma),'bo-')
    plt.xlabel('Time (mins)'); plt.ylabel('Thrust Angle (deg)'); plt.title(title)
    plt.grid(True)
    '''
    # ------------------------------------------------------------------    
    #   Throttle
    # ------------------------------------------------------------------
    plt.figure("Throttle History")
    axes = plt.gca()
    for i in range(len(results.segments)):
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        eta  = results.segments[i].conditions.propulsion.throttle[:,0]
        axes.plot(time, eta, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Throttle')
    axes.grid(True)

    # ------------------------------------------------------------------    
    #   Angle of Attack
    # ------------------------------------------------------------------
    plt.figure("Angle of Attack History")
    axes = plt.gca()    
    for i in range(len(results.segments)):     
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        aoa = results.segments[i].conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        axes.plot(time, aoa, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Angle of Attack (deg)')
    axes.grid(True)        
    
    # ------------------------------------------------------------------    
    #   Vehicle Mass
    # ------------------------------------------------------------------
    plt.figure("Vehicle Mass")
    axes = plt.gca()
    for i in range(len(results.segments)):
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        mass = results.segments[i].conditions.weights.total_mass[:,0]
        axes.plot(time, mass, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Vehicle Mass (kg)')
    axes.grid(True)
    
    
    
    # ------------------------------------------------------------------    
    #   Mass Gain Rate
    # ------------------------------------------------------------------
    plt.figure("Mass Accumulation Rate")
    axes = plt.gca()    
    for i in range(len(results.segments)):     
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        mdot = results.segments[i].conditions.propulsion.fuel_mass_rate[:,0]
        axes.plot(time, -mdot, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Mass Accumulation Rate(kg/s)')
    axes.grid(True)    
    
    
 

    
    
    # ------------------------------------------------------------------    
    #   Altitude
    # ------------------------------------------------------------------
    plt.figure("Altitude")
    axes = plt.gca()    
    for i in range(len(results.segments)):     
        time     = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        altitude = results.segments[i].conditions.freestream.altitude[:,0] /Units.km
        axes.plot(time, altitude, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Altitude (km)')
    axes.grid(True)
    
    
    # ------------------------------------------------------------------    
    #   Horizontal Distance
    # ------------------------------------------------------------------
    '''
    title = "Ground Distance"
    plt.figure(title)
    for i in range(len(results.segments)):
        plt.plot(results.segments[i].t/60,results.segments[i].conditions.frames.inertial.position_vector[:,0])
    plt.xlabel('Time (mins)'); plt.ylabel('ground distance(km)'); plt.title(title)
    plt.grid(True)
    '''
    
    # ------------------------------------------------------------------    
    #   Energy
    # ------------------------------------------------------------------
    
    title = "Energy and Power"
    fig=plt.figure(title)
    #print results.segments[0].Ecurrent[0]
    for segment in results.segments:
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        axes = fig.add_subplot(2,1,1)
        axes.plot(time, segment.conditions.propulsion.battery_energy/vehicle.propulsion_model.battery.max_energy,'bo-')
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('State of Charge of the Battery')
        axes.grid(True)
        axes = fig.add_subplot(2,1,2)
        axes.plot(time, -segment.conditions.propulsion.battery_draw,'bo-')
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Electric Power (Watts)')
        axes.grid(True)
    """
    for i in range(len(results.segments)):
        plt.plot(results.segments[i].t/60, results.segments[i].Ecurrent_lis/battery_lis.TotalEnergy,'ko-')
      
        #results.segments[i].Ecurrent_lis/battery_lis.TotalEnergy
    """
    # ------------------------------------------------------------------    
    #   Aerodynamics
    # ------------------------------------------------------------------
    fig = plt.figure("Aerodynamic Forces")
    for segment in results.segments.values():
        
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
    for segment in results.segments.values():
        
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        CLift  = segment.conditions.aerodynamics.lift_coefficient[:,0]
        CDrag  = segment.conditions.aerodynamics.drag_coefficient[:,0]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]

        axes = fig.add_subplot(4,1,1)
        axes.plot( time , CLift , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('CL')
        axes.grid(True)
        
        axes = fig.add_subplot(4,1,2)
        axes.plot( time , CDrag , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('CD')
        axes.grid(True)
        
        axes = fig.add_subplot(4,1,3)
        axes.plot( time , Drag   , 'bo-' )
        axes.plot( time , Thrust , 'ro-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Drag and Thrust (N)')
        axes.grid(True)
        
        axes = fig.add_subplot(4,1,4)
        axes.plot( time , CLift/CDrag , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('CL/CD')
        axes.grid(True)
    plt.show()
    raw_input('Press Enter To Quit')
    return     
   
  
  
if __name__ == '__main__':
    main()
