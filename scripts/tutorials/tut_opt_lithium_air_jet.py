#by M. Vegh
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
from SUAVE.Core import Units as Units
# ----------------------------------------------------------------------
#   Inputs
# ----------------------------------------------------------------------

def main():
    global iteration_number #use global variable to keep track of how long optimization has gone
    global disp_results
    disp_results=0
    #############
    '''
    Input Variable Order
    P_mot        =2E7 /(10.**7.);  
    climb_alt_1  =.01;  
    climb_alt_2  =.1;   
    climb_alt_3  =1;    
    climb_alt_4  =2;    
    climb_alt_5  =3;    
    alpha_rc     =-1.2; 
    alpha_tc     =-1.3; 
    wing_sweep   =0.1;  
    vehicle_S    =45*1./100.;   
    Vclimb_1     =120.*1./100.; 
    Vclimb_2     =130.*1./100.;  
    Vclimb_3     =200.*1./100.;  
    Vclimb_4     =210.*1./100.;  
    Vclimb_5     =230.*1./100.;  
    desc_alt_1   =2.;   
    desc_alt_2   =1;    
    cruise_range=2900.*1./1000;  
    '''
    #range =2800 km
    #inputs=[ 1.01420090388 , 0.00211788693039 , 1.23478937042 , 1.41300163708 , 1.70807214708 , 7.1561218447 , -0.375058275159 , -1.46732112946 , 0.0119959909091 , 1.14938713948 , 1.29630577308 , 0.584463567988 , 1.65584269711 , 1.64579846566 , 1.55989976031 , 4.48764563837 , 3.39193333997 , 1.74925037953 ]
    #range=4800 km
    
    lower_bounds=[1,  .01,.01,0.1,0.1,  0.1, -5., -5.,0.01,  50./100.,  50./100., 50./100.,  50./100.,  50./100.,  50./100.,   .1,   .1, 1800./1000.]
    upper_bounds=[1E3, 8.,10, 10., 10., 12., 5. , 5., 25. , 250./100., 230./100, 230./100., 230./100., 230./100., 230./100. , 11.,  11., 3800./1000.]
    mybounds=[]
    for i in range(len(lower_bounds)):
        mybounds.append((lower_bounds[i], upper_bounds[i]))
    
    #inputs=[ 1.88514586133 , 0.349266307744 , 1.14260896528 , 1.21671932472 , 1.79760174766, 6.35335448285 , -0.309886795812 , -0.31341919074 , 0.00166861827626 , 0.983447388245 , 1.12345545034 , 1.65177066127 , 2.00829558073 , 1.96584942855 , 1.74983511068 , 6.25967162737 , 2.12705627453 , 4.13378138263 ]
    inputs=[ 1.8703475788339 , 0.353159963696 , 1.15703905021 , 1.23611214124 , 1.8070948473, 6.25523915312 , -0.311625603746 , -0.315170759822 , 0.00168449627666 , 0.999984394693 , 1.11932671775 , 1.63716789738 , 1.93253518819 , 1.94175566908 , 1.76039206949 , 6.26088884866 , 2.13178114629 , 4.14627176885 ]
    
    #mass= 94770.6517907 kg
    
    
    #inputs=[ 1.74409397687 , 3.91161546688 , 4.47317473095 , 5.04623400264 , 8.22123179161, 9.52266939113 , 0.807812232455 , 0.00678379350742 , 30.0404283705 , 1.21228197462 , 1.04875982776 , 1.64983694073 , 2.02558103462 , 1.81874623408 , 2.2435440488 , 9.47766173206 , 1.62065835069 , 4.28748780567 ]
        
    #mass=88936.4594853 kg
    
    iteration_number=1
    #out=sp.optimize.differential_evolution(run, mybounds, disp=1)
   
    if disp_results==0:  #run optimization
        out=sp.optimize.fmin(run, inputs, disp=1)
    else:
        run(inputs)    #display and plot the results
    #mass=84489.0450279
    #out=run(inputs)
    


# ----------------------------------------------------------------------
#   Calls
# ----------------------------------------------------------------------
    
def run(inputs):                #sizing loop to enable optimization

    global disp_results                #1 for displaying results, 0 for optimize    
    global iteration_number
    print 'iteration number=', iteration_number
       #print inputs of each iteration so they can be directly copied
    print '[',
    for j in range(len(inputs)):
        print inputs[j],
        if j!=len(inputs)-1:
            print ',',
    print ']'

    iteration_number+=1
    time1=time.time()
        
    i=0 #counter for inputs

    #optimization parameters; begin from same initial guess
    m_guess    = 64204.6490117
    Ereq_guess = 117167406053.0
    Preq_guess = 8007935.5158
    target_range=4800                       #minimum flight range of the aircraft (constraint)
      
    Ereq=[Ereq_guess]
    mass=[ m_guess ]   
    '''
    if np.isnan(m_guess) or m_guess>1E7:   #handle nan's so it doesn't result in weird initial guess
        m_guess=10000.

    if np.isnan(Ereq_guess) or Ereq_guess>1E13:
        Ereq_guess=1E10
        print 'Ereq nan'
    if np.isnan(Preq_guess) or Preq_guess>1E10:
        Preq_guess=1E6
        print 'Preq nan'
    '''
    print 'mguess=', m_guess

    
    tol=.05 #percentage difference in mass and energy between iterations
    dm=10000. #initialize error
    dE=10000.
    if disp_results==0:
        max_iter=5
    else:
        max_iter=5
    j=0
    P_mot=inputs[0]*10.**7.
    Preq=P_mot 
   
    while abs(dm)>tol or abs(dE)>tol:      #size the vehicle
        m_guess=mass[j]
        Ereq_guess=Ereq[j]
        configs, analyses = full_setup(inputs,m_guess, Ereq_guess)
        simple_sizing(configs,analyses, m_guess,Ereq_guess,Preq)
        mission = analyses.missions.base
        battery=configs.base.energy_network['battery']
        battery.current_energy=battery.max_energy
        configs.finalize()
        analyses.finalize()
        configs.cruise.energy_network['battery']=battery #make it so all configs handle the exact same battery object
        configs.takeoff.energy_network['battery']=battery
        configs.landing.energy_network['battery']=battery
        #initialize battery in mission
        mission.segments[0].battery_energy=battery.max_energy
        results = evaluate_mission(configs,mission)
       
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
     #vehicle sized:now find field length
    results = evaluate_field_length(configs,analyses, mission,results) #now evaluate field length
 
    if  disp_results:
        #unpack inputs
        i=0
        P_mot      =inputs[i]*10**7;       i+=1
        climb_alt_1=inputs[i];       i+=1
        climb_alt_2=inputs[i];       i+=1
        climb_alt_3=inputs[i];       i+=1
        climb_alt_4=inputs[i];       i+=1
        climb_alt_5=inputs[i];       i+=1
        alpha_rc   =inputs[i];       i+=1
        alpha_tc   =inputs[i];       i+=1
        wing_sweep =inputs[i];       i+=1; 
        vehicle_S  =inputs[i]*100;   i+=1
        Vclimb_1   =inputs[i]*100;   i+=1
        Vclimb_2   =inputs[i]*100;   i+=1
        Vclimb_3   =inputs[i]*100;   i+=1
        Vclimb_4   =inputs[i]*100;   i+=1
        Vclimb_5   =inputs[i]*100;   i+=1
        desc_alt_1 =inputs[i];       i+=1
        desc_alt_2 =inputs[i];       i+=1
        cruise_range=inputs[i]*1000; i+=1
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
        post_process(mission, configs,results)
        return
    else: #include penalty functions if you are not displaying the results
        #penalty function if power not high enough
        vehicle=configs.base
        results=evaluate_penalty(vehicle, results, inputs,target_range)
        
    #print vehicle.mass_properties.m_full/(results.segments[-1].vectors.r[-1,0]/1000.),' ', vehicle.mass_properties.m_full, ' ',results.segments[-1].vectors.r[-1,0]/1000. , inputs
    print 'landing mass=', results.segments[-1].conditions.weights.total_mass[-1,0], ' total range=',results.segments[-1].conditions.frames.inertial.position_vector[-1,0] /1000
    
    print 'Ereq=',Ereq_guess, 'Preq=', results.Pmax
    print 'takeoff field length=', results.field_length.takeoff,'landing field length=', results.field_length.landing
    time2=time.time()       #time between iterations
    print 't=', time2-time1, 'seconds'
 
 
    #scale vehicle objective function to be of order 1
    return results.segments[-1].conditions.weights.total_mass[-1,0]/(10.**4.)
    
def evaluate_penalty(vehicle,results, inputs,target_range):
    i=0
    #use penalty functions to constrain problem; unpack inputs, scaling them propery
    P_mot      =inputs[i]*10.**7.;   i+=1
    climb_alt_1=inputs[i]      ;     i+=1
    climb_alt_2=inputs[i]      ;     i+=1
    climb_alt_3=inputs[i]      ;     i+=1
    climb_alt_4=inputs[i]      ;     i+=1
    climb_alt_5=inputs[i]      ;     i+=1
    alpha_rc   =inputs[i]      ;     i+=1
    alpha_tc   =inputs[i]      ;     i+=1
    wing_sweep =inputs[i]      ;     i+=1
    vehicle_S  =inputs[i]*100  ;     i+=1
    Vclimb_1   =inputs[i]*100  ;     i+=1
    Vclimb_2   =inputs[i]*100  ;     i+=1
    Vclimb_3   =inputs[i]*100  ;     i+=1
    Vclimb_4   =inputs[i]*100  ;     i+=1
    Vclimb_5   =inputs[i]*100  ;     i+=1
    desc_alt_1 =inputs[i]      ;     i+=1
    desc_alt_2 =inputs[i]      ;     i+=1
    cruise_range=inputs[i]*1000;     i+=1
    V_cruise=230.
    print 'P_mot=', P_mot
    print 'results.Pmax=', results.Pmax
    results.segments[-1].conditions.weights.total_mass[-1,0]+=abs(min(0, P_mot-results.Pmax))
    #add penalty function for takeoff and landing field length
    #results.segments[-1].conditions.weights.total_mass[-1,0]+=100.*abs(min(0, 1500-results.field_length.takeoff, 1500-results.field_length.landing))
    #add penalty functions for twist, ensuring that trailing edge is >-5 degrees
    results.segments[-1].conditions.weights.total_mass[-1,0]+=100000.*abs(min(0, alpha_tc+5))
    results.segments[-1].conditions.weights.total_mass[-1,0]+=100000.*abs(max(0, alpha_rc-5))
    #now add penalty function if range is not met
    results.segments[-1].conditions.weights.total_mass[-1,0]+=1000.*abs(min(results.segments[-1].conditions.frames.inertial.position_vector[-1,0]/1000.-target_range,0,))
    #add penalty function for washin
    results.segments[-1].conditions.weights.total_mass[-1,0]+=10000.*abs(min(0, alpha_rc-alpha_tc))
    
    #make sure that angle of attack is below 15 degrees but above -15 degrees
    max_alpha=np.zeros(len(results.segments))
    min_alpha=np.zeros(len(results.segments))
    for i in range(len(results.segments)):
        aoa=results.segments[i].conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        max_alpha[i]=max(aoa)
        min_alpha[i]=min(aoa)
    max_alpha=max(max_alpha)
    min_alpha=min(min_alpha)
    print 'max_alpha=', max_alpha
    print 'min_alpha=', min_alpha
    results.segments[-1].conditions.weights.total_mass[-1,0]+=10000.*abs(min(0, 15-max_alpha))+10000.*abs(min(0, 15+min_alpha))
    
    #now add penalty function if wing sweep is too high
    results.segments[-1].conditions.weights.total_mass[-1,0]+=10000.*abs(min(0, 30.-wing_sweep, wing_sweep))
    
    #penalty function in case altitude segments don't match up
    results.segments[-1].conditions.weights.total_mass[-1,0]+=100000*abs(min(climb_alt_5-climb_alt_4, climb_alt_5-climb_alt_3, climb_alt_5-climb_alt_2, climb_alt_5-climb_alt_1,
                                                                         climb_alt_4-climb_alt_3, climb_alt_4-climb_alt_2, climb_alt_4-climb_alt_1,
                                                                         climb_alt_3-climb_alt_2, climb_alt_2-climb_alt_1, climb_alt_3-climb_alt_1, 0.))
    
    #penalty function in case descent altitude segments don't match up
    results.segments[-1].conditions.weights.total_mass[-1,0]+=100000*abs(min(0., climb_alt_5-desc_alt_1, desc_alt_1-desc_alt_2))

    return results
        
    

# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup(m_guess,Ereq, Preq, max_alt,wing_sweep,alpha_rc, alpha_tc, vehicle_S , V_cruise  ):
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Embraer E190 Electric'

    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    vehicle.envelope.ultimate_load                  = 3.5
    vehicle.envelope.limit_load                     = 1.5
    vehicle.num_eng                                 = 2.                        # Number of engines on the aircraft
    vehicle.passengers                              = 110.                      # Number of passengers
    vehicle.wt_cargo                                = 0.  * Units.kilogram      # Mass of cargo
    vehicle.num_seats                               = 110.                      # Number of seats on aircraft
    vehicle.systems.control                         = "partially powered"       # Specify fully powered, partially powered or anything else is fully aerodynamic
    vehicle.systems.accessories                     = "medium-range"              # Specify what type of aircraft you have
    vehicle.w2h                                     = 16.     * Units.meters    # Length from the mean aerodynamic center of wing to mean aerodynamic center of the horizontal tail
    vehicle.w2v                                     = 20.     * Units.meters    # Length from the mean aerodynamic center of wing to mean aerodynamic center of the vertical tail
    vehicle.reference_area                          = vehicle_S 

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
   
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
    wing.dynamic_pressure_ratio    = 1.0
    wing.flap_type                 = 'double_slotted'
    wing.twists.root               =alpha_rc*Units.degrees 
    wing.twists.tip                =alpha_tc*Units.degrees 
    wing.vertical                  =False
    wing.high_lift                 = True                 
    wing.span_efficiency           =1.0

    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    horizontal = SUAVE.Components.Wings.Wing()
    horizontal.tag = 'horizontal_stabilizer'
    
    horizontal.spans.projected          = 12.08     * Units.meters    # Span of the horizontal tail
    horizontal.sweep                    = wing.sweep      # Sweep of the horizontal tail
    horizontal.chords.mean_aerodynamic  = 2.4      * Units.meters    # Length of the mean aerodynamic chord of the horizontal tail
    horizontal.thickness_to_chord       = 0.11                      # Thickness-to-chord ratio of the horizontal tail
    horizontal.aspect_ratio             = 5.5        
    horizontal.symmetric                = True       
    horizontal.thickness_to_chord       = 0.11       
    horizontal.taper                    = 0.11       
    horizontal.dynamic_pressure_ratio   = 0.9 
    c_ht                                 = 1.   #horizontal tail sizing coefficient
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.horizontal_tail_planform_raymer(horizontal,wing,vehicle.w2h,c_ht )
    
    # add to vehicle
    vehicle.append_component(horizontal)

    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    vertical = SUAVE.Components.Wings.Wing()
    vertical.tag = 'vertical_stabilizer'    
    
   
    vertical.aspect_ratio       = 1.7          #
    #wing.span      = 100           #
    vertical.sweep              = wing_sweep * Units.deg  #
    vertical.spans.projected     = 5.3     * Units.meters    # Span of the vertical tail
    #vertical.symmetric = False    
    vertical.thickness_to_chord = 0.12          #
    vertical.taper              = 0.10          #
    vertical.twists.root   =0.
    vertical.twists.tip    =0.
    c_vt                        =.09
    vertical.t_tail    = False      # Set to "yes" for a T-tail
    
    vertical.vertical  =True
    vertical.dynamic_pressure_ratio  = 1.0
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.vertical_tail_planform_raymer(vertical, wing, vehicle.w2v, c_vt)
    
   
    # add to vehicle
    vehicle.append_component(vertical)
    
    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    fuselage                           = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag                       = 'fuselage'
    
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

    # add to vehicle
    vehicle.append_component(fuselage)
 
    ################
    
    # ------------------------------------------------------------------
    #  Propulsion
    # ------------------------------------------------------------------
    
    
    atm = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    p1, T1, rho1, a1, mew1 = atm.compute_values(0.)
    p2, T2, rho2, a2, mew2 = atm.compute_values(max_alt*Units.km)
  
    
    sizing_segment = SUAVE.Components.Propulsors.Segments.Segment()
    sizing_segment.M   = 230./a2        
    sizing_segment.alt = max_alt
    sizing_segment.T   = T2           
    
    sizing_segment.p   = p2     
    #create battery
    battery = SUAVE.Components.Energy.Storages.Batteries.Variable_Mass.Lithium_Air()
    #battery.specific_energy=2000.*Units.Wh/Units.kg
    battery.tag = 'battery'
   
    # attributes
 
    ducted_fan= SUAVE.Components.Propulsors.Ducted_Fan()
    ducted_fan.tag                       ='ducted_fan'
    ducted_fan.diffuser_pressure_ratio   = 0.98
    ducted_fan.fan_pressure_ratio        = 1.65
    ducted_fan.fan_nozzle_pressure_ratio = 0.99
    ducted_fan.design_thrust             = Preq*1.5/V_cruise 
    ducted_fan.number_of_engines         =2.0    
    ducted_fan.engine_sizing_ducted_fan(sizing_segment)   #calling the engine sizing method 
    
    # ------------------------------------------------------------------
    #  Energy Network
    # ------------------------------------------------------------------ 
    
    #define the energy network
    net=SUAVE.Components.Energy.Networks.Battery_Ducted_Fan()
    net.propulsor=ducted_fan
    net.nacelle_diameter=ducted_fan.nacelle_diameter
    net.engine_length=ducted_fan.engine_length
    net.tag='network'
    net.append(ducted_fan)
    net.battery=battery
    net.number_of_engines=ducted_fan.number_of_engines
    
    vehicle.energy_network=net
    vehicle.propulsors.append(ducted_fan)

    #vehicle.propulsors.append(turbofan)
    #vehicle.propulsion_model=net
    return vehicle
    
##################################################################
def simple_sizing(configs, analyses, m_guess, Ereq, Preq):
    from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform
    
# ------------------------------------------------------------------
    #   Define New Gross Takeoff Weight
    # ------------------------------------------------------------------
    #now add component weights to the gross takeoff weight of the vehicle
   
    base = configs.base
    base.pull_base()
    base.mass_properties.max_takeoff=m_guess
    base.mass_properties.max_zero_fuel=m_guess  #just used for weight calculation
    mission=analyses.missions.base.segments
    airport=analyses.missions.base.airport
    atmo            = airport.atmosphere
    #determine geometry of fuselage as well as wings
    fuselage=base.fuselages['fuselage']
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.fuselage_planform(fuselage)
    fuselage.areas.side_projected   = fuselage.heights.maximum*fuselage.lengths.cabin*1.1 #  Not correct
    base.wings['main_wing'] = wing_planform(base.wings['main_wing'])
    base.wings['horizontal_stabilizer'] = wing_planform(base.wings['horizontal_stabilizer']) 
    
    base.wings['vertical_stabilizer']   = wing_planform(base.wings['vertical_stabilizer'])
    #calculate position of horizontal stabilizer
    base.wings['horizontal_stabilizer'].aerodynamic_center[0]= base.w2h- \
    (base.wings['horizontal_stabilizer'].origin[0] + \
     base.wings['horizontal_stabilizer'].aerodynamic_center[0] - \
     base.wings['main_wing'].origin[0] - base.wings['main_wing'].aerodynamic_center[0])
    #wing areas
    for wing in base.wings:
        wing.areas.wetted   = 2.00 * wing.areas.reference
        wing.areas.affected = 0.60 * wing.areas.reference
        wing.areas.exposed  = 0.75 * wing.areas.wetted
  
  
    cruise_altitude= mission['climb_3'].altitude_end
    conditions = atmo.compute_values(cruise_altitude)
    sizing_segment = SUAVE.Components.Propulsors.Segments.Segment()
    sizing_segment.M   = mission['cruise'].air_speed/conditions.speed_of_sound       
    sizing_segment.alt = cruise_altitude
    sizing_segment.T   = conditions.temperature        
    sizing_segment.p   = conditions.pressure
    conditions0 = atmo.compute_values(12500.*Units.ft) #cabin pressure
    p0 = conditions0.pressure
    fuselage_diff_pressure=max(conditions0.pressure-conditions.pressure,0)
    fuselage.differential_pressure = fuselage_diff_pressure
    
    battery   =base.energy_network['battery']
    ducted_fan=base.propulsors['ducted_fan']
    SUAVE.Methods.Power.Battery.Sizing.initialize_from_energy_and_power(battery,Ereq,Preq)
    battery.current_energy=[battery.max_energy] #initialize list of current energy
    m_air       =SUAVE.Methods.Power.Battery.Variable_Mass.find_total_mass_gain(battery)
    #now add the electric motor weight
    motor_mass=ducted_fan.number_of_engines*SUAVE.Methods.Weights.Correlations.Propulsion.air_cooled_motor((Preq)*Units.watts/ducted_fan.number_of_engines)
    propulsion_mass=SUAVE.Methods.Weights.Correlations.Propulsion.integrated_propulsion(motor_mass/ducted_fan.number_of_engines,ducted_fan.number_of_engines)
    
    ducted_fan.mass_properties.mass=propulsion_mass
   
    breakdown = analyses.configs.base.weights.evaluate()
    breakdown.battery=battery.mass_properties.mass
    breakdown.air=m_air
    base.mass_properties.breakdown=breakdown
    m_fuel=0.
    #print breakdown
    base.mass_properties.operating_empty     = breakdown.empty 
    #weight =SUAVE.Methods.Weights.Correlations.Tube_Wing.empty_custom_eng(vehicle, ducted_fan)
    m_full=breakdown.empty+battery.mass_properties.mass+breakdown.payload
    m_end=m_full+m_air
    base.mass_properties.takeoff                 = m_full
    base.store_diff()

    # Update all configs with new base data    
    for config in configs:
        config.pull_base()

    
    ##############################################################################
    # ------------------------------------------------------------------
    #   Define Configurations
    # ------------------------------------------------------------------
    landing_config=configs.landing
    landing_config.wings['main_wing'].flaps.angle =  20. * Units.deg
    landing_config.wings['main_wing'].slats.angle  = 25. * Units.deg
    landing_config.mass_properties.landing = m_end
    landing_config.store_diff()
    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------
    
    return 

###############################################################################################################################################
# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------
def mission_setup( analyses,climb_alt_1,climb_alt_2,climb_alt_3, climb_alt_4, climb_alt_5, desc_alt_1, desc_alt_2,Vclimb_1, Vclimb_2, Vclimb_3, Vclimb_4,Vclimb_5,  V_cruise , cruise_range):
   
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'the_mission'
    
    # atmospheric model
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet = SUAVE.Attributes.Planets.Earth()
    
    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    
    mission.airport = airport
    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments
    
    # base segment
    base_segment = Segments.Segment()
    
    # ------------------------------------------------------------------
    #   First Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------
    
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"
    
    # connect vehicle configuration
    segment.analyses.extend( analyses.takeoff )
    segment.altitude_start=0.0
    segment.altitude_end  =climb_alt_1* Units.km
    # pick two:
    segment.air_speed    = Vclimb_1        # m/s
    segment.climb_rate       = 3000.*(Units.ft/Units.minute)

    
    # add to mission
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------
    #   Second Climb Segment: constant Speed, constant segment angle 
    # ------------------------------------------------------------------    
   
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_2"
    segment.analyses.extend( analyses.cruise )
    # connect vehicle configuration
    segment.altitude_end=climb_alt_2* Units.km
    
    # pick two:

    segment.air_speed       =Vclimb_2
    segment.climb_rate      =2500.*(Units.ft/Units.minute)
    
    # add to mission
    mission.append_segment(segment)

    
    # ------------------------------------------------------------------
    #   Third Climb Segment: constant velocity, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_3"

    # connect vehicle configuration
    #segment.config = vehicle.configs.cruise
    segment.analyses.extend( analyses.cruise )
    segment.altitude_end=climb_alt_3* Units.km
    # pick two:
    segment.air_speed        = Vclimb_3
    segment.climb_rate       =1800.*(Units.ft/Units.minute)
    
    # add to mission
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Fourth Climb Segment: constant velocity, constant segment angle 
    # ------------------------------------------------------------------    
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_4"

    # connect vehicle configuration
    #segment.config = vehicle.configs.cruise
    segment.analyses.extend( analyses.cruise )
    segment.altitude_end=climb_alt_4* Units.km
    # pick two:
    segment.air_speed        = Vclimb_4
    segment.climb_rate       =900.*(Units.ft/Units.minute)
    
    # add to mission
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------
    #   Fifth Climb Segment: Constant Speed, Constant Climb Rate
    # ------------------------------------------------------------------    
    
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_5"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )
    segment.altitude_end=climb_alt_5* Units.km
    
    # pick two:
    segment.air_speed      = Vclimb_5
    segment.climb_rate       =300.*(Units.ft/Units.minute)
    # add to mission
    mission.append_segment(segment)    
    
    
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"
    
    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )
    segment.air_speed       = V_cruise    # m/s
    segment.distance   =cruise_range*Units.km
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   First Descent Segment: constant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_1"
    
    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )
    
    # segment attributes
    segment.altitude_end    =  desc_alt_1  * Units.km
    segment.air_speed       = 440.0 * Units.knots
    segment.descent_rate    =2600. * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment)
    

    # ------------------------------------------------------------------    
    #   Second Descent Segment: constant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_2"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.altitude_end   =  desc_alt_2 * Units.km # km
    segment.air_speed    = 365.0 * Units.knots
    segment.descent_rate    =2300. * Units['ft/min']
    # append to mission
    mission.append_segment(segment)

       # ------------------------------------------------------------------    
    #   Third Descent Segment: constant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_3"

    # connect vehicle configuration
    segment.analyses.extend( analyses.landing )
    # segment attributes
    segment.altitude_end   =  0.  # km
    segment.air_speed    = 250.0 * Units.knots
    segment.descent_rate = 1500. * Units['ft/min']
    # append to mission
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   Mission definition complete   
    
    return mission




# ----------------------------------------------------------------------
#   Evaluate the Mission
# ----------------------------------------------------------------------
def evaluate_mission(configs,mission):
    
    # ------------------------------------------------------------------    
    #   Run Mission
    # ------------------------------------------------------------------
    
    results = mission.evaluate()
    
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
    print 'e_current_min=',e_current_min
    print "e_total=", results.e_total
    print "Pmax=", Pmax
    print "e_current_min=", e_current_min
    return results

#########################################################################################################################
def evaluate_field_length(configs,analyses,mission,results):
    
    # unpack
    airport = mission.airport
    
    takeoff_config = configs.takeoff
    landing_config = configs.landing
   
    # evaluate
    TOFL = estimate_take_off_field_length(takeoff_config,analyses,airport)
    LFL = estimate_landing_field_length (landing_config, analyses,airport)
    
    # pack
    field_length = SUAVE.Core.Data()
    field_length.takeoff = TOFL[0]
    field_length.landing = LFL[0]
    
    results.field_length = field_length
 
    
    return results
    
#########################################################################################################################
def configs_setup(vehicle):
    
    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------
    
    configs = SUAVE.Components.Configs.Config.Container()
    
    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)
    
    # ------------------------------------------------------------------
    #   Cruise Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'cruise'
    
    configs.append(config)
    
    # ------------------------------------------------------------------
    #   Takeoff Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'takeoff'
    config.wings['main_wing'].flaps.angle = 20. * Units.deg
    config.wings['main_wing'].slats.angle = 25. * Units.deg
    config.V2_VS_ratio = 1.21
    
    configs.append(config)
   
    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'landing'
    
    config.wings['main_wing'].flaps.angle = 30. * Units.deg
    config.wings['main_wing'].slats.angle = 25. * Units.deg
    config.Vref_VS_ratio = 1.23
    configs.append(config)    
    
    # done!
    return configs

def full_setup(inputs,m_guess,Ereq_guess):
    #unpack inputs
    i=0;
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
    
    #V_cruise =inputs[i];         i+=1
    V_cruise=230.
    


    # vehicle data
    vehicle = vehicle_setup(m_guess,Ereq_guess, Preq, climb_alt_5,wing_sweep, alpha_rc, alpha_tc, vehicle_S, V_cruise)
   
    configs  = configs_setup(vehicle)
    
    # vehicle analyses
    configs_analyses = analyses_setup(configs)
    
    # mission analyses
    mission = mission_setup(configs_analyses,climb_alt_1,climb_alt_2,climb_alt_3, climb_alt_4,climb_alt_5, desc_alt_1, desc_alt_2, Vclimb_1, Vclimb_2, Vclimb_3, Vclimb_4,Vclimb_5,V_cruise , cruise_range)
    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses
    
    return configs, analyses
def analyses_setup(configs):
    
    analyses = SUAVE.Analyses.Analysis.Container()
    
    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config)
        analyses[tag] = analysis
       
    # adjust analyses for configs
    
    # takeoff_analysis
    analyses.takeoff.aerodynamics.drag_coefficient_increment = 0.1000
    
    # landing analysis
    aerodynamics = analyses.landing.aerodynamics
    # do something here eventually
    
    return analyses
def base_analysis(vehicle):

    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------     
    analyses = SUAVE.Analyses.Vehicle()
    
    # ------------------------------------------------------------------
    #  Basic Geometry Relations
    sizing = SUAVE.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)
    
    # ------------------------------------------------------------------
    #  Weights
    weights = SUAVE.Analyses.Weights.Weights()
    weights.settings.empty_weight_method= \
           SUAVE.Methods.Weights.Correlations.Tube_Wing.empty_custom_eng
    weights.vehicle = vehicle
    analyses.append(weights)
   
    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)
    
    # ------------------------------------------------------------------
    #  Stability Analysis
    stability = SUAVE.Analyses.Stability.Fidelity_Zero()
    stability.geometry = vehicle
    analyses.append(stability)
    
    # ------------------------------------------------------------------
    #  Energy
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.energy_network #what is called throughout the mission (at every time step))
    analyses.append(energy)
    
    # ------------------------------------------------------------------
    #  Planet Analysis
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)
    
    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)    
    
    # done!
    return analyses
def missions_setup(base_mission):

    # the mission container
    missions = SUAVE.Analyses.Mission.Mission.Container()
    missions.base = base_mission
    
    return missions
    #########################################################################################################################
# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def post_process(mission,configs, results):
    battery=configs.base.energy_network['battery']

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
    #   Mass Gain Rate
    # ------------------------------------------------------------------
    plt.figure("Mass Accumulation Rate")
    axes = plt.gca()    
    for i in range(len(results.segments)):     
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        mdot = results.segments[i].conditions.weights.vehicle_mass_rate[:,0]
        axes.plot(time, -mdot, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Mass Accumulation Rate(kg/s)')
    axes.grid(True)    
    
    


    # ------------------------------------------------------------------    
    #   Trajectory
    # ------------------------------------------------------------------
    plt.figure("Trajectory")
    axes = plt.gca()    
    for i in range(len(results.segments)):   
        ground_dist    = results.segments[i].conditions.frames.inertial.position_vector[:,0] / Units.km
        altitude = results.segments[i].conditions.freestream.altitude[:,0] /Units.km
        axes.plot(ground_dist, altitude, 'bo-')
    axes.set_xlabel('Ground Distance(km)')
    axes.set_ylabel('Altitude (km)')
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
    '''
    # ------------------------------------------------------------------    
    #   Vehicle Mass
    # ------------------------------------------------------------------
    fig=plt.figure("Vehicle Mass")
    axes = plt.gca()
    for i in range(len(results.segments)):
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        mass = results.segments[i].conditions.weights.total_mass[:,0]
        axes.plot(time, mass, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Vehicle Mass (kg)')
    axes.grid(True)
    
    
    # ------------------------------------------------------------------    
    #   State of Charge
    # ------------------------------------------------------------------
    
    title = "State of Charge"
    fig=plt.figure(title)
    
    axes = plt.gca() 
    #print results.segments[0].Ecurrent[0]
    for segment in results.segments:
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        axes.plot(time, segment.conditions.propulsion.battery_energy[:,0]/battery.max_energy,'bo-')
        #print 'E=',segment.conditions.propulsion.battery_energy
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('State of Charge of the Battery')
        axes.grid(True)

     # ------------------------------------------------------------------    
    #   Power
    # ------------------------------------------------------------------
    
    title = "Battery Discharge Power"
    fig=plt.figure(title)
    axes = plt.gca() 
    #print results.segments[0].Ecurrent[0]
    for segment in results.segments:
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        axes.plot(time, -segment.conditions.propulsion.battery_draw[:,0]/Units.MW,'bo-')
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Electric Power (MW)')
        axes.grid(True)
        
    '''    
    # ------------------------------------------------------------------    
    #  Mass, State of Charge, Power
    # ------------------------------------------------------------------
    fig = plt.figure("Electric Aircraft Outputs")
    fig.set_figheight(10)
    fig.set_figwidth(6.5)
    for segment in results.segments.values():
        
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        mass = segment.conditions.weights.total_mass[:,0]
        state_of_charge=segment.conditions.propulsion.battery_energy[:,0]/battery.max_energy
        battery_power=-segment.conditions.propulsion.battery_draw[:,0]/Units.MW

        axes = fig.add_subplot(3,1,1)
        axes.plot( time , mass , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('vehicle mass')
        axes.grid(True)
        
        axes = fig.add_subplot(3,1,2)
        axes.plot( time , state_of_charge , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('State of Charge')
        axes.grid(True)
        
        axes = fig.add_subplot(3,1,3)
        axes.plot( time , battery_power , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Battery Discharge Power')
        axes.grid(True)
    
        
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
        
        
        
    # ------------------------------------------------------------------
    #   Drag Breakdown
    # ------------------------------------------------------------------
    fig = plt.figure("Drag Components")
    axes = plt.gca()
    for i, segment in enumerate(results.segments.values()):

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        drag_breakdown = segment.conditions.aerodynamics.drag_breakdown
        cdp = drag_breakdown.parasite.total[:,0]
        cdi = drag_breakdown.induced.total[:,0]
        cdc = drag_breakdown.compressible.total[:,0]
        cdm = drag_breakdown.miscellaneous.total[:,0]
        cd  = drag_breakdown.total[:,0]

  
        axes.plot( time , cdp , 'ko-', label='CD_P' )
        axes.plot( time , cdi , 'bo-', label='CD_I' )
        axes.plot( time , cdc , 'go-', label='CD_C' )
        axes.plot( time , cdm , 'yo-', label='CD_M' )
        axes.plot( time , cd  , 'ro-', label='CD'   )
        if i == 0:
             axes.legend(loc='upper center')            
           

    axes.set_xlabel('Time (min)')
    axes.set_ylabel('CD')
    axes.grid(True)    
        
        
        
        
        
        
    plt.show()
    raw_input('Press Enter To Quit')
    
    
    return     
   
  
  
if __name__ == '__main__':
    main()
