""" Performance.py: Methods for Mission and Performance Analysis """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import math
import Units
import copy

from SUAVE.Core            import Data
from SUAVE.Attributes.Results   import Result, Segment
from Utilities                  import chebyshev_data, pseudospectral

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def evaluate_mission(mission):

    results = copy.deepcopy(mission)

    # evaluate each segment 
    for i in range(len(results.segments)):

        segment = results.segments[i]
        # print mission.segments[i]

        # determine ICs for this segment
        if i == 0:                                              # first segment of mission
            segment.m0 = results.m0
            segment.t0 = 0.0
        else:                                                   # inherit ICs from previous segment
            segment.m0 = results.segments[i-1].m[-1]
            segment.t0 = results.segments[i-1].t[-1]

        # run segment
        pseudospectral(segment)

    return results
          
def EvaluateSegment(config,segment,ICs,solver):

    problem = Problem()
    options = Options()                                # ODE integration options
    problem.config = config
    problem.segment = segment
    err = False
    
    if segment.type.lower() == 'climb':             # Climb segment
        
        problem.tag = "Climb"
        problem.f = EquationsOfMotion.General2DOF

        # initial conditions
        problem.z0 = ICs

        if solver.lower() == "RK45":

            # boundary conditions
            problem.zmin = np.zeros(5)
            problem.zmin[0] = None                              # x: no minimum
            problem.zmin[1] = None                               # Vx: must be > 0
            problem.zmin[2] = 0.0                               # y: y > 0 (must climb)
            problem.zmin[3] = None                               # Vy: Vy > 0 (must climb)
            problem.zmin[4] = config.Mass_Props.m_flight_min     # m: mimumum mass (out of fuel)
        
            problem.zmax = np.zeros(5)
            problem.zmax[0] = None                              # x: no maximum
            problem.zmax[1] = None                              # Vx: liftoff when Vx >= V_liftoff
            problem.zmax[2] = segment.altitude[1]               # y: climb target
            problem.zmax[3] = None                              # Vy: no maximum 
            problem.zmax[4] = None                              # m: no maximum

            # check if this problem is well-posed 
            #y0 = segment.altitude[0]
            #p, T, rho, a, mew = segment.atmosphere.ComputeValues(y0,"all")

            # estimate time step
            problem.h0 = 0.01

        elif solver.lower() == "PS":

            # final conditions   
            problem.zf = np.zeros(5)
            problem.zf[0] = None                              # x: no maximum
            problem.zf[1] = None                              # Vx: liftoff when Vx >= V_liftoff
            problem.zf[2] = segment.altitude                  # y: climb target
            problem.zf[3] = None                              # Vy: no maximum 
            problem.zf[4] = None                              # m: no maximum
            problem.tf = 0.0                                  # final time

            # check if this problem is well-posed


    elif segment.type.lower() == 'descent':         # Descent segment
        
        options.end_type = "Altitude"
        options.DOFs = 2

    elif segment.type.lower() == 'cruise':          # Cruise segment
        
        options.end_type = segment.end.type
        options.end_value = segment.end.value
        options.DOFs = 2

    elif segment.type.lower() == 'takeoff':         # Takeoff segment (on runway)
        
        problem.tag = "Takeoff"
        problem.f = EquationsOfMotion.Ground1DOF

        # initial conditions
        problem.z0 = ICs

        # final conditions
        problem.zmin = np.zeros(5)
        problem.zmin[0] = None                              # x: no minimum
        problem.zmin[1] = 0.0                               # Vx: must be > 0
        problem.zmin[2] = None                              # y: N/A
        problem.zmin[3] = None                              # Vy: N/A
        problem.zmin[4] = config.Mass_Props.m_flight_min     # m: mimumum mass (out of fuel)
        
        problem.zmax = np.zeros(5)
        problem.zmax[0] = None                              # x: no maximum
        problem.zmax[1] = config.V_liftoff                  # Vx: liftoff when Vx >= V_liftoff
        problem.zmax[2] = None                              # y: N/A
        problem.zmax[3] = None                              # Vy: N/A
        problem.zmax[4] = None                              # m: no maximum

        # check this configuration / segment 
        #y0 = segment.altitude[0]
        #p, T, rho, a, mew = segment.atmosphere.ComputeValues(y0,"all")

        # estimate time step
        problem.h0 = 1.0

    elif segment.type.lower() == 'landing':         # landing segemnt (on runway)
        
        # set up initial conditions
        z0[0] = mission_segment.altitude[0]            # km
        z0[1] = 0.0                                    # km
        options.stop = "Velocity"
        options.EOM = "1DOF"

    elif segment.type.lower() == 'PASS':
        results_segment = EvaluatePASS(config,segment)
    else:
        print "Unknown mission segment type: " + segment.type + "; mission aborted"
    
    if err:
        print "Segment / Configuration check failed"
        results_segment = None
    else:

        print problem.z0
        print problem.h0

        # integrate EOMs:
        solution = RK45(problem,options)

        # package results:
        results_segment = Segment()                     # RESULTS segment container
        results_segment.tag = problem.tag
        results_segment.state.t = solution.t
        results_segment.state.x = solution.z[:,0]
        results_segment.state.Vx = solution.z[:,1]
        results_segment.state.y = solution.z[:,2]
        results_segment.state.Vy = solution.z[:,3]
        results_segment.state.m = solution.z[:,4]
        results_segment.exit = solution.exit
        results_segment.mission_segment = segment
        results_segment.config = config

    return results_segment

def CheckSegment(config,segment,ICs):

    check = Data()
    check.errors = []; check.warnings = []; check.error = False; check.warning = False

    if segment.type.lower() == 'climb':                     # Climb segment

        name = "Climb Segment: "
        
        # check variables existence & values
        try: 
            segment.altitude
        except NameError:
            check.errors.append(name + "altitude value not defined; please check inputs"); check.error = True
        else:
            if segment.altitude <= 0.0:
                check.errors.append(name + "final altitude is <= 0; please check inputs"); check.error = True
        
        try: 
            segment.atmosphere
        except NameError:
            check.errors.append(name + "atmosphere not defined; please check inputs"); check.error = True
        else:
            pass

        try: 
            config.Function.Propulsion
        except NameError:
            check.errors.append(name + "no propulsion function defined; please check inputs"); check.error = True
        else:
            pass

        try: 
            config.Function.Aero
        except NameError:
            check.errors.append(name + "no aerodynamic function defined; please check inputs"); check.error = True
        else:
            pass

        # check ICs
        if not check.error:
            if ICs[1] <= 0.0:
                check.errors.append(name + "vehicle is at rest flying backward"); check.error = True
            if ICs[2] >= segment.altitude:
                check.errors.append(name + "vehicle is above specified final altitude for climb segment"); check.error = True
            if ICs[3] < 0.0:
                check.warnings.append(name + "vehicle is falling at beginning of climb segment"); check.warning = True
     
        # check physics
        if not check.error:

            # start point
            state = State(); z = ICs
            state.ComputeState(z,segment,config)
   
            L, D = config.Functions.Aero(state)*state.q*config.S            # N
            T, mdot = config.Functions.Propulsion(state)                    # N, kg/s

            # common trig terms
            sin_gamma_alpha = np.sin(state.gamma - state.alpha)
            cos_gamma_alpha = np.cos(state.gamma - state.alpha)
    
            # drag terms
            Dv = np.zeros(2)
            Dv[0] = -D*cos_gamma_alpha                              # x-component 
            Dv[1] = -D*sin_gamma_alpha                              # y-component

            # lift terms
            Lv = np.zeros(2)
            Lv[0] = L*sin_gamma_alpha                               # x-component 
            Lv[1] = L*cos_gamma_alpha                               # y-component

            # thurst terms
            Tv = np.zeros(2)
            Tv[0] = T*np.cos(np.radians(state.gamma + state.delta)) # x-component
            Tv[1] = T*np.sin(np.radians(state.gamma + state.delta)) # y-component

            d2xdt2 = (Dv[0] + Lv[0] + Tv[0])/state.m
            d2ydt2 = (Dv[1] + Lv[1] + Tv[1])/state.m - state.g

    return err, warn, reason

def initialize_takeoff(config,segment):
    
    # initialize
    takeoff = None; N = segment.options.Npoints; m = 5

    # estimate liftoff speed in this configuration
    liftoff_state = estimate_takeoff_speed(config,segment)   
    
    # check for T > D at liftoff
    if liftoff_state.D > liftoff_state.T:
        print "Vehcile cannot take off: Drag > Trust at liftoff speed in this configuration."
        print "Adding phantom lift to compensate."
        
        CL_phantom = 0.0
        while dV > tol:        
       
            z[1] = V_lo
            state.compute_state(z,config,segment,["no vectors", "constant altitude"])
            V_lo_new = np.sqrt(2*(m0*g0 - state.T*np.sin(state.gamma))/(state.CL*state.rho*config.S))
            dV = np.abs(V_lo_new - V_lo)
            print "dV = ", dV
            V_lo = V_lo_new 

    else:
        T_lo = state.T; mdot_lo = state.mdot

    # get average properties over takeoff
    z[1] = 0.0; state.compute_state(z,config,segment,["no vectors", "constant altitude"])
    T0 = state.T; mdot0 = state.mdot
    T = (T0 + T_lo)/2; mdot = (mdot0 + mdot_lo)/2

    # estimate time to liftoff
    print state.CD, state.CL
    print 0.5*state.CD*state.rho*config.S*V_lo**2
    print state.T
    C = 0.5*state.CD*state.rho*config.S
    k1 = np.sqrt(C*T)/mdot; k2 = np.sqrt(C/T)
    p = k1*np.arctanh(k2*V_lo)
    
    tf = (m0/mdot)*(1 - np.exp(-p))

    # estimate state variables 
    t_cheb, D, I = chebyshev_data(N)
    t = t_cheb*tf
    z = np.zeros((N,m))
    z[:,1] = k2*np.tanh(k1*np.log(m0/(m0 - mdot*t)))                # Vx
    z[:,2] = segment.airport.altitude*np.ones(N)      # y

    # integrate Vx for x(t)
    z[:,0] = np.dot(I,z[:,1])                                       # x

    # pack up problem class
    takeoff = Problem()
    takeoff.tag = "Takeoff"
    takeoff.Npoints = N

    takeoff.f = EquationsOfMotion.ground_1DOF
    takeoff.FC = EquationsOfMotion.takeoff_speed
    takeoff.scale.z = zscale   
    takeoff.scale.t = tf
    takeoff.scale.F = m0*g0
    takeoff.t = t
    takeoff.z0 = z
    takeoff.t0 = 0.0                # initial time
    takeoff.tf = tf                 # final time guess
    takeoff.config = config

    return takeoff

def estimate_takeoff_speed(config,segment,tol=1e-6,guess=0.0):

    # initilize
    N = segment.options.Npoints; m = 5; z = np.zeros(m)
    
    state = State(); 
    m0 = config.Mass_Props.m_takeoff                                 # kg
    g0 = segment.planet.sea_level_gravity                          # m/s^2
    z[2] = segment.airport.altitude                                 # m    
    z[4] = m0                                                       # kg
    state.alpha = np.radians(segment.climb_angle)                   # rad

    # estimate liftoff speed in this configuration
    dV = 1.0; V_lo = guess
    while dV > tol:        
       
        z[1] = V_lo
        state.compute_state(z,config,segment,["no vectors", "constant altitude"])
        V_lo_new = np.sqrt(2*(m0*g0 - state.T*np.sin(state.gamma))/(state.CL*state.rho*config.S))
        dV = np.abs(V_lo_new - V_lo)
        # print "dV = ", dV
        V_lo = V_lo_new

    return state

def compute_maximum_ground_speed(config,segment,tol=1e-6,guess=0.0):

    # initilize
    N = segment.options.Npoints; m = 5 
    
    state = State(); z = np.zeros(m)
    z[2] = segment.airport.altitude                                 # m    
    z[4] = config.Mass_Props.m_takeoff                                 # kg
    state.alpha = 0.0                                               # rad

    # estimate liftoff speed in this configuration
    dV = 1.0; V = guess
    while dV > tol:        
       
        z[1] = V
        state.compute_state(z,config,segment,["no vectors", "constant altitude"])
        V_new = np.sqrt(2*state.T*np.cos(state.delta)/(state.CD*state.rho*config.S))
        dV = np.abs(V_new - V)
        V = V_new

    return state

def ComputeICs(segment):
    raise NotImplementedError

def EvaluatePASS(vehicle,mission):
    """ Compute the Pass Performance Calculations using SU AA 241 course notes 
    """

    # unpack
    maxto = vehicle.Mass.mtow
    mzfw_ratio = vehicle.Mass.fmzfw
    sref = vehicle.Wing['main_wing'].sref
    sfc_sfcref = vehicle.Turbo_Fan['TheTurboFan'].sfc_TF
    sls_thrust = vehicle.Turbo_Fan['TheTurboFan'].thrust_sls
    eng_type = vehicle.Turbo_Fan['TheTurboFan'].type_engine
    out = Data()

    # Calculate
    fuel_manuever = WeightManeuver(maxto)
    fuel_climb_added = WeightClimbAdded(vehicle,mission,maxto)
    reserve_fuel = FuelReserve(mission,maxto,mzfw_ratio,0)
    out.range,fuel_cruise = CruiseRange(vehicle,mission,maxto,fuel_burn_maneuever,fuel_climb_added,reserve_fuel,sfc_sfcref,sls_thrust,eng_type,mzfw_ratio)
    out.fuel_burn = (2 * fuel_manuever) + fuel_climb_added + fuel_cruise
    out.tofl = TOFL(vehicle,mission,maxto,sref,sfc_sfcref,sls_thrust,eng_type)
    out.climb_grad = ClimbGradient(vehicle,mission,maxto,sfc_sfcref,sls_thrust,eng_type)
    out.lfl = LFL(vehicle,mission,maxto,mzfw_ratio,fuel_burn_maneuever,reserve_fuel,sref,sfc_sfcref,sls_thrust,eng_type)

    return out

def WeightManeuver(maxto):
    """Fuel burned in the warm-up, taxi, take-off, approach, and landing segments. Assumed to be 0.7% of max takeoff weight
        
        Assumptions:
            all segments combined have a fixed fuel burn ratio
        
        Input:
        
        Outputs:
            fuel_burn_maneuver
    """
    # AA 241 Notes Section 11.4
    
    # Calculate
    fuel_burn_maneuever = 0.0035 * maxto # Only calculates all the total Maneuever fuel burn
    
    return fuel_burn_maneuever

def TOFL(vehicle,mission,maxto,sref,sfc_sfcref,sls_thrust,eng_type):
    """ Calculating the TOFL based on a parametric fit from the AA 241 Notes
        
        Assumptions:
        
        Inputs:
            mission.segments['Take-Off'].alt
            mission.segments['Take-Off'].flap_setting
            mission.segments['Take-Off'].slat_setting
            mission.segments['Take-Off'].mach
            mission.segments['Take-Off'].atmo
        
        Outputs:
            tofl
        
    """
            
    # AA 241 Notes Section 11.1
    
    # Atmosphere
    atmo = mission.segments['Take-Off'].atmo # Not needed while only have ISA
    atm = Atmosphere()
    atm.alt = mission.segments['Take-Off'].alt # Needs to be in meters
    atm.dt = 0.0  #Return ISA+(value)C properties
    Climate.EarthIntlStandardAtmosphere(atm)
    
    # Unpack
    rho_standard = constants.rho0 #density at standard sea level temperature and pressure
    rho = atm.density # density at takeoff altitude
    speed_of_sound = atm.vsnd # should be in m/s
    flap = mission.segments['Take-Off'].flap_setting
    slat = mission.segments['Take-Off'].slat_setting
    mach_to = mission.segments['Take-Off'].mach
    
    # Calculate
        
    clmax = 0 # Need way to calculate in Aerodynamics<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    sfc,thrust_takeoff = Propulsion.engine_analysis( 0.7 * mach_to, speed_of_sound,sfc_sfcref,sls_thrust,eng_type)  #use 70 percent of takeoff velocity
    sigma = rho / rho_standard
    engine_number = len(vehicle.Propulsors.get_TurboFans())
    index = maxto ** 2 / ( sigma * clmax * sref * (engine_number * thrust_takeoff) )
    if engine_number == 2:
        tofl = 857.4 + 28.43 * index + 0.0185 * index ** 2
    elif engine_number == 3:
        tofl = 667.9 + 26.91 * index + 0.0123 * index ** 2
    elif engine_number == 4:
        tofl = 486.7 + 26.20 * index + 0.0093 * index ** 2
    else:
        tofl = 486.7 + 26.20 * index + 0.0093 * index ** 2
        print 'Incorrect number of engines for TOFL' #Error Message for <2 or >4 engines
    
    return tofl

def WeightClimbAdded(vehicle,mission,maxto):
    """Fuel increment added to cruise fuel burned over distance when aircraft is climbing to the cruise altitude
        
        Assumptions:
        
        Inputs:
            mission.segments['Initial Cruise'].mach
            mission.segments['Initial Cruise'].atmo
            mission.segments['Initial Cruise'].alt
        
        Outputs:
            fuel_burn_climb_inc
        
     """
    # AA 241 Notes Section 11.3
    
    # Atmosphere
    atmo = mission.segments['Initial Cruise'].atmo
    atm = Atmosphere()
    atm.alt  = mission.segments['Initial Cruise'].alt # in km
    atm.dt = 0.0  #Return ISA+(value)C properties
    Climate.EarthIntlStandardAtmosphere(atm)
    
    # Unpack
    cruise_mach = mission.segments['Initial Cruise'].mach
    speed_of_sound = atm.vsnd #in m/s
    
    # Calculate
    cruise_speed = speed_of_sound * cruise_mach / 1000 * 3600 # in km/hr
    cruise_velocity = Units.ConvertDistance(cruise_speed,'km','nm') #Converting to knots
    alt_kft = Units.ConvertLength( alt * 1000 ,'m','ft') / 1000 #Converting to kft
    fuel_burn_climb_inc = maxto * (alt_kft / 31.6 + (cruise_velocity / 844) ** 2)
    
    return fuel_burn_climb_inc

def ClimbGradient(vehicle,mission,maxto,sfc_sfcref,sls_thrust,eng_type):
    """Estimating the climb gradient of the aircraft
        
        Assumptions:
        
        Inputs:
            mission.segments['Initial Cruise'].atmo
            mission.segments['Initial Cruise'].alt
            mission.segments['Take-Off'].mach
            vehicle.Turbo_Fan['TheTurboFan'].pos_dy
            vehicle.Wing['Vertical Tail'].span
        
        Outputs:
            climb_grad
        
        
     """
    # AA 241 Notes Section 11.3
    
    # Atmosphere
    atmo = mission.segments['Initial Cruise'].atmo
    atm = Atmosphere()
    atm.alt  = mission.segments['Initial Cruise'].alt # in km
    atm.dt = 0.0  #Return ISA+(value)C properties
    Climate.EarthIntlStandardAtmosphere(atm)
    
    # Unpack
    mach_to = mission.segments['Take-Off'].mach
    speed_of_sound = atm.vsnd # should be in m/s
    pressure = atm.pressure # ambient static pressure
    rho = atm.density # density at takeoff altitude
    y_engine = vehicle.Turbo_Fan['TheTurboFan'].pos_dy # distance from fuselage centerline to critical engine
    height_vert = vehicle.Wing['Vertical Tail'].span # Make sure this is the right dimension
    
    length_vert = 0 # distance from c.g. to vertical tail a.c. <<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    
    # Calculate
    sfc,thrust_single = Propulsion.engine_analysis(mach_to,speed_of_sound,sfc_sfcref,sls_thrust,eng_type) #Critical takeoff thrust
    
    drag_climb = 0 # Need to determine how to calculate the drag in climb configuration <<<
    
    area_inlet = 0 # Need to calculate the area of the engine inlet <<<<<<<<<<<<<<<<<<<<<<<
    
    drag_windmill = 0.0044 * pressure * area_inlet
    vel_climb = mach_to * speed_of_sound # Velocity at takeoff
    pressure_dyn = 0.5 * rho * vel_climb ** 2
    drag_trim = y_engine ** 2 * (thrust_single+drag_windmill) ** 2 / (pressure_dyn * math.pi * height_vert ** 2 * length_vert ** 2)
    drag_engine_out = drag_climb + drag_windmill + drag_trim
    climb_grad = (thrust_single-drag_engine_out) / maxto # at constant speed
    
    return climb_grad

def CruiseRange(vehicle,mission,maxto,fuel_burn_maneuever,fuel_burn_climb_inc,reserve_fuel,sfc_sfcref,sls_thrust,eng_type,mzfw_ratio):
    """Calculate the range from the initial and final weight, uses the Breguet range equation to calculate the range after the segment. Output is in kilometers
        
        Assumptions:
            Steady level flight so lift = weight and thrust = drag
            The Mach number and speed of sound were average values based on the prescribed initial and final values specified 
        
        Inputs:
            mzfw_ratio = vehicle.Mass.fmzfw
            mission.segments['Initial Cruise'].atmo
            mission.segments['Initial Cruise'].alt
            mission.segments['Final Cruise'].alt
        
        Outputs
            range_cruise
            fuel_cruise
        
    """
    # AA 241 Notes Section 11.4
    
    # Atmosphere
    atmo = mission.segments['Initial Cruise'].atmo
    atm = Atmosphere()
    atm.dt = 0.0  #Return ISA+(value)C properties
    
    # Initial Cruise Atmospheric Conditions
    atm.alt  = mission.segments['Initial Cruise'].alt # in km
    Climate.EarthIntlStandardAtmosphere(atm)
    speed_of_sound_in = atm.vsnd
    
    # Final Cruise Atmospheric Conditions
    atm.alt = mission.segments['Final Cruise'].alt
    Climate.EarthIntlStandardAtmosphere(atm)
    speed_of_sound_fin = atm.vsnd
    
    # Unpack
    mach_cr_initial = mission.segments['Initial Curise'].mach
    mach_cr_final = mission.segments['FInal Cruise'].mach
    
    # Calculate
    mzfw = maxto * mzfw_ratio
    weight_initial = maxto - fuel_burn_maneuever - fuel_burn_climb_inc
    weight_final = mzfw + fuel_burn_maneuever + reserve_fuel
    lift = (weight_final + weight_initial) / 2 # Assume Steady Level Flight
    mach_cr_av = (mach_cr_final + mach_cr_initial) / 2 # Average Mach number
    speed_of_sound_av = (speed_of_sound_fin +speed_of_sound_in) / 2 # Average speed of sound
    sfc,thrust = Propulsion.engine_analysis( mach_cr_average, speed_of_sound_av, sfc_sfcref,sls_thrust,eng_type)
    drag = thrust # Assume steady level flight
    cruise_velocity = (mach_cr_final * speed_of_sound_fin + mach_cr_initial *speed_of_sound_in) / 2 * 3600 / 1000 # averaged velocities at initial and final cruise
    range_cruise = cruise_velocity / sfc * lift / drag * math.log( weight_initial /weight_final) #in km
    fuel_cruise = weight_initial - weight_final
    
    return (range_cruise,fuel_cruise)

def LFL(vehicle,mission,maxto,mzfw_ratio,fuel_burn_maneuever,reserve_fuel,sref,sfc_sfcref,sls_thrust,eng_type):
    
    """ Calculating the landing distance of an aircraft through use of combined aircraft deceleration at constant altitude and ground deceleration
        
        Assumptions:
            Braking coefficient was assumed to be 0.2 (from http://www.airporttech.tc.faa.gov/naptf/att07/2002%20TRACK%20S.pdf/S10.pdf)
        
        Inputs:
        
        Outputs:
        
        
    """
    # AA 241 Notes Section 11.2
        
    # Atmosphere
    atmo = mission.segments['Landing'].atmo
    atm = Atmosphere()
    atm.alt  = mission.segments['Landing'].alt # in km
    atm.dt = 0.0  #Return ISA+(value)C properties
    Climate.EarthIntlStandardAtmosphere(atm)
        
    # Unpack
    rho = atm.density
    speed_of_sound = atm.vsnd
    flap = mission.segments['Landing'].flap_setting
    slat = mission.segments['Landing'].slat_setting
    mach_land = mission.segments['Landing'].mach
        
    # Calculations
    mzfw = maxto * mzfw_ratio
    weight_landing = mzfw + fuel_burn_maneuever + reserve_fuel
    
    clmax = 0 # Need to calculate the maximum lift coefficient<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    vel_stall = math.sqrt( 2 * weight_landing / (rho * clmax *sref))
    vel_50 = 1.3 * vel_stall
    
    lift_landing = 0 # Lift generated in the landing configuration<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    drag_landing = 0 # Drag in the landing configuration<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    sfc,thrust_engine = Propulsion.engine_analysis(mach_land,speed_of_sound,sfc_sfcref,sls_thrust,eng_type)
    engine_number = len(vehicle.Propulsors.get_TurboFans())
    thrust_landing = thrust_engine * engine_number
    lift_drag_eff = lift_landing / (thrust_landing - drag_landing)
    vel_landing = 1.25 * vel_stall
    landing_ground = 50 * lift_drag_eff + lift_drag_eff * (vel_50 ** 2 - vel_landing ** 2) / 2 / constants.grav
    mu = 0.2 # braking coefficient of friction
    
    lift_ground = 0 #lift in landing configuration on ground<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    drag_ground = 0 #drag in landing configuration on ground<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    resistance = mu * (weight_landing - lift_ground) + drag_ground
    landing_ground = vel_landing ** 2 * weight_landing / 2 / resistance / constants.grav
    lfl = (landing_air + landing_ground) / 0.6
    
    return lfl

def FuelReserve(mission,maxto,mzfw_ratio,fuel_cruise):
    """ Setting a prescribed reserve fuel value of 8% of MZFW from Pass Notes
        
        Inputs:
            mission.segment['Reserve'].fFuelRes
        
        Outputs:
            reserve_fuel
        
    """
    #AA 241 Notes Section 11.4
    if mission[0].seg_type == 'pass':
        mzfw = mzfw_ratio * maxto
        reserve_fuel = 0.08 * mzfw
    else:
        # Unpack
        frac_burn = mission.segment['Reserve'].fFuelRes
        reserve_fuel = frac_burn * fuel_cruise

    return reserve_fuel

# ----------------------------------------------------------------------
#  Utility Methods
# ----------------------------------------------------------------------

