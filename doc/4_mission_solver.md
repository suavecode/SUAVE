## Mission Solver Structure

This is a high level overview of how the mission solver functions. The purpose is to show the structure that is used for an existing mission, and show where changes should be made if different functionality is desired.

### File Structure

Mission scripts are split into two folders in the SUAVE repository. The first is in trunk/SUAVE/**Analyses/Mission**/Segments, and the second is in trunk/SUAVE/**Methods/Missions**/Segments. As with other types of analyses and methods, the distinction between these is that the Analyses folder contains classes that are built to use functions stored in the Methods folder. This division is done to make it easier to build new analysis classes using a mix of available methods. 

A typical mission segment analysis file contains four keys parts. The first specifies default user inputs, unknowns, and residuals. The inputs are used to provide the analysis with conditions that need to be met, while the unknowns and residuals are used as part of the solution process. The second sets the initialization functions for the analysis, which are run at the beginning. The third picks the convergence method and specifies the functions that will be used during iteration. The fourth finalizes the data and processes it for results output.

### Initialization

For this tutorial, we will be considering the constant speed constant altitude cruise segment. The files are available [here (Analysis)](https://github.com/suavecode/SUAVE/blob/develop/trunk/SUAVE/Analyses/Mission/Segments/Cruise/Constant_Speed_Constant_Altitude.py) and [here (Method)](https://github.com/suavecode/SUAVE/blob/develop/trunk/SUAVE/Methods/Missions/Segments/Cruise/Constant_Speed_Constant_Altitude.py). This class also inherits information from more general segment classes, which include many of the processing functions. As with other segments, the user will specify key conditions. For this case, altitude, air speed, and distance are the necessary inputs. If the user does not specify an altitude, it will be taken automatically from the last value in the previous segment. These inputs must be specified in some way for the mission segment to be evaluated. They are shown below as well:

    self.altitude  = None
    self.air_speed = 10. * Units['km/hr']
    self.distance  = 10. * Units.km

The other set of segment specific initial values are the values used for solving the segment (typically this means satisfying a force balance at every evaluation point). These can be changed by the user if needed, but the default values should perform fine for most cases. 

    self.state.unknowns.throttle   = ones_row(1) * 0.5
    self.state.unknowns.body_angle = ones_row(1) * 0.0
    self.state.residuals.forces    = ones_row(2) * 0.0

Here throttle and body angle are the unknowns, and the values shown here are the values they will start at. The residuals will be computed based on these unknowns, so their initial value is not important. Instead they are initialized just to create the necessary data structure. The ones_row line will create a numpy array with the number of elements needed for evaluation.

### Evaluation Details

Most of the missions in SUAVE, including this one, are broken into several points in time based on a Chebyshev polynomial. This causes the points to be closer together at either end of the segment. The choice of a Chebyshev polynomial (which creates cosine spacing) provides better convergence and smoothness properties versus other methods such as linear spacing.

<img src="http://suave.stanford.edu/images/drag_components_2.png" width="800" height="234" />

At each of these points the aerodynamic analysis is queried to find CL and CD, which are then converted to lift and drag. These values will be dependent on the body angle unknown and other aerodynamic parameters. Thrust is found from the vehicle's energy network, which is dependent on the throttle unknown. A weight is determined by looking at the initial weight and subsequent mass rate (typically corresponding with fuel burn). In this cruise segment, these forces are summed in 2D and the results are put in the residuals. The functions needed to arrive these forces are found in the Update Conditions section of the [Analysis file](dox_link). This section is also shown below in one of the steps to create a new mission.  

Once the evaluation process has been performed at all points, the unknowns and residuals are fed back to the solve routine, which in this case is scipy's fsolve. The file that performs this process is [here](https://github.com/suavecode/SUAVE/blob/develop/trunk/SUAVE/Methods/Missions/Segments/converge_root.py). This routine continues evaluating the points until convergence is reached. Once this happens, post processing is done to put the data in the results output.

### Using Multiple Segments

Multiple segments can be run sequentially by appending them in the desired order. Examples of this are in all the tutorial files that have an aircraft fly a full mission. In addition, the full mission can be run simultaneously will all segment constraints used together. If you are interested in doing something like this, please ask us about it on our [forum](/forum).

#### Process Summary

Mission Setup

* Initializes default values for unknowns
* Initializes set of functions used to determine residuals
* Reads user input for segment parameters
* Adds the analysis group to be used (including the vehicle and items like atmosphere)
* Appends segments in order

Evaluate

* Varies unknowns until residual convergence is reached using scipy's fsolve
* Repeats process for each segment until full mission is complete

### Adding New Mission Segments

The segment described above uses two unknowns to solve force residuals in two dimensions. This general setup works well for many problems of interest, but SUAVE is designed to accommodate other mission analysis types as well. A user may want to add control surface deflection and solve for moments as well, or look at forces in all three dimensions. 

In addition, a user may want to modify how the mission is flown, as is done with the many other segments currently available. They may want to modify how the mission is solved, such as is done in our single point evaluation segments where finite differencing is not relevant.

Here we will explain the process of modifying our constant speed constant rate climb segment to be constant throttle constant speed. This still uses 2D force balance but changes the profile. There are four functions that are modified here. The first is shown below. The functions can be found in [here]() and [here]()

    def initialize_conditions(segment,state):
    
	    # unpack
	    climb_rate = segment.climb_rate
	    air_speed  = segment.air_speed   
	    alt0       = segment.altitude_start 
	    altf       = segment.altitude_end
	    t_nondim   = state.numerics.dimensionless.control_points
	    conditions = state.conditions  
	
	    # check for initial altitude
	    if alt0 is None:
	        if not state.initials: raise AttributeError('initial altitude not set')
	        alt0 = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]
	
	    # discretize on altitude
	    alt = t_nondim * (altf-alt0) + alt0
	    
	    # process velocity vector
	    v_mag = air_speed
	    v_z   = -climb_rate # z points down
	    v_x   = np.sqrt( v_mag**2 - v_z**2 )
	    
	    # pack conditions    
	    conditions.frames.inertial.velocity_vector[:,0] = v_x
	    conditions.frames.inertial.velocity_vector[:,2] = v_z
	    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
	    conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude in this context

This function initializes speed and altitude based on the given climb rate, airspeed, and altitude end points. t_nondim gives nondimensional time in cosine spacing from 0 to 1 in order to pick the values at the points to be evaluated. Unfortunately, when we use constant throttle we cannot know beforehand exactly how altitude (or climb rate in this case) will vary with time, so altitude cannot be spaced with this method. Instead a different function is used to initialize conditions:

    def initialize_conditions(segment,state):
    
	    # unpack
	    throttle   = segment.throttle
	    air_speed  = segment.air_speed   
	    alt0       = segment.altitude_start 
	    altf       = segment.altitude_end
	    t_nondim   = state.numerics.dimensionless.control_points
	    conditions = state.conditions  
	
	    # check for initial altitude
	    if alt0 is None:
	        if not state.initials: raise AttributeError('initial altitude not set')
	        alt0 = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]
	
	    # pack conditions  
	    conditions.propulsion.throttle[:,0] = throttle
	    conditions.frames.inertial.velocity_vector[:,0] = air_speed # start up value

Here only the throttle and air speed are loaded in, and discretization of other values will need to occur later so that it is part of the iteration loop. This requires a new function that updates the altitude differentials.

    def update_differentials_altitude(segment,state):

	    # unpack
	    t = state.numerics.dimensionless.control_points
	    D = state.numerics.dimensionless.differentiate
	    I = state.numerics.dimensionless.integrate
	
	    
	    # Unpack segment initials
	    alt0       = segment.altitude_start 
	    altf       = segment.altitude_end    
	    conditions = state.conditions  
	
	    r = state.conditions.frames.inertial.position_vector
	    v = state.conditions.frames.inertial.velocity_vector
	    
	    # check for initial altitude
	    if alt0 is None:
	        if not state.initials: raise AttributeError('initial altitude not set')
	        alt0 = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]    
	    
	    # get overall time step
	    vz = -v[:,2,None] # Inertial velocity is z down
	    dz = altf- alt0    
	    dt = dz / np.dot(I[-1,:],vz)[-1] # maintain column array
	    
	    # Integrate vz to get altitudes
	    alt = alt0 + np.dot(I*dt,vz)
	
	    # rescale operators
	    t = t * dt
	
	    # pack
	    t_initial = state.conditions.frames.inertial.time[0,0]
	    state.conditions.frames.inertial.time[:,0] = t_initial + t[:,0]
	    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
	    conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude in this context    
	
	    return

In this function, t, D, and I are numpy arrays that allow approximate differentiation and integration. Since the total time is not known without determining the climb rate, we must first determine the time required to reach the final altitude. The line `dt = dz / np.dot(I[-1,:],vz)[-1]` does this with the integrator providing the amount of altitude gained if the velocities were spread across just one second instead of the full segment time. This gives the scaling quantity `dt` that is then used to get the altitude at every point in `alt = alt0 + np.dot(I*dt,vz)`. The values for altitude are then are then packed for use in other functions.

The above allows us to deal with discretization without a known profile, but we also must calculate the velocity in order to use this. This is done with another added function.

    def update_velocity_vector_from_wind_angle(segment,state):
    
	    # unpack
	    conditions = state.conditions 
	    v_mag  = segment.air_speed 
	    alpha  = state.unknowns.wind_angle[:,0][:,None]
	    theta  = state.unknowns.body_angle[:,0][:,None]
	    
	    # Flight path angle
	    gamma = theta-alpha
	    
	    # process
	    v_x =  v_mag * np.cos(gamma)
	    v_z = -v_mag * np.sin(gamma) # z points down
	    
	    # pack
	    conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0]
	    conditions.frames.inertial.velocity_vector[:,2] = v_z[:,0]
	    
	    return conditions

This uses our new set of unknowns to determine the velocities. 

Additionally, since the unknowns are different we must change the function that unpacks them. Wind angle does not need to be stored so it is not included here.

    def unpack_body_angle(segment,state):
    
	    # unpack unknowns
	    theta  = state.unknowns.body_angle
	    
	    # apply unknowns
	    state.conditions.frames.body.inertial_rotations[:,1] = theta[:,0]

We now add these functions to the segment process list.

        # --------------------------------------------------------------
        #   Initialize - before iteration
        # --------------------------------------------------------------
        initialize = self.process.initialize
        
        initialize.expand_state            = Methods.expand_state
        initialize.differentials           = Methods.Common.Numerics.initialize_differentials_dimensionless
        initialize.conditions              = Methods.Climb.Constant_Throttle_Constant_Speed.initialize_conditions
        initialize.velocities              = Methods.Climb.Constant_Throttle_Constant_Speed.update_velocity_vector_from_wind_angle
        initialize.differentials_altitude  = Methods.Climb.Constant_Throttle_Constant_Speed.update_differentials_altitude      

and

        # Unpack Unknowns
        iterate.unknowns = Process()
        iterate.unknowns.mission           = Methods.Climb.Constant_Throttle_Constant_Speed.unpack_body_angle 
        
        # Update Conditions
        iterate.conditions = Process()
        iterate.conditions.velocities      = Methods.Climb.Constant_Throttle_Constant_Speed.update_velocity_vector_from_wind_angle
        iterate.conditions.differentials_a = Methods.Climb.Constant_Throttle_Constant_Speed.update_differentials_altitude
        iterate.conditions.differentials_b = Methods.Common.Numerics.update_differentials_time
        iterate.conditions.acceleration    = Methods.Common.Frames.update_acceleration
        iterate.conditions.altitude        = Methods.Common.Aerodynamics.update_altitude
        iterate.conditions.atmosphere      = Methods.Common.Aerodynamics.update_atmosphere
        iterate.conditions.gravity         = Methods.Common.Weights.update_gravity
        iterate.conditions.freestream      = Methods.Common.Aerodynamics.update_freestream
        iterate.conditions.orientations    = Methods.Common.Frames.update_orientations
        iterate.conditions.aerodynamics    = Methods.Common.Aerodynamics.update_aerodynamics
        iterate.conditions.stability       = Methods.Common.Aerodynamics.update_stability
        iterate.conditions.propulsion      = Methods.Common.Energy.update_thrust
        iterate.conditions.weights         = Methods.Common.Weights.update_weights
        iterate.conditions.forces          = Methods.Common.Frames.update_forces
        iterate.conditions.planet_position = Methods.Common.Frames.update_planet_position

If you have any questions that are not answered in other tutorials or the FAQ please ask on our [forum](/forum) page. This is also the place to go if you want help building a more elaborate evaluation, such as one that includes moments.