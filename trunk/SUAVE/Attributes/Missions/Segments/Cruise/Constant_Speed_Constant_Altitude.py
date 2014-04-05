
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np

# SUAVE imports
from SUAVE.Attributes.Missions.Segments import Aerodynamic_Segment

# import units
from SUAVE.Attributes import Units
km = Units.km
hr = Units.hr

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Constant_Speed_Constant_Altitude(Aerodynamic_Segment):

    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  
    
    def __defaults__(self):
        self.tag = 'Constant Speed, Constant Altitude Cruise'
        
       # --- User Inputs
        
        self.altitude = 10. * km
        self.speed    = 10. * km/hr
        self.range    = 10. * km

        
        # -- Conditions 
        
        # uses a ton of defaults from Aerodynamic_Segment
        
        
        # --- Unknowns
        unknowns = self.unknowns
        unknowns.controls.throttle = np.ones([1,1])
        unknowns.controls.theta    = np.ones([1,1])
        
        # --- Residuals
        residuals = self.residuals
        residuals.controls.Fx = np.ones([1,1])
        residuals.controls.Fz = np.ones([1,1])
        
        return

    # ------------------------------------------------------------------
    #   Methods For Initialization
    # ------------------------------------------------------------------  
    
    def check_inputs(self):
        """ Segment.check():
            error checking of segment inputs
        """
        
        ## CODE
        
        return
    
    def initialize_conditions(self,conditions,differentials,initials=None):
        """ Segment.initialize_conditions(conditions)
            update the segment conditions
            pin down as many condition variables as possible in this function
            
            Assumptions:
                --
                
        """
        
        # unpack inputs
        N        = self.options.n_control_points
        alt      = self.altitude
        xf       = self.range
        speed    = self.speed
        atmo     = self.atmosphere
        planet   = self.planet
        t_nondim = differentials.t
        
        if initials:
            t_initial = initials.frames.inertial[0,0]
            m_initial = initials.weights.total_mass[0,0]
        else:
            t_initial = 0.0
            m_initial = self.config.Mass_Props.m_takeoff
        
        # freestream details
        conditions.freestream.altitude[:,0] = alt
        conditions = self.compute_atmosphere(conditions,atmo)
        conditions = self.compute_gravity(conditions,planet)
        
        conditions.frames.inertial.velocity_vector[:,0] = speed
        conditions = self.compute_freestream(conditions)
        
        # weights
        conditions.weights.total_mass[:,0] = m_initial
        
        # dimensionalize time
        t_final = xf / speed + t_initial
        time =  t_nondim * (t_final-t_initial) + t_initial
        conditions.frames.inertial.time[:,0] = time[:,0]
        
        # done
        return conditions
    
    
    # ------------------------------------------------------------------
    #   Methods For Solver Iterations
    # ------------------------------------------------------------------    
    
    def update_conditions(self,unknowns,conditions,differentials):
        """ Segment.update_conditions(unknowns, conditions, differentials)
            if needed, updates the conditions given the current free unknowns and differentials
            called once per segment solver iteration
        """
        
        # unpack models
        aero_model = self.config.aerodynamics_model
        prop_model = self.config.propulsion_model
        
        # unpack unknowns
        throttle = unknowns.controls.throttle
        theta    = unknowns.controls.theta
        
        # apply unknowns
        conditions.propulsion.throttle[:,0]            = throttle[:,0]
        conditions.frames.body.inertial_rotations[:,1] = theta[:,0]
        
        # angle of attacks
        conditions = self.compute_orientations(conditions)
        
        # aerodynamics
        conditions = self.compute_aerodynamics(aero_model,conditions)
        
        # propulsion
        conditions = self.compute_propulsion(prop_model,conditions)
        
        # weights
        conditions = self.compute_weights(conditions,differentials)
        
        # total forces
        conditions = self.compute_forces(conditions)
        
        
        return conditions
    
    def solve_residuals(self,unknowns,residuals,conditions,differentials):
        """ Segment.solve_residuals(unknowns, conditions, differentials)
            the hard work, solves the residuals for the free unknowns
            called once per segment solver iteration

            Usage Notes - 
                after this method, residuals composed into a final residual vector:
                    R = [ [ d(unknowns.states)/dt - residuals.states   ] ;
                          [                         residuals.controls ] ;
                          [                         residuals.finals   ] ] = [0] ,
                    where the segment solver will find a root of R = [0]

        """
        
        # unpack inputs
        FT = conditions.frames.inertial.total_force_vector
        
        # process
        residuals.controls.Fx[:,0] = FT[:,0]
        residuals.controls.Fz[:,0] = FT[:,2]
        
        # pack outputs
        ## CODE
        
        return residuals
    
    # ------------------------------------------------------------------
    #   Methods For Post-Solver
    # ------------------------------------------------------------------    
    
    def post_process(self,unknowns,conditions,differentials):
        """ Segment.post_process(unknowns, conditions, differentials)
            post processes the conditions after converging the segment solver
            
            Usage Notes - 
                use this to store the unknowns and any other interesting in conditions
                    for later plotting
                for clarity and style, be sure to define any new fields in segment.__defaults__
            
        """
        
        # unpack inputs
        ## CODE
        
        # setup
        ## CODE
        
        # process
        ## CODE
        
        # pack outputs
        ## CODE
        
        return
    
    
        
    