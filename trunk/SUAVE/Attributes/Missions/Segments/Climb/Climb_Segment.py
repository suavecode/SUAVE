
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np
import time

# SUAVE imports
from SUAVE.Attributes.Missions.Segments import Aerodynamic_Segment

# import units
from SUAVE.Attributes import Units
km = Units.km
hr = Units.hr

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Climb_Segment(Aerodynamic_Segment):

    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  
    
    def __defaults__(self):
        self.tag = 'Base Climb Segment'
        
       # --- User Inputs
        
        self.altitude_start = None      # optional
        self.altitude_end   = 10. * km
        self.battery_energy = 0.0
        self.latitude       = 0.0
        self.longitude      = 0.0        
       
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
    
    def initialize_conditions(self,conditions,numerics,initials=None):
        """ Segment.initialize_conditions(conditions,numerics,initials=None)
            update the segment conditions
            pin down as many condition variables as possible in this function
            
            Inputs:
                conditions - the conditions data dictionary, with initialized zero arrays, 
                             with number of rows = segment.conditions.n_control_points
                initials - a data dictionary with 1-row column arrays pulled from the last row
                           of the previous segment's conditions data, or none if no previous segment
                
            Outputs:
                conditions - the conditions data dictionary, updated with the 
                             values that can be precalculated
            
            Assumptions:
                --
                
            Usage Notes:
                may need to inspect segment (self) for user inputs
                will be called before solving the segments free unknowns
                
        """
        
        # gets initial mass and time from previous segment
        conditions = Aerodynamic_Segment.initialize_conditions(self,conditions,numerics,initials)
        
        # unpack inputs
        alt0     = self.altitude_start 
        altf     = self.altitude_end
        atmo     = self.analyses.atmosphere
        planet   = self.analyses.planet
        t_nondim = numerics.dimensionless_time
        
        # check for initial altitude
        if alt0 is None:
            if not initials: raise AttributeError('initial altitude not set')
            alt0 = -1.0 * initials.frames.inertial.position_vector[0,2]
            self.altitude_start = alt0
        
        # discretize on altitude
        alt = t_nondim * (altf-alt0) + alt0
        
        # pack conditions
        conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
        conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude in this context
        
        # freestream atmosphereric conditions
        conditions = self.compute_atmosphere(conditions,atmo)
        conditions = self.compute_gravity(conditions,planet)
        
        # done
        return conditions
    

    # ------------------------------------------------------------------
    #   Methods For Iterations
    # ------------------------------------------------------------------  

    def update_velocity_vector(self,unknowns,conditions):
        """ helper function to set velocity_vector
            called from update_differentials()
        """
        
        # conditions.frames.inertial.velocity_vector[:,2] = ?
        
        return conditions

    def update_differentials(self,conditions,numerics,unknowns):
        """ Segment.update_differentials(conditions, numerics, unknowns)
            updates the differential operators t, D and I
            must return in dimensional time, with t[0] = 0
            
            Works with a segment discritized in vertical position, altitude
            
            Inputs - 
                unknowns      - data dictionary of segment free unknowns
                conditions    - data dictionary of segment conditions
                numerics - data dictionary of non-dimensional differential operators
                
            Outputs - 
                numerics - udpated data dictionary with dimensional numerics 
            
            Assumptions - 
                outputed operators are in dimensional time for the current solver iteration
                works with a segment discritized in vertical position, altitude
                
        """
        
        # update the clib rate
        conditions = self.update_velocity_vector(unknowns,conditions)
        
        # unpack
        t = numerics.dimensionless_time
        D = numerics.differentiate_dimensionless
        I = numerics.integrate_dimensionless
        
        r = conditions.frames.inertial.position_vector
        v = conditions.frames.inertial.velocity_vector
        
        dz = r[-1,2] - r[0,2]
        vz = v[:,2,None] # maintain column array
        
        # get overall time step
        dt = np.dot( I[-1,:] * dz , 1/ vz[:,0] )
        
        # rescale operators
        D = D / dt
        I = I * dt
        t = t * dt
        
        # pack
        numerics.time = t
        numerics.differentiate_time = D
        numerics.integrate_time = I

        # time
        t_initial = conditions.frames.inertial.time[0,0]
        conditions.frames.inertial.time[:,0] = t_initial + t[:,0]
        
        return numerics

    def post_process(self,conditions,numerics,unknowns):
        
        x0 = conditions.frames.inertial.position_vector[0,0]
        vx = conditions.frames.inertial.velocity_vector[:,0]
        I  = numerics.integrate_time
        
        x = np.dot(I,vx) + x0
        
        conditions.frames.inertial.position_vector[:,0] = x
        
        conditions = Aerodynamic_Segment.post_process(self,conditions,numerics,unknowns)
        
        return conditions
        