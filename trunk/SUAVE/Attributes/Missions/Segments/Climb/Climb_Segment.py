
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

class Climb_Segment(Aerodynamic_Segment):

    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  
    
    def __defaults__(self):
        self.tag = 'Base Climb Segment'
        
       # --- User Inputs
        
        self.altitude_start = 1. * km
        self.altitude_end   = 10. * km
        
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
        conditions = Aerodynamic_Segment.initialize_conditions(self,conditions,differentials,initials)        
        
        # unpack inputs
        alt0     = self.altitude_start
        altf     = self.altitude_end
        atmo     = self.atmosphere
        planet   = self.planet
        t_nondim = differentials.t
        
        # discretize on altitude
        alt = t_nondim * (altf-alt0) + alt0
        
        # pack conditions
        conditions.frames.inertial.position_vector[:,2] = alt[:,0]
        conditions.freestream.altitude[:,0]             = alt[:,0]
        
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

    def update_differentials(self,unknowns,conditions,differentials):
        """ Segment.update_differentials(unknowns, conditions, differentials)
            updates the differential operators t, D and I
            must return in dimensional time, with t[0] = 0
            
            Works with a segment discritized in vertical position, altitude
            
            Inputs - 
                unknowns      - data dictionary of segment free unknowns
                conditions    - data dictionary of segment conditions
                differentials - data dictionary of non-dimensional differential operators
                
            Outputs - 
                differentials - udpated data dictionary with dimensional differentials 
            
            Assumptions - 
                outputed operators are in dimensional time for the current solver iteration
                works with a segment discritized in vertical position, altitude
                
        """
        
        # update the clib rate
        conditions = self.update_velocity_vector(unknowns,conditions)
        
        # unpack
        t = differentials.t
        D = differentials.D
        I = differentials.I
        
        r = conditions.frames.inertial.position_vector
        v = conditions.frames.inertial.velocity_vector
        
        dz = r[-1,2] - r[0,2]
        vz = v[:,2,None] # maintain column array
        
        # rescale operators
        D = D / dz * vz
        I = I * dz / vz
        t = np.dot(I,t)
        
        # pack
        differentials.t = t
        differentials.D = D
        differentials.I = I

        # time
        t_initial = conditions.frames.inertial.time[0,0]
        conditions.frames.inertial.time[:,0] = t_initial + t[:,0]
        
        return differentials
