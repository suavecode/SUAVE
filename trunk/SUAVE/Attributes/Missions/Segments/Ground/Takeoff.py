
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np

# SUAVE imports
from Ground_Segment import Ground_Segment
from SUAVE.Structure import Data

# import units
from SUAVE.Attributes import Units
km = Units.km
hr = Units.hr
deg = Units.deg

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Takeoff(Ground_Segment):

    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  

    def __defaults__(self):

        self.tag = "Takeoff Segment"

        self.velocity_start       = 0.0
        self.velocity_end         = 150 * Units.knots
        self.friction_coefficient = 0.04
        self.throttle             = 1.0
        self.battery_energy = 0.0
        self.latitude       = 0.0
        self.longitude      = 0.0        

        return


    def initialize_conditions(self,conditions,numerics,initials=None):

        conditions = Ground_Segment.initialize_conditions(self,conditions,numerics,initials)

        # default initial time, position, and mass
        t_initial = 0.0
        r_initial = conditions.frames.inertial.position_vector[0,:][None,:]
        m_initial = self.config.mass_properties.takeoff

        # apply initials
        conditions.weights.total_mass[:,0]   = m_initial
        conditions.frames.inertial.time[:,0] = t_initial
        conditions.frames.inertial.position_vector[:,:] = r_initial[:,:]

        throttle = self.throttle	
        conditions.propulsion.throttle[:,0] = throttle

        return conditions



    # ------------------------------------------------------------------
    #   Methods For Post-Solver
    # ------------------------------------------------------------------    

    def post_process(self,conditions,numerics,unknowns):
        """ Segment.post_process(conditions,numerics,unknowns)
            post processes the conditions after converging the segment solver.
            Packs up the estimated distance for rotation in addition to the final 
            position vector found in the superclass post_process method.
        """

        conditions = Ground_Segment.post_process(self, conditions, numerics, unknowns)

        # process
        # Assume 3.5 seconds for rotation, with a constant groundspeed
        rotation_distance = conditions.frames.inertial.velocity_vector[-1,0] * 3.5

        # pack outputs
        conditions.frames.inertial.rotation_distance = np.ones([1,1])
        conditions.frames.inertial.rotation_distance[0,0] = rotation_distance

        return conditions    