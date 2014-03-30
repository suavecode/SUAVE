""" Segment.py: Class for storing mission segment data """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Structure                                import Data, Data_Exception
from SUAVE.Structure                                import Container as ContainerBase
from SUAVE.Attributes.Planets                       import Planet
from SUAVE.Attributes.Atmospheres                   import Atmosphere


# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

class Segment(Data):
    """ Top-level Mission Segment """
    def __defaults__(self):
        
        self.tag = 'Segment'
        self.dofs = 2
        
        self.planet     = Planet()
        self.atmosphere = Atmosphere()
        
        self.complex  = True
        self.jacobian = "complex"

        # numerical options
        self.options = Data()
        self.options.tag = 'Solution Options'
        self.options.tolerance_solution            = 1e-8
        self.options.tolerance_boundary_conditions = 1e-8
        self.options.n_control_points              = 16
        
        # base matricies
        ones_1col = np.zeros([1,1])
        ones_2col = np.zeros([1,2])
        ones_3col = np.zeros([1,3])
        
        # --- Conditions
        
        # user shouldn't touch these on input script
        # they will be shared with analysis modules and naming is important
        
        # setup conditions
        conditions = Data()
        conditions.freestream   = Data()
        conditions.inertial     = Data()
        conditions.body         = Data()
        conditions.weights      = Data()
        conditions.aerodynamics = Data()
        conditions.engergy      = Data()
        
        # freestream conditions
        conditions.freestream.velocity_vector    = ones_3col
        conditions.freestream.mach_number        = ones_1col
        conditions.freestream.angle_of_attack    = ones_1col
        conditions.freestream.pressure           = ones_1col
        conditions.freestream.temperature        = ones_1col
        conditions.freestream.density            = ones_1col
        conditions.freestream.speed_of_sound     = ones_1col
        conditions.freestream.viscosity          = ones_1col
        conditions.freestream.reynolds_number    = ones_1col
        conditions.freestream.dynamic_pressure   = ones_1col        
        
        # inertial conditions
        conditions.inertial.position_vector      = ones_3col
        conditions.inertial.velocity_vector      = ones_3col
        conditions.inertial.acceleration_vector  = ones_3col
        conditions.inertial.total_force_vector   = ones_3col      
        conditions.inertial.time                 = ones_1col
        
        # body conditions
        conditions.body.velocity_vector          = ones_3col
        conditions.body.acceleration_vector      = ones_3col
        conditions.body.lift_force_vector        = ones_3col
        conditions.body.drag_force_vector        = ones_3col
        conditions.body.thrust_force_vector      = ones_3col          
        
        # weights conditions
        conditions.weights.total_mass            = ones_1col
        conditions.weights.breakdown             = Data()
        
        # aerodynamics conditions
        conditions.aerodynamics.lift_coefficient = ones_1col
        conditions.aerodynamics.drag_coefficient = ones_1col
        conditions.aerodynamics.lift_coefficient = ones_1col
        conditions.aerodynamics.lift_breakdown   = Data()
        conditions.aerodynamics.drag_breakdown   = Data()
        conditions.aerodynamics.thrust_breakdown = Data()
        
        # energy conditions
        conditions.energy.gravity                = ones_1col

        # efficiencies 
        self.efficiency = Data()
        

    def initialize_vectors(self):

        N = self.options.N
        zips = np.zeros((N,3))
        if self.complex: zips = zips + 0j
        
        # initialize arrays
        self.vectors.r    = zips + 0  # position vector (m)
        self.vectors.V    = zips + 0  # velocity vector (m/s)
        self.vectors.D    = zips + 0  # drag vector (N)
        self.vectors.L    = zips + 0  # lift vector (N)
        self.vectors.v    = zips + 0  # velocity unit tangent vector
        self.vectors.l    = zips + 0  # lift unit tangent vector
        self.vectors.u    = zips + 0  # thrust unit tangent vector
        self.vectors.F    = zips + 0  # thrust vector (N)
        self.vectors.Ftot = zips + 0  # total force vector (N)
        self.vectors.a    = zips + 0  # acceleration (m/s^2)
        self.alpha        = zips[:,0] + 0  # alpha (rad)

        return


# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------

class Container(ContainerBase):
    pass

Segment.Container = Container
