# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Structure                    import Data, Data_Exception
from SUAVE.Structure                    import Container as ContainerBase
from SUAVE.Attributes.Planets           import Planet
from SUAVE.Attributes.Atmospheres       import Atmosphere
from SUAVE.Methods.Utilities.Chebyshev  import chebyshev_data

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

class Aerodynamic_Segment(Data):
    
    def __defaults__(self):
        self.tag = 'Aerodynamic Segment'
        
        # presumably we fly in an atmosphere
        self.planet     = Planet()
        self.atmosphere = Atmosphere()        


        # --- Conditions and Unknowns
        
        # user shouldn't change these in an input script
        # only used for processing / post processing
        # they will be shared with analysis modules and meaningful naming is important
        
        # base matricies
        # use a trivial operation to copy the array
        ones_1col = np.zeros([1,1])
        ones_2col = np.zeros([1,2])
        ones_3col = np.zeros([1,3])        
        
        
        # --- Conditions        
        
        # setup conditions
        conditions = Data()
        conditions.inertial     = Data()
        conditions.body         = Data()        
        conditions.freestream   = Data()
        conditions.aerodynamics = Data()
        conditions.weights      = Data()
        conditions.engergy      = Data()
        
        # inertial conditions
        conditions.inertial.position_vector      = ones_3col + 0
        conditions.inertial.velocity_vector      = ones_3col + 0
        conditions.inertial.acceleration_vector  = ones_3col + 0
        conditions.inertial.total_force_vector   = ones_3col + 0      
        conditions.inertial.time                 = ones_1col + 0
        
        # body conditions
        conditions.body.velocity_vector          = ones_3col + 0
        conditions.body.acceleration_vector      = ones_3col + 0
        conditions.body.total_force_vector       = ones_3col + 0
        conditions.body.lift_force_vector        = ones_3col + 0
        conditions.body.drag_force_vector        = ones_3col + 0
        conditions.body.thrust_force_vector      = ones_3col + 0          
        
        # freestream conditions
        conditions.freestream.velocity_vector    = ones_3col + 0
        conditions.freestream.mach_number        = ones_1col + 0
        conditions.freestream.angle_of_attack    = ones_1col + 0
        conditions.freestream.pressure           = ones_1col + 0
        conditions.freestream.temperature        = ones_1col + 0
        conditions.freestream.density            = ones_1col + 0
        conditions.freestream.speed_of_sound     = ones_1col + 0
        conditions.freestream.viscosity          = ones_1col + 0
        conditions.freestream.reynolds_number    = ones_1col + 0
        conditions.freestream.dynamic_pressure   = ones_1col + 0  
        
        # aerodynamics conditions
        conditions.aerodynamics.lift_coefficient = ones_1col + 0
        conditions.aerodynamics.drag_coefficient = ones_1col + 0
        conditions.aerodynamics.lift_coefficient = ones_1col + 0
        conditions.aerodynamics.lift_breakdown   = Data()
        conditions.aerodynamics.drag_breakdown   = Data()
        conditions.aerodynamics.thrust_breakdown = Data()
        
        # weights conditions
        conditions.weights.total_mass            = ones_1col + 0
        conditions.weights.breakdown             = Data()
        
        # energy conditions
        conditions.energies.total_energy         = ones_1col + 0
        conditions.energies.total_efficiency     = ones_1col + 0
        conditions.energies.gravity_energy       = ones_1col + 0
        
        
        # --- Unknowns        
        
        # setup unknowns
        unknowns = Data()
        unknowns.states   = Data()
        unknowns.controls = Data()
        unknowns.finals   = Data()
        
        # an example
        ## unknowns.states.gamma = ones_1col + 0
        
        
        # --- Numerics
        
        # numerical options
        self.options = Data()
        self.options.tag = 'Solution Options'
        self.options.n_control_points              = 16
        self.options.jacobian                      = "complex"
        self.options.tolerance_solution            = 1e-8
        self.options.tolerance_boundary_conditions = 1e-8        
        
        # differentials
        self.differentials = Data()
        self.differentials.t = t
        self.differentials.D = D
        self.differentials.I = I
        self.differentials.method = chebyshev_data
        
        return
      

# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------

class Container(ContainerBase):
    pass

Segment.Container = Container
