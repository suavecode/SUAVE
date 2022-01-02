## @ingroup Components-Constraints
# Costs.py
#
# Created:
# Modified: Feb 2016, T. MacDonald
# Modified: Feb 2016, T. Orra

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Components import Component
from SUAVE.Core import Data, Units
import numpy as np


# ----------------------------------------------------------------------
# Industrial Costs class
# ----------------------------------------------------------------------
## @ingroup Components-Costs
class Constraints(Data):
    """A class containing constraint analysis variables.
    
    Assumptions:
    None
    
    Source:
    N/A
    """     
    def __defaults__(self):
        """This sets the default values used in the industrial cost methods.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """  
        # inputs
        self.tag                = 'Constraint analysis'
        self.plot_tag           = False


        # Defines default constraint analyses
        self.analyses = Data()
        self.analyses.takeoff    = True
        self.analyses.cruise     = True
        self.analyses.max_cruise = False
        self.analyses.landing    = True
        self.analyses.OEI_climb  = True
        self.analyses.turn       = False
        self.analyses.climb      = False
        self.analyses.ceiling    = False
        
        self.wing_loading      = np.arange(2, 200, 0.25) * Units['force_pound/foot**2']
        self.design_point_type = None

        # Default parameters for the constraint analysis
        # take-off
        self.takeoff = Data()
        self.takeoff.runway_elevation     = 0.0
        self.takeoff.ground_run           = 0.0
        self.takeoff.rolling_resistance   = 0.05
        self.takeoff.liftoff_speed_factor = 1.1
        self.takeoff.delta_ISA            = 0.0
        # climb
        self.climb = Data()
        self.climb.altitude   = 0.0
        self.climb.airspeed   = 0.0
        self.climb.climb_rate = 0.0
        self.climb.delta_ISA  = 0.0
        # OEI climb
        self.OEI_climb = Data()
        self.OEI_climb.climb_speed_factor = 1.2
        # cruise
        self.cruise = Data()
        self.cruise.altitude        = 0.0
        self.cruise.delta_ISA       = 0.0
        self.cruise.airspeed        = 0.0
        self.cruise.thrust_fraction = 0.0
        # max cruise
        self.max_cruise = Data()
        self.max_cruise.altitude        = 0.0
        self.max_cruise.delta_ISA       = 0.0
        self.max_cruise.mach            = 0.0
        self.max_cruise.thrust_fraction = 0.0
        # turn
        self.turn = Data()
        self.turn.angle           = 0.0
        self.turn.altitude        = 0.0
        self.turn.delta_ISA       = 0.0
        self.turn.airspeed        = 0.0
        self.turn.specific_energy = 0.0
        # ceiling
        self.ceiling = Data()
        self.ceiling.altitude  = 0.0
        self.ceiling.delta_ISA = 0.0
        self.ceiling.airspeed  = 0.0
        # landing
        self.landing = Data()
        self.landing.ground_roll           = 0.0
        self.landing.approach_speed_factor = 1.23
        self.landing.runway_elevation      = 0.0
        self.landing.delta_ISA             = 0.0 

        # Default aircraft properties
        # geometry
        self.geometry = Data()
        self.geometry.aspect_ratio                  = 0.0 
        self.geometry.taper                         = 0.0
        self.geometry.thickness_to_chord            = 0.0
        self.geometry.sweep_quarter_chord           = 0.0
        self.geometry.high_lift_configuration_type  = None
        # engine
        self.engine = Data()
        self.engine.type                    = None
        self.engine.number                  = 0
        self.engine.bypass_ratio            = 0.0 
        self.engine.throttle_ratio          = 1.0   
        self.engine.afterburner             = False
        self.engine.method                  = 'Mattingly'
        # propeller
        self.propeller = Data()
        self.propeller.takeoff_efficiency   = 0.0
        self.propeller.climb_efficiency     = 0.0
        self.propeller.cruise_efficiency    = 0.0
        self.propeller.turn_efficiency      = 0.0
        self.propeller.ceiling_efficiency   = 0.0
        self.propeller.OEI_climb_efficiency = 0.0

        # Define aerodynamics
        self.aerodynamics = Data()
        self.aerodynamics.oswald_factor   = 0.0
        self.aerodynamics.cd_takeoff      = 0.0   
        self.aerodynamics.cl_takeoff      = 0.0   
        self.aerodynamics.cl_max_takeoff  = 0.0
        self.aerodynamics.cl_max_landing  = 0.0  
        self.aerodynamics.cd_min_clean    = 0.0
        self.aerodynamics.fuselage_factor = 0.974
        self.aerodynamics.viscous_factor  = 0.38

        return