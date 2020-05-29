## @ingroup Analyses-Mission-Segments-Hover
# Takeoff.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from .Ground_Constant_Time import Ground_Constant_Time
from SUAVE.Methods.Missions import Segments as Methods

# Units
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Hover
class Rotation(Ground_Constant_Time):
    """ Segment for takeoff. Integrates equations of motion
        including rolling friction.
        
        Assumptions:
        Notes Regarding Friction Coefficients
        Dry asphalt or concrete: .04 brakes off, .4 brakes on
        Wet asphalt or concrete: .05 brakes off, .225 brakes on
        Icy asphalt or concrete: .02 brakes off, .08 brakes on
        Hard turf:               .05 brakes off, .4 brakes on
        Firm dirt:               .04 brakes off, .3 brakes on
        Soft turf:               .07 brakes off, .2 brakes on
        Wet grass:               .08 brakes off, .2 brakes on
        
        Source: General Aviation Aircraft Design: Applied Methods and Procedures,
        by Snorri Gudmundsson, copyright 2014, published by Elsevier, Waltham,
        MA, USA [p.938]
    """    

    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  

    def __defaults__(self):
        """ This sets the default solver flow. Anything in here can be modified after initializing a segment.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """         

        self.velocity_start       = 1.0
        self.time                 = 1.0
        self.friction_coefficient = 0.04
        self.throttle             = 1.0
        self.final_body_angle     = 10. * Units.deg
        
        # initials and unknowns
        ones_row = self.state.ones_row
        self.state.unknowns.velocity_x            = ones_row(1) * 0.0
        self.state.residuals.forces               = ones_row(1) * 0.0        

        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
    
        initialize = self.process.initialize
        initialize.conditions = Methods.Ground.Rotation.initialize_conditions
        
        return


