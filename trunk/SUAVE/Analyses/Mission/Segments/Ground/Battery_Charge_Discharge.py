## @ingroup Analyses-Mission-Segments-Ground
# Battery_Charge_Discharge.py
#
# Created: Apr 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Analyses.Mission.Segments import Aerodynamic
from SUAVE.Methods.Missions import Segments as Methods
from SUAVE.Analyses.Mission.Segments import Conditions 
from SUAVE.Methods.skip import skip 

from SUAVE.Analyses import Process
# Units
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Ground
class Battery_Charge_Discharge(Aerodynamic): 

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
        
        # --------------------------------------------------------------
        #   User inputs
        # --------------------------------------------------------------
        self.altitude               = None
        self.time                   = 1.0 * Units.seconds
        self.overcharge_contingency = 1.25
        self.battery_discharge      = True  
        
        # --------------------------------------------------------------
        #   State
        # --------------------------------------------------------------
    
        # conditions
        self.state.conditions.update( Conditions.Aerodynamics() )
    
        # initials and unknowns
        ones_row = self.state.ones_row
    
    
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
    
        # --------------------------------------------------------------
        #   Initialize - before iteration
        # --------------------------------------------------------------
        initialize = self.process.initialize
    
        initialize.expand_state            = Methods.expand_state
        initialize.differentials           = Methods.Common.Numerics.initialize_differentials_dimensionless
        initialize.conditions              = Methods.Ground.Battery_Charge_Discharge.initialize_conditions
      
        # --------------------------------------------------------------
        #   Converge - starts iteration
        # --------------------------------------------------------------
        converge = self.process.converge
        
        converge.converge_root             = Methods.converge_root        
    
        # --------------------------------------------------------------
        #   Iterate - this is iterated
        # --------------------------------------------------------------
        iterate = self.process.iterate
                
        # Update Initials
        iterate.initials = Process()
        iterate.initials.time              = Methods.Common.Frames.initialize_time
        iterate.initials.weights           = Methods.Common.Weights.initialize_weights
        iterate.initials.inertial_position = Methods.Common.Frames.initialize_inertial_position
        iterate.initials.planet_position   = Methods.Common.Frames.initialize_planet_position
        
        # Unpack Unknowns
        iterate.unknowns = Process()
        iterate.unknowns.mission           =  Methods.Ground.Battery_Charge_Discharge.unpack_unknowns
        
        # Update Conditions
        iterate.conditions = Process()
        iterate.conditions.differentials   = Methods.Common.Numerics.update_differentials_time
        iterate.conditions.altitude        = Methods.Common.Aerodynamics.update_altitude
        iterate.conditions.atmosphere      = Methods.Common.Aerodynamics.update_atmosphere
        iterate.conditions.gravity         = Methods.Common.Weights.update_gravity
        iterate.conditions.freestream      = Methods.Common.Aerodynamics.update_freestream
        iterate.conditions.orientations    = Methods.Common.Frames.update_orientations
        iterate.conditions.propulsion      = Methods.Common.Energy.update_thrust
        iterate.conditions.aerodynamics    = Methods.Common.Aerodynamics.update_aerodynamics
        iterate.conditions.stability       = Methods.Common.Aerodynamics.update_stability
        iterate.conditions.weights         = Methods.Common.Weights.update_weights
        iterate.conditions.forces          = Methods.Common.Frames.update_forces
        iterate.conditions.planet_position = Methods.Common.Frames.update_planet_position
    
        # Solve Residuals
        iterate.residuals = Process()      
        
        # --------------------------------------------------------------
        #   Finalize - after iteration
        # --------------------------------------------------------------
        finalize = self.process.finalize
        
        # Post Processing
        finalize.post_process = Process()        
        finalize.post_process.inertial_position = Methods.Common.Frames.integrate_inertial_horizontal_position
        finalize.post_process.stability         = skip
        finalize.post_process.aero_derivatives  = skip
        finalize.post_process.noise             = skip
        
        return
