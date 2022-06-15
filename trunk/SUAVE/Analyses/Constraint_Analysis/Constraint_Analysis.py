## @ingroup Analyses-Constraint_Analysis
# Constraint_Analysis.py
#
# Created:  Oct 2020, S. Karpuk 
# Modified: 
#          

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Core import Units, Data
from SUAVE.Methods.Constraint_Analysis.compute_take_off_constraint  import compute_take_off_constraint
from SUAVE.Methods.Constraint_Analysis.compute_climb_constraint     import compute_climb_constraint
from SUAVE.Methods.Constraint_Analysis.compute_OEI_climb_constraint import compute_OEI_climb_constraint
from SUAVE.Methods.Constraint_Analysis.compute_turn_constraint      import compute_turn_constraint
from SUAVE.Methods.Constraint_Analysis.compute_cruise_constraint    import compute_cruise_constraint
from SUAVE.Methods.Constraint_Analysis.compute_ceiling_constraint   import compute_ceiling_constraint
from SUAVE.Methods.Constraint_Analysis.compute_landing_constraint   import compute_landing_constraint


# Package imports
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------
## @ingroup Analyses-Constraint_Analysis
class Constraint_Analysis():
    """Creates a constraint diagram using classical sizing methods

    Assumptions:
        None

    Source:
        S. Gudmundsson "General Aviation aircraft Design. Applied methods and procedures", Butterworth-Heinemann (6 Nov. 2013), ISBN - 0123973082
        L. Loftin,Subsonic Aircraft: Evolution and the Matching of Size to Performance, NASA Ref-erence Publication 1060, August 1980
        M. Nita, D.Scholtz, 'Estimating the Oswald factor from basic aircraft geometrical parameters',Deutscher Luft- und Raumfahrtkongress 20121DocumentID: 281424

    
    """
    
    def __init__(self):
        """This sets the default values.
    
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
            
        self.tag                = 'Constraint analysis'
        self.plot_tag           = False


        # Defines default constraint analyses
        self.analyses            = Data()
        self.analyses.takeoff    = Data()
        self.analyses.cruise     = Data()
        self.analyses.max_cruise = Data()
        self.analyses.landing    = Data()
        self.analyses.OEI_climb  = Data()
        self.analyses.turn       = Data()
        self.analyses.climb      = Data()
        self.analyses.ceiling    = Data()

        self.analyses.takeoff.compute    = True
        self.analyses.cruise.compute     = True
        self.analyses.max_cruise.compute = False
        self.analyses.landing.compute    = True
        self.analyses.OEI_climb.compute  = True
        self.analyses.turn.compute       = False
        self.analyses.climb.compute      = False
        self.analyses.ceiling.compute    = False
        
        self.wing_loading      = np.arange(2, 200, 0.25) * Units['force_pound/foot**2']
        self.design_point_type = None

        # Default parameters for the constraint analysis
        # take-off
        self.analyses.takeoff.runway_elevation     = 0.0
        self.analyses.takeoff.ground_run           = 0.0
        self.analyses.takeoff.rolling_resistance   = 0.05
        self.analyses.takeoff.liftoff_speed_factor = 1.1
        self.analyses.takeoff.delta_ISA            = 0.0
        # climb
        self.analyses.climb.altitude   = 0.0
        self.analyses.climb.airspeed   = 0.0
        self.analyses.climb.climb_rate = 0.0
        self.analyses.climb.delta_ISA  = 0.0
        # OEI climb
        self.analyses.OEI_climb.climb_speed_factor = 1.2
        # cruise
        self.analyses.cruise.altitude        = 0.0
        self.analyses.cruise.delta_ISA       = 0.0
        self.analyses.cruise.airspeed        = 0.0
        self.analyses.cruise.thrust_fraction = 0.0
        # max cruise
        self.analyses.max_cruise.altitude        = 0.0
        self.analyses.max_cruise.delta_ISA       = 0.0
        self.analyses.max_cruise.mach            = 0.0
        self.analyses.max_cruise.thrust_fraction = 0.0
        # turn
        self.analyses.turn.angle           = 0.0
        self.analyses.turn.altitude        = 0.0
        self.analyses.turn.delta_ISA       = 0.0
        self.analyses.turn.mach            = 0.0
        self.analyses.turn.specific_energy = 0.0
        self.analyses.turn.thrust_fraction = 0.0
        # ceiling
        self.analyses.ceiling.altitude  = 0.0
        self.analyses.ceiling.delta_ISA = 0.0
        self.analyses.ceiling.mach      = 0.0
        # landing
        self.analyses.landing.ground_roll           = 0.0
        self.analyses.landing.approach_speed_factor = 1.23
        self.analyses.landing.runway_elevation      = 0.0
        self.analyses.landing.delta_ISA             = 0.0 

        # Default aircraft properties

        # engine
        self.engine = Data()
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

        
    def create_constraint_diagram(self,vehicle): 
        """Creates a constraint diagram

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
            Constraint diagram data
            Design wing loading and thrust-to-weight ratio

        Properties Used:
        """

        # Go through each constraint estimation to find corresponding thrust-to-weight ratios
        analyses          = self.analyses
        plot_legend       = []

        constraint_results = Data()
        constraint_results.constraint_matrix = []

        if analyses.takeoff.compute == True:
            constraint_results.constraint_matrix.append(compute_take_off_constraint(self,vehicle))
            plot_legend.append("Take-off")

        if analyses.climb.compute == True:
            constraint_results.constraint_matrix.append(compute_climb_constraint(self,vehicle))
            plot_legend.append("Climb")

        if analyses.turn.compute == True:
            constraint_results.constraint_matrix.append(compute_turn_constraint(self,vehicle))
            plot_legend.append("Turn")           

        if analyses.max_cruise.compute == True:
            constraint_results.constraint_matrix.append(compute_cruise_constraint(self,vehicle,'max cruise'))
            plot_legend.append("Max Cruise") 

        if analyses.max_cruise.compute == False:
            constraint_results.constraint_matrix.append(compute_cruise_constraint(self,vehicle,'cruise'))
            plot_legend.append("Normal Cruise") 

        if analyses.landing.compute == True:
            WS_landing = compute_landing_constraint(self,vehicle)

        if analyses.OEI_climb.compute == True:
            constraint_results.constraint_matrix.append(compute_OEI_climb_constraint(self,vehicle))
            plot_legend.append("OEI climb") 

        if analyses.ceiling.compute == True:
            constraint_results.constraint_matrix.append(compute_ceiling_constraint(self,vehicle))
            plot_legend.append("Ceiling") 


        # Find the design point based on user preferences
        WS = self.wing_loading
        combined_curve                           = np.amax(constraint_results.constraint_matrix,0)
        constraint_results.combined_design_curve = combined_curve 

        if self.design_point_type == 'minimum thrust-to-weight' or  self.design_point_type == 'minimum power-to-weight':
            design_TW = min(combined_curve)                              
            design_WS = WS[np.argmin(combined_curve)] 
        elif self.design_point_type == 'maximum wing loading':
            design_WS = WS_landing[0]
            design_TW = np.interp(design_WS,WS,combined_curve) 

        # Check the landing constraint
        if design_WS > WS_landing[0]:
            design_WS = WS_landing[0]
            design_TW = np.interp(design_WS, WS, combined_curve) 


        # Pack outputs
        constraint_results.des_wing_loading     = design_WS
        constraint_results.des_thrust_to_weight = design_TW
        constraint_results.landing_wing_loading = WS_landing
        constraint_results.wing_loading         = self.wing_loading
        constraint_results.constraint_line      = combined_curve
        constraint_results.plot_legend          = plot_legend 


        return constraint_results


    
