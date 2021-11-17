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
import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Methods.Constraint_Analysis.compute_take_off_constraint  import compute_take_off_constraint
from SUAVE.Methods.Constraint_Analysis.compute_climb_constraint     import compute_climb_constraint
from SUAVE.Methods.Constraint_Analysis.compute_OEI_climb_constraint import compute_OEI_climb_constraint
from SUAVE.Methods.Constraint_Analysis.compute_turn_constraint      import compute_turn_constraint
from SUAVE.Methods.Constraint_Analysis.compute_cruise_constraint    import compute_cruise_constraint
from SUAVE.Methods.Constraint_Analysis.compute_ceiling_constraint   import compute_ceiling_constraint
from SUAVE.Methods.Constraint_Analysis.compute_landing_constraint   import compute_landing_constraint
from SUAVE.Input_Output.Results.plot_constraint_diagram                     import plot_constraint_diagram

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
        """This sets the default values and methods for the analysis.

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
        self.engine.degree_of_hybridization = 0.0
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
        

    def create_constraint_diagram(constraint_analysis): 
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
        analyses          = constraint_analysis.analyses
        plot_legend       = []

        constraint_analysis.constraint_matrix = []

        if analyses.takeoff == True:
            constraint_analysis.constraint_matrix.append(compute_take_off_constraint(constraint_analysis))
            plot_legend.append("Take-off")

        if analyses.climb == True:
            constraint_analysis.constraint_matrix.append(compute_climb_constraint(constraint_analysis))
            plot_legend.append("Climb")

        if analyses.turn == True:
            constraint_analysis.constraint_matrix.append(compute_turn_constraint(constraint_analysis))
            plot_legend.append("Turn")           

        if analyses.max_cruise == True:
            constraint_analysis.constraint_matrix.append(compute_cruise_constraint(constraint_analysis,'max cruise'))
            plot_legend.append("Max Cruise") 

        if analyses.max_cruise == False:
            constraint_analysis.constraint_matrix.append(compute_cruise_constraint(constraint_analysis,'cruise'))
            plot_legend.append("Normal Cruise") 

        if analyses.landing == True:
            WS_landing = compute_landing_constraint(constraint_analysis)

        if analyses.OEI_climb == True:
            constraint_analysis.constraint_matrix.append(compute_OEI_climb_constraint(constraint_analysis))
            plot_legend.append("OEI climb") 

        if analyses.ceiling == True:
            constraint_analysis.constraint_matrix.append(compute_ceiling_constraint(constraint_analysis))
            plot_legend.append("Ceiling") 


        # Find the design point based on user preferences
        WS = constraint_analysis.wing_loading
        combined_curve                            = np.amax(constraint_analysis.constraint_matrix,0)
        constraint_analysis.combined_design_curve = combined_curve 

        if constraint_analysis.design_point_type == 'minimum thrust-to-weight' or  constraint_analysis.design_point_type == 'minimum power-to-weight':
            design_TW = min(combined_curve)                              
            design_WS = WS[np.argmin(combined_curve)] 
        elif constraint_analysis.design_point_type == 'maximum wing loading':
            design_WS = WS_landing[0]
            design_TW = np.interp(design_WS,WS,combined_curve) 

        # Check the landing constraint
        if design_WS > WS_landing[0]:
            design_WS = WS_landing[0]
            design_TW = np.interp(design_WS, WS, combined_curve) 


        # Pack outputs
        constraint_analysis.des_wing_loading     = design_WS
        constraint_analysis.des_thrust_to_weight = design_TW
        constraint_analysis.landing_wing_loading = WS_landing
        constraint_analysis.constraint_line      = combined_curve
        constraint_analysis.plot_legend          = plot_legend 



        return constraint_analysis


    
