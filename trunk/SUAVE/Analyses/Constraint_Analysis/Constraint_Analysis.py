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
from SUAVE.Input_Output.Results.plot_constraint_diagram             import plot_constraint_diagram
from SUAVE.Vehicle import Vehicle

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
        analyses          = vehicle.constraints.analyses
        plot_legend       = []

        constraint_results = Data()
        constraint_results.constraint_matrix = []

        if analyses.takeoff.compute == True:
            constraint_results.constraint_matrix.append(compute_take_off_constraint(vehicle))
            plot_legend.append("Take-off")

        if analyses.climb.compute == True:
            constraint_results.constraint_matrix.append(compute_climb_constraint(vehicle))
            plot_legend.append("Climb")

        if analyses.turn.compute == True:
            constraint_results.constraint_matrix.append(compute_turn_constraint(vehicle))
            plot_legend.append("Turn")           

        if analyses.max_cruise.compute == True:
            constraint_results.constraint_matrix.append(compute_cruise_constraint(vehicle,'max cruise'))
            plot_legend.append("Max Cruise") 

        if analyses.max_cruise.compute == False:
            constraint_results.constraint_matrix.append(compute_cruise_constraint(vehicle,'cruise'))
            plot_legend.append("Normal Cruise") 

        if analyses.landing.compute == True:
            WS_landing = compute_landing_constraint(vehicle)

        if analyses.OEI_climb.compute == True:
            constraint_results.constraint_matrix.append(compute_OEI_climb_constraint(vehicle))
            plot_legend.append("OEI climb") 

        if analyses.ceiling.compute == True:
            constraint_results.constraint_matrix.append(compute_ceiling_constraint(vehicle))
            plot_legend.append("Ceiling") 


        # Find the design point based on user preferences
        WS = vehicle.constraints.wing_loading
        combined_curve                           = np.amax(constraint_results.constraint_matrix,0)
        constraint_results.combined_design_curve = combined_curve 

        if vehicle.constraints.design_point_type == 'minimum thrust-to-weight' or  vehicle.constraints.design_point_type == 'minimum power-to-weight':
            design_TW = min(combined_curve)                              
            design_WS = WS[np.argmin(combined_curve)] 
        elif vehicle.constraints.design_point_type == 'maximum wing loading':
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
        constraint_results.wing_loading         = vehicle.constraints.wing_loading
        constraint_results.constraint_line      = combined_curve
        constraint_results.plot_legend          = plot_legend 


        return constraint_results


    
