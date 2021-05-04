## @ingroup Methods-Performance
# maximum_lift_to_drag.py
#
# Created: Dec 2020, E. Botero
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
import numpy as np
import scipy as sp

import SUAVE

# ----------------------------------------------------------------------
#  Propeller Range and Endurance Speeds
# ----------------------------------------------------------------------


def maximum_lift_to_drag(analyses,altitude,CL_max,up_bnd,delta_isa):
        """ Computes L/D max at give altitude. This runs a mini mission wrapped by an optimizer to find the L/D max.
        up_bnd is the fastest airspeed that the optimizer can try. The output is a dictionary containing the maximum 
        lift to drag value as well as the airspeed.

        Assumptions:
        No propulsion source is given


        Source:
        N/A


        Inputs:
        analyses.atmosphere        [-]
        analyses.aerodynamics      [-]
        altitude                   [m]
        CL_max                     [float]
        up_bnd                     [m/s]
        delta_isa                  [deg C]

        Outputs:
        results.L_D_max.air_speed  [m/s]
        results.L_D_max.value      [-]

        Properties Used:
        N/A
        """           


        # setup a mission that runs a single point segment without propulsion
        def mini_mission():
        
                # ------------------------------------------------------------------
                #   Initialize the Mission
                # ------------------------------------------------------------------
                mission = SUAVE.Analyses.Mission.Sequential_Segments()
                mission.tag = 'the_mission'
        
                # ------------------------------------------------------------------
                #  Single Point Segment 1: constant Speed, constant altitude
                # ------------------------------------------------------------------ 
                segment = SUAVE.Analyses.Mission.Segments.Single_Point.Set_Speed_Set_Altitude_No_Propulsion()
                segment.tag = "single_point" 
                segment.analyses.extend(analyses) 
                segment.altitude              = altitude
                segment.temperature_deviation = delta_isa
        
                # add to misison
                mission.append_segment(segment)    
        
                return mission


        # This is what's called by the optimizer for L/D Max
        def single_point_LDmax(X):

                # Modify the mission for the next iteration
                mission.segments.single_point.air_speed = X

                # Run the Mission      
                point_results = mission.evaluate()    

                CL = point_results.segments.single_point.conditions.aerodynamics.lift_coefficient
                CD = point_results.segments.single_point.conditions.aerodynamics.drag_coefficient

                L_D = -CL/CD # Negative because optimizers want to make things small
                
                return L_D


        # ------------------------------------------------------------------
        #   Run the optimizer to solve
        # ------------------------------------------------------------------    

        # Setup the a mini mission
        mission = mini_mission()

        # Takeoff mass:
        mass = analyses.aerodynamics.geometry.mass_properties.takeoff

        # Calculate the stall speed
        Vs = stall_speed(analyses,mass,CL_max,altitude,delta_isa)[0][0]

        # The final results to save
        results = Data()

        # Wrap an optimizer around both functions to solve for L/D Max
        outputs_ld = sp.optimize.minimize_scalar(single_point_LDmax,bounds=(Vs,up_bnd),method='bounded')    

        # Pack the results
        results.L_D_max = Data()
        results.L_D_max.air_speed = outputs_ld.x
        results.L_D_max.value     = -outputs_ld.fun[0][0]    

        return results    


def stall_speed(analyses,mass,CL_max,altitude,delta_isa):

        # Unpack
        atmo  = analyses.atmosphere 
        S     = analyses.aerodynamics.geometry.reference_area
        
        # Calculations
        atmo_values       = atmo.compute_values(altitude,delta_isa)
        rho               = atmo_values.density
        sea_level_gravity = atmo.planet.sea_level_gravity
        
        W = mass*sea_level_gravity 
        
        V = np.sqrt(2*W/(rho*S*CL_max))
        
        return V