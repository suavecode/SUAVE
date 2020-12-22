## @ingroup Analyses-Stability
# Fidelity_Zero.py
# 
# Created:  Andrew, July 2014
# Modified: M. Vegh, November 2015         
# Modified: Feb 2016, Andrew Wendorff
#           Mar 2020, M. Clarke
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

import SUAVE

from SUAVE.Core import Data

# local imports
from .Stability import Stability


# import SUAVE methods
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Tube_Wing.taw_cmalpha import taw_cmalpha
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Tube_Wing.taw_cnbeta import taw_cnbeta
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.datcom import datcom
from SUAVE.Methods.Flight_Dynamics.Dynamic_Stability import Approximations as Approximations
from SUAVE.Methods.Flight_Dynamics.Dynamic_Stability import Full_Linearized_Equations as Full_Linearized_Equations
from SUAVE.Methods.Flight_Dynamics.Dynamic_Stability.Full_Linearized_Equations import Supporting_Functions as Supporting_Functions

# package imports
import numpy as np


# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

## @ingroup Analyses-Stability
class Fidelity_Zero(Stability):
    """This is an analysis based on low-fidelity models.

    Assumptions:
    Subsonic

    Source:
    Primarily based on adg.stanford.edu, see methods for details
    """ 
    def __defaults__(self):
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
        # Initialize quantities

        self.configuration                                    = Data()
        self.geometry                                         = Data() 

        return

    def finalize(self):
        """Finalizes the surrogate needed for lift calculation.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        self.geometry
        """                         
        # unpack
        geometry         = self.geometry      # really a vehicle object
        configuration    = self.configuration 

        configuration.mass_properties = geometry.mass_properties

        if 'fuel' in geometry: #fuel has been assigned(from weight statements)
            configuration.fuel = geometry.fuel
        else: #assign as zero to allow things to run
            fuel = SUAVE.Components.Physical_Component()
            fuel.mass_properties.mass = 0.
            configuration.fuel        = fuel

    def __call__(self,conditions):
        """ Process vehicle to setup geometry, condititon and configuration

        Assumptions:
        None

        Source:
        N/4

        Inputs:
        conditions - DataDict() of aerodynamic conditions
        results    - DataDict() of moment coeffients and stability and body axis derivatives

        Outputs:
        None

        Properties Used:
        self.geometry
        """         

        # unpack
        configuration = self.configuration
        geometry      = self.geometry 
        q             = conditions.freestream.dynamic_pressure
        Sref          = geometry.reference_area    
        mach          = conditions.freestream.mach_number
        velocity      = conditions.freestream.velocity
        density       = conditions.freestream.density
        Span          = geometry.wings['main_wing'].spans.projected
        mac           = geometry.wings['main_wing'].chords.mean_aerodynamic
        cg_x          = geometry.mass_properties.center_of_gravity[0]  
        aero          = conditions.aerodynamics

        # set up data structures
        stability         = Data()
        stability.static  = Data()
        stability.dynamic = Data()

        # Calculate CL_alpha 
        conditions.lift_curve_slope = datcom(geometry.wings['main_wing'],mach)

        # Calculate change in downwash with respect to change in angle of attack
        for surf in geometry.wings:
            sref          = surf.areas.reference
            span          = (surf.aspect_ratio * sref ) ** 0.5
            surf.CL_alpha = datcom(surf,mach)
            surf.ep_alpha = Supporting_Functions.ep_alpha(surf.CL_alpha, sref, span)

        # Static Stability Methods
        stability.static.Cm_alpha,stability.static.Cm0, stability.static.CM  = taw_cmalpha(geometry,mach,conditions,configuration)

        if 'vertical_stabilizer' in geometry.wings:
            stability.static.Cn_beta  = taw_cnbeta(geometry,conditions,configuration)
        else:
            stability.static.Cn_beta = np.zeros_like(mach)

        # calculate the static margin
        stability.static.static_margin = -stability.static.Cm_alpha/conditions.lift_curve_slope
        
        # neutral point 
        stability.static.neutral_point = cg_x + mac*stability.static.static_margin
        
        # Dynamic Stability
        if np.count_nonzero(configuration.mass_properties.moments_of_inertia.tensor) > 0:    
            # Dynamic Stability Approximation Methods - valid for non-zero I tensor

            # Derivative of yawing moment with respect to the rate of yaw
            cDw = aero.drag_breakdown.parasite['main_wing'].parasite_drag_coefficient # Might not be the correct value
            l_v = geometry.wings['vertical_stabilizer'].origin[0][0] + geometry.wings['vertical_stabilizer'].aerodynamic_center[0] - geometry.wings['main_wing'].origin[0][0] - geometry.wings['main_wing'].aerodynamic_center[0]
            stability.static.Cn_r = Supporting_Functions.cn_r(cDw, geometry.wings['vertical_stabilizer'].areas.reference, Sref, l_v, span, geometry.wings['vertical_stabilizer'].dynamic_pressure_ratio, geometry.wings['vertical_stabilizer'].CL_alpha)

            # Derivative of rolling moment with respect to roll rate
            stability.static.Cl_p = Supporting_Functions.cl_p(conditions.lift_curve_slope, geometry)

            # Derivative of roll rate with respect to sideslip (dihedral effect)
            if 'dihedral' in geometry.wings['main_wing']:
                stability.static.Cl_beta = Supporting_Functions.cl_beta(geometry, stability.static.Cl_p)
            else:
                stability.static.Cl_beta = np.zeros(1)

            stability.static.Cy_beta = 0

            # Derivative of pitching moment with respect to pitch rate
            l_t                    = geometry.wings['horizontal_stabilizer'].origin[0][0] + geometry.wings['horizontal_stabilizer'].aerodynamic_center[0] - geometry.wings['main_wing'].origin[0][0] - geometry.wings['main_wing'].aerodynamic_center[0] #Need to check this is the length of the horizontal tail moment arm       
            stability.static.Cm_q  = Supporting_Functions.cm_q(conditions.lift_curve_slope, l_t,mac) # Need to check Cm_i versus Cm_alpha

            # Derivative of pitching rate with respect to d(alpha)/d(t)
            stability.static.Cm_alpha_dot = Supporting_Functions.cm_alphadot(stability.static.Cm_alpha, geometry.wings['horizontal_stabilizer'].ep_alpha, l_t, mac) # Need to check Cm_i versus Cm_alpha

            # Derivative of Z-axis force with respect to angle of attack  
            stability.static.Cz_alpha = Supporting_Functions.cz_alpha(aero.drag_coefficient,conditions.lift_curve_slope)

            dutch_roll = Approximations.dutch_roll(velocity, stability.static.Cn_beta, Sref, density, Span, configuration.mass_properties.moments_of_inertia.tensor[2][2], stability.static.Cn_r)
            stability.dynamic.dutchRollFreqHz             = dutch_roll.natural_frequency
            stability.dynamic.dutchRollDamping            = dutch_roll.damping_ratio

            if stability.static.Cl_p.all() != 0:
                stability.dynamic.rollSubsistenceTimeConstant   = Approximations.roll(configuration.mass_properties.moments_of_inertia.tensor[2][2], Sref, density, velocity, Span, stability.static.Cl_p)      
                stability.static.Cy_phi                                  = Supporting_Functions.cy_phi(aero.lift_coefficient)
                stability.static.Cl_r                                    = Supporting_Functions.cl_r(aero.lift_coefficient) # Will need to be changed
                stability.dynamic.spiralSubsistenceTimeConstant = Approximations.spiral(conditions.weights.total_mass, velocity, density, Sref, stability.static.Cl_p, stability.static.Cn_beta, stability.static.Cy_phi,\
                                                                            stability.static.Cl_beta, stability.static.Cn_r, stability.static.Cl_r)
            
            short_period_res                               = Approximations.short_period(velocity, density, Sref, mac, stability.static.Cm_q, stability.static.Cz_alpha, conditions.weights.total_mass, stability.static.Cm_alpha,\
                                                            configuration.mass_properties.moments_of_inertia.tensor[1][1], stability.static.Cm_alpha_dot)
            stability.dynamic.shortPeriodFreqHz  = short_period_res.natural_frequency 
            stability.dynamic.shortPeriodDamp    = short_period_res.damping_ratio 
            
            phugoid_res                               = Approximations.phugoid(conditions.freestream.gravity, conditions.freestream.velocity, aero.drag_coefficient, aero.lift_coefficient)
            stability.dynamic.phugoidFreqHz = phugoid_res.natural_frequency
            stability.dynamic.phugoidDamp   = phugoid_res.damping_ratio

            # Dynamic Stability Full Linearized Methods
            if stability.static.Cy_beta != 0 and stability.static.Cl_p.all() != 0 and stability.static.Cl_beta.all() != 0:
                theta = conditions.frames.wind.body_rotations[:,1]
                stability.static.Cy_psi       = Supporting_Functions.cy_psi(aero.lift_coefficient, theta)
                stability.static.CL_u         = 0
                stability.static.Cz_u         = Supporting_Functions.cz_u(aero.lift_coefficient,velocity,stability.static.cL_u)
                stability.static.Cz_alpha_dot = Supporting_Functions.cz_alphadot(stability.static.Cm_alpha, geometry.wings['horizontal_stabilizer'].ep_alpha)
                stability.static.Cz_q         = Supporting_Functions.cz_q(stability.static.Cm_alpha)
                stability.static.Cx_u         = Supporting_Functions.cx_u(aero.drag_coefficient)
                stability.static.Cx_alpha     = Supporting_Functions.cx_alpha(aero.lift_coefficient, conditions.lift_curve_slope)

                lateral_directional = Full_Linearized_Equations.lateral_directional(velocity, stability.static.cn_beta , Sref, density, Span, configuration.mass_properties.moments_of_inertia.tensor[2][2], stability.static.Cn_r,\
                                    configuration.mass_properties.moments_of_inertia.tensor[0][0], stability.static.Cl_p, configuration.mass_properties.moments_of_inertia.tensor[0][2], stability.static.Cl_r, stability.static.Cl_beta,\
                                    stability.static.Cn_p, stability.static.Cy_phi, stability.static.Cy_psi, stability.static.Cy_beta, conditions.weights.total_mass)
                longitudinal        = Full_Linearized_Equations.longitudinal(velocity, density, Sref, mac, stability.static.Cm_q, stability.static.Cz_alpha, conditions.weights.total_mass, stability.static.Cm_alpha, \
                                    configuration.mass_properties.moments_of_inertia.tensor[1][1], stability.static.Cm_alpha_dot, stability.static.Cz_u, stability.static.Cz_alpha_dot, stability.static.Cz_q, -aero.lift_coefficient,\
                                    theta, stability.static.Cx_u, stability.static.Cx_alpha)                    
                stability.dynamic.dutchRollFreqHz     = lateral_directional.dutch_natural_frequency
                stability.dynamic.dutchRollDamping    = lateral_directional.dutch_damping_ratio
                stability.dynamic.spiralSubsistenceTimeConstant  = lateral_directional.spiral_tau
                stability.dynamic.rollSubsistenceTimeConstant    = lateral_directional.roll_tau
                stability.dynamic.shortPeriodFreqHz             = longitudinal.short_natural_frequency
                stability.dynamic.shortPeriodDamp               = longitudinal.short_damping_ratio
                stability.dynamic.phugoidFreqHz                 = longitudinal.phugoid_natural_frequency
                stability.dynamic.phugoidDamp                   = longitudinal.phugoid_damping_ratio
                                                                        
        return stability 
