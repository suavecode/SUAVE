## @ingroup Analyses-Stability
# Fidelity_Zero.py
# 
# Created:  Andrew, July 2014
# Modified: M. Vegh, November 2015         
# Modified: Feb 2016, Andrew Wendorff

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

        self.configuration = Data()

        self.geometry      = Data()

        self.stability_model = Data()
        self.stability_model.short_period = Data()
        self.stability_model.short_period.natural_frequency = 0.0
        self.stability_model.short_period.damping_ratio     = 0.0
        self.stability_model.phugoid = Data()
        self.stability_model.phugoid.damping_ratio     = 0.0
        self.stability_model.phugoid.natural_frequency = 0.0
        self.stability_model.roll_tau                  = 0.0
        self.stability_model.spiral_tau                = 0.0 
        self.stability_model.dutch_roll = Data()
        self.stability_model.dutch_roll.damping_ratio     = 0.0
        self.stability_model.dutch_roll.natural_frequency = 0.0

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
        geometry         = self.geometry #really a vehicle object
        configuration    = self.configuration
        stability_model  = self.stability_model

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
        N/A

        Inputs:
        conditions - DataDict() of aerodynamic conditions
        results    - DataDict() of moment coeffients and stability derivatives

        Outputs:
        None

        Properties Used:
        self.geometry
        """         

        # unpack
        configuration   = self.configuration
        geometry        = self.geometry
        stability_model = self.stability_model

        q             = conditions.freestream.dynamic_pressure
        Sref          = geometry.reference_area    
        mach          = conditions.freestream.mach_number
        velocity      = conditions.freestream.velocity
        density       = conditions.freestream.density
        Span          = geometry.wings['main_wing'].spans.projected
        mac           = geometry.wings['main_wing'].chords.mean_aerodynamic
        aero          = conditions.aerodynamics

        # set up data structures
        static_stability  = Data()
        dynamic_stability = Data()

        # Calculate CL_alpha 
        conditions.lift_curve_slope = datcom(geometry.wings['main_wing'],mach)

        # Calculate change in downwash with respect to change in angle of attack
        for surf in geometry.wings:
            sref          = surf.areas.reference
            span          = (surf.aspect_ratio * sref ) ** 0.5
            surf.CL_alpha = datcom(surf,mach)
            surf.ep_alpha = Supporting_Functions.ep_alpha(surf.CL_alpha, sref, span)

        # Static Stability Methods
        static_stability.cm_alpha,static_stability.cm0, static_stability.CM  = taw_cmalpha(geometry,mach,conditions,configuration)

        if 'vertical_stabilizer' in geometry.wings:
            static_stability.cn_beta  = taw_cnbeta(geometry,conditions,configuration)
        else:
            static_stability.cn_beta = np.zeros_like(mach)

        # calculate the static margin
        static_stability.static_margin = -static_stability.cm_alpha/conditions.lift_curve_slope

        # Dynamic Stability
        if np.count_nonzero(configuration.mass_properties.moments_of_inertia.tensor) > 0:    
            # Dynamic Stability Approximation Methods - valid for non-zero I tensor

            # Derivative of yawing moment with respect to the rate of yaw
            cDw = aero.drag_breakdown.parasite['main_wing'].parasite_drag_coefficient # Might not be the correct value
            l_v = geometry.wings['vertical_stabilizer'].origin[0] + geometry.wings['vertical_stabilizer'].aerodynamic_center[0] - geometry.wings['main_wing'].origin[0] - geometry.wings['main_wing'].aerodynamic_center[0]
            dynamic_stability.cn_r = Supporting_Functions.cn_r(cDw, geometry.wings['vertical_stabilizer'].areas.reference, Sref, l_v, span, geometry.wings['vertical_stabilizer'].dynamic_pressure_ratio, geometry.wings['vertical_stabilizer'].CL_alpha)

            # Derivative of rolling moment with respect to roll rate
            dynamic_stability.cl_p = Supporting_Functions.cl_p(conditions.lift_curve_slope, geometry)

            # Derivative of roll rate with respect to sideslip (dihedral effect)
            if 'dihedral' in geometry.wings['main_wing']:
                dynamic_stability.cl_beta = Supporting_Functions.cl_beta(geometry, dynamic_stability.cl_p)
            else:
                dynamic_stability.cl_beta = np.zeros(1)

            dynamic_stability.cy_beta = 0

            # Derivative of pitching moment with respect to pitch rate
            l_t                    = geometry.wings['horizontal_stabilizer'].origin[0] + geometry.wings['horizontal_stabilizer'].aerodynamic_center[0] - geometry.wings['main_wing'].origin[0] - geometry.wings['main_wing'].aerodynamic_center[0] #Need to check this is the length of the horizontal tail moment arm       
            dynamic_stability.cm_q = Supporting_Functions.cm_q(conditions.lift_curve_slope, l_t,mac) # Need to check Cm_i versus Cm_alpha

            # Derivative of pitching rate with respect to d(alpha)/d(t)
            dynamic_stability.cm_alpha_dot = Supporting_Functions.cm_alphadot(static_stability.cm_alpha, geometry.wings['horizontal_stabilizer'].ep_alpha, l_t, mac) # Need to check Cm_i versus Cm_alpha

            # Derivative of Z-axis force with respect to angle of attack  
            dynamic_stability.cz_alpha = Supporting_Functions.cz_alpha(aero.drag_coefficient,conditions.lift_curve_slope)


            stability_model.dutch_roll = Approximations.dutch_roll(velocity, static_stability.cn_beta, Sref, density, Span, configuration.mass_properties.moments_of_inertia.tensor[2][2], dynamic_stability.cn_r)

            if dynamic_stability.cl_p.all() != 0:
                stability_model.roll_tau   = Approximations.roll(configuration.mass_properties.moments_of_inertia.tensor[2][2], Sref, density, velocity, Span, dynamic_stability.cl_p)
                dynamic_stability.cy_phi   = Supporting_Functions.cy_phi(aero.lift_coefficient)
                dynamic_stability.cl_r     = Supporting_Functions.cl_r(aero.lift_coefficient) # Will need to be changed
                stability_model.spiral_tau = Approximations.spiral(conditions.weights.total_mass, velocity, density, Sref, dynamic_stability.cl_p, static_stability.cn_beta, dynamic_stability.cy_phi, dynamic_stability.cl_beta, dynamic_stability.cn_r, dynamic_stability.cl_r)
            stability_model.short_period   = Approximations.short_period(velocity, density, Sref, mac, dynamic_stability.cm_q, dynamic_stability.cz_alpha, conditions.weights.total_mass, static_stability.cm_alpha, configuration.mass_properties.moments_of_inertia.tensor[1][1], dynamic_stability.cm_alpha_dot)
            stability_model.phugoid        = Approximations.phugoid(conditions.freestream.gravity, conditions.freestream.velocity, aero.drag_coefficient, aero.lift_coefficient)

            # Dynamic Stability Full Linearized Methods
            if dynamic_stability.cy_beta != 0 and dynamic_stability.cl_p.all() != 0 and dynamic_stability.cl_beta.all() != 0:
                theta = conditions.frames.wind.body_rotations[:,1]
                dynamic_stability.cl_psi       = Supporting_Functions.cy_psi(aero.lift_coefficient, theta)
                dynamic_stability.cL_u         = 0
                dynamic_stability.cz_u         = Supporting_Functions.cz_u(aero.lift_coefficient,velocity,dynamic_stability.cL_u)
                dynamic_stability.cz_alpha_dot = Supporting_Functions.cz_alphadot(static_stability.cm_alpha, geometry.wings['horizontal_stabilizer'].ep_alpha)
                dynamic_stability.cz_q         = Supporting_Functions.cz_q(static_stability.cm_alpha)
                dynamic_stability.cx_u         = Supporting_Functions.cx_u(aero.drag_coefficient)
                dynamic_stability.cx_alpha     = Supporting_Functions.cx_alpha(aero.lift_coefficient, conditions.lift_curve_slope)

                lateral_directional = Full_Linearized_Equations.lateral_directional(velocity, static_stability.cn_beta , Sref, density, Span, configuration.mass_properties.moments_of_inertia.tensor[2][2], dynamic_stability.cn_r, configuration.mass_properties.moments_of_inertia.tensor[0][0], dynamic_stability.cl_p, configuration.mass_properties.moments_of_inertia.tensor[0][2], dynamic_stability.cl_r, dynamic_stability.cl_beta, dynamic_stability.cn_p, dynamic_stability.cy_phi, dynamic_stability.cy_psi, dynamic_stability.cy_beta, conditions.weights.total_mass)
                longitudinal        = Full_Linearized_Equations.longitudinal(velocity, density, Sref, mac, dynamic_stability.cm_q, dynamic_stability.cz_alpha, conditions.weights.total_mass, static_stability.cm_alpha, configuration.mass_properties.moments_of_inertia.tensor[1][1], dynamic_stability.cm_alpha_dot, dynamic_stability.cz_u, dynamic_stability.cz_alpha_dot, dynamic_stability.cz_q, -aero.lift_coefficient, theta, dynamic_stability.cx_u, dynamic_stability.cx_alpha)                    
                stability_model.dutch_roll.natural_frequency   = lateral_directional.dutch_natural_frequency
                stability_model.dutch_roll.damping_ratio       = lateral_directional.dutch_damping_ratio
                stability_model.spiral_tau                     = lateral_directional.spiral_tau
                stability_model.roll_tau                       = lateral_directional.roll_tau
                stability_model.short_period.natural_frequency = longitudinal.short_natural_frequency
                stability_model.short_period.damping_ratio     = longitudinal.short_damping_ratio
                stability_model.phugoid.natural_frequency      = longitudinal.phugoid_natural_frequency
                stability_model.phugoid.damping_ratio          = longitudinal.phugoid_damping_ratio

        # pack results
        results = Data()
        results.static  = static_stability
        results.dynamic = dynamic_stability

        return results
