# Fidelity_Zero.py
# 
# Created:  Andrew, July 2014
# Modified:        


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

import SUAVE

from SUAVE.Structure import Data
from SUAVE.Attributes import Units

# import methods
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Tube_Wing.taw_cmalpha import taw_cmalpha
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Tube_Wing.taw_cnbeta import taw_cnbeta
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.datcom import datcom
from SUAVE.Methods.Flight_Dynamics.Dynamic_Stability import Approximations as Approximations
from SUAVE.Methods.Flight_Dynamics.Dynamic_Stability import Full_Linearized_Equations as Full_Linearized_Equations
from SUAVE.Methods.Flight_Dynamics.Dynamic_Stability.Full_Linearized_Equations import Supporting_Functions as Supporting_Functions


# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Fidelity_Zero(Data):
    """ SUAVE.Attributes.Aerodynamics.Fidelity_Zero
        aerodynamic model that builds a surrogate model for clean wing 
        lift, using vortex lattic, and various handbook methods
        for everything else
        
        this class is callable, see self.__call__
        
    """
    
    def __defaults__(self):
        
        # Initialize quantities
        
        self.configuration = Data()
        self.geometry      = Data()
        
        return
    
    def initialize(self,vehicle):
                        
        # unpack
        geometry         = self.geometry
        configuration    = self.configuration
        
        # copy geometry
        for k in ['Fuselages','Wings','Propulsors']:
            geometry[k] = deepcopy(vehicle[k])
        
        # reference area
        geometry.reference_area = vehicle.S
        configuration.mass_props = vehicle.Mass_Props
        
    
    def __call__(self,conditions):
            """ process vehicle to setup geometry, condititon and configuration
                
                Inputs:
                    conditions - DataDict() of aerodynamic conditions
                    results - DataDict() of 
                    
                Outputs:

                    
                Assumptions:

                    
            """
            
            # unpack
            configuration = self.configuration
            geometry      = self.geometry
            q             = conditions.freestream.dynamic_pressure
            Sref          = geometry.reference_area    
            mach          = conditions.freestream.mach_number
            velocity      = conditions.freestream.velocity
            density       = conditions.freestream.density
            Span          = geometry.Wings['Main Wing'].span
            mac           = geometry.Wings['Main Wing'].chord_mac
            
            # Calculate CL_alpha 
            if not conditions.has_key('lift_curve_slope'):
                conditions.lift_curve_slope = (datcom(geometry.Wings['Main Wing'],mach))
            
            # Calculate change in downwash with respect to change in angle of attack
            for surf in geometry.Wings:
                e = surf.e
                sref = surf.sref
                span = (surf.ar * sref ) ** 0.5
                surf.CL_alpha = datcom(surf,mach)
                surf.ep_alpha = Supporting_Functions.ep_alpha(surf.CL_alpha, sref, span, e)
            
            # Static Stability Methods
            conditions.aerodynamics.cm_alpha = taw_cmalpha(geometry,mach,conditions,configuration)
            conditions.aerodynamics.cn_beta = taw_cnbeta(geometry,conditions,configuration)
            
            if np.count_nonzero(configuration.mass_props.I_cg) > 0:         
                # Dynamic Stability Approximation Methods
                if not conditions.aerodynamics.has_key('cn_r'):  
                    cDw = conditions.aerodynamics.drag_breakdown.parasite['Main Wing'].parasite_drag_coefficient # Might not be the correct value
                    l_v = geometry.Wings['Vertical Stabilizer'].origin[0] + geometry.Wings['Vertical Stabilizer'].aero_center[0] - geometry.Wings['Main Wing'].origin[0] - geometry.Wings['Main Wing'].aero_center[0]
                    conditions.aerodynamics.cn_r = Supporting_Functions.cn_r(cDw, geometry.Wings['Vertical Stabilizer'].sref, Sref, l_v, span, geometry.Wings['Vertical Stabilizer'].eta, geometry.Wings['Vertical Stabilizer'].CL_alpha)
                if not conditions.aerodynamics.has_key('cl_p'):
                    conditions.aerodynamics.cl_p = 0 # Need to see if there is a low fidelity way to calculate cl_p
                    
                if not conditions.aerodynamics.has_key('cl_beta'):
                    conditions.aerodynamics.cl_beta = 0 # Need to see if there is a low fidelity way to calculate cl_p
                
                l_t = geometry.Wings['Horizontal Stabilizer'].origin[0] + geometry.Wings['Horizontal Stabilizer'].aero_center[0] - geometry.Wings['Main Wing'].origin[0] - geometry.Wings['Main Wing'].aero_center[0] #Need to check this is the length of the horizontal tail moment arm       
                
                if not conditions.aerodynamics.has_key('cm_q'):
                    conditions.aerodynamics.cm_q = Supporting_Functions.cm_q(conditions.lift_curve_slope, l_t,mac) # Need to check Cm_i versus Cm_alpha
                
                if not conditions.aerodynamics.has_key('cm_alpha_dot'):
                    conditions.aerodynamics.cm_alpha_dot = Supporting_Functions.cm_alphadot(conditions.lift_curve_slope, geometry.Wings['Horizontal Stabilizer'].ep_alpha, l_t, mac) # Need to check Cm_i versus Cm_alpha
                    
                if not conditions.aerodynamics.has_key('cz_alpha'):
                    conditions.aerodynamics.cz_alpha = Supporting_Functions.cz_alpha(conditions.aerodynamics.drag_coefficient,conditions.lift_curve_slope)                   
                
                conditions.dutch_roll = Approximations.dutch_roll(velocity, conditions.aerodynamics.cn_beta, Sref, density, Span, configuration.mass_props.I_cg[2][2], conditions.aerodynamics.cn_r)
                
                if conditions.aerodynamics.cl_p != 0:                 
                    roll_tau = Approximations.roll(configuration.mass_props.I_cg[2][2], Sref, density, velocity, Span, conditions.aerodynamics.cl_p)
                    if conditions.aerodynamics.cl_beta != 0:
                        conditions.aerodynamics.cy_phi = Supporting_Functions.cy_phi(conditions.aerodynamics.lift_coefficient)
                        conditions.aerodynamics.cl_r = Supporting_Functions.cl_r( conditions.aerodynamics.lift_coefficient) # Will need to be changed
                        spiral_tau = Approximations.spiral(conditions.weights.total_mass, velocity, density, Sref, conditions.aerodynamics.cl_p, conditions.aerodynamics.cn_beta, conditions.aerodynamics.cy_phi, conditions.aerodynamics.cl_beta, conditions.aerodynamics.cn_r, conditions.aerodynamics.cl_r)
                conditions.short_period = Approximations.short_period(velocity, density, Sref, mac, conditions.aerodynamics.cm_q, conditions.aerodynamics.cz_alpha, conditions.weights.total_mass, conditions.aerodynamics.cm_alpha, configuration.mass_props.I_cg[1][1], conditions.aerodynamics.cm_alpha_dot)
                conditions.phugoid = Approximations.phugoid(conditions.freestream.gravity, conditions.freestream.velocity, conditions.aerodynamics.drag_coefficient, conditions.aerodynamics.lift_coefficient)
                
                # Dynamic Stability Full Linearized Methods
                #lateral_directional = Full_Linearized_Equations.lateral_directional(velocity, Cn_Beta, S_gross_w, density, span, I_z, Cn_r, I_x, Cl_p, J_xz, Cl_r, Cl_Beta, Cn_p, Cy_phi, Cy_psi, Cy_Beta, mass)
                #longitudinal = Full_Linearized_Equations.longitudinal(velocity, density, S_gross_w, mac, Cm_q, Cz_alpha, mass, Cm_alpha, Iy, Cm_alpha_dot, Cz_u, Cz_alpha_dot, Cz_q, Cw, Theta, Cx_u, Cx_alpha)
            
            return 