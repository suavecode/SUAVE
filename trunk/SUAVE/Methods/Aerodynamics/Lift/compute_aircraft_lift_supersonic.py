# compute_aircraft_lift.py
# 
# Created:  Anil V., Dec 2013
# Modified: Anil, Trent, Tarik, Feb 2014 
# Modified: Anil  April 2014 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
# suave imports

from SUAVE.Structure import Data
from SUAVE.Attributes import Units

from SUAVE.Attributes.Results import Result
#from SUAVE import Vehicle
from SUAVE.Components.Wings import Wing
from SUAVE.Components.Fuselages import Fuselage
from SUAVE.Components.Propulsors import Turbofan
from SUAVE.Geometry.Two_Dimensional.Planform import wing_planform
from SUAVE.Geometry.Two_Dimensional.Planform import fuselage_planform

#from SUAVE.Attributes.Aerodynamics.Aerodynamics_Surrogate import Aerodynamics_Surrogate
#from SUAVE.Attributes.Aerodynamics.Aerodynamics_Surrogate import Interpolation
from SUAVE.Attributes.Aerodynamics.Aerodynamics_1d_Surrogate import Aerodynamics_1d_Surrogate
from SUAVE.Methods.Aerodynamics.Drag import compute_aircraft_drag



from SUAVE.Attributes.Aerodynamics.Configuration   import Configuration
from SUAVE.Attributes.Aerodynamics.Conditions      import Conditions
from SUAVE.Attributes.Aerodynamics.Geometry        import Geometry


from SUAVE.Methods.Aerodynamics.Lift.weissenger_vortex_lattice import weissinger_vortex_lattice
from SUAVE.Methods.Aerodynamics.Lift.High_lift_correlations.vortex_lift import vortex_lift
#from SUAVE.Methods.Aerodynamics.Lift import compute_aircraft_lift
#from SUAVE.Methods.Aerodynamics.Drag import compute_aircraft_drag


# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp


# ----------------------------------------------------------------------
#  The Function
# ----------------------------------------------------------------------

def compute_aircraft_lift_supersonic(conditions,configuration,geometry):
    """ SUAVE.Methods.Aerodynamics.compute_aircraft_lift(conditions,configuration,geometry)
        computes the lift associated with an aircraft 
        
        Inputs:
            conditions - data dictionary with fields:
                mach_number - float or 1D array of freestream mach numbers
                angle_of_attack - floar or 1D array of angle of attacks
                
            configuration - data dictionary with fields:
                surrogate_models.lift_coefficient - a callable function or class 
                    with inputs of angle of attack and outputs of lift coefficent
                fuselage_lift_correction - the correction to fuselage contribution to lift
                    
            geometry - used for wing
            
        
        Outputs:
            CL - float or 1D array of lift coefficients of the total aircraft
        
        Updates:
            conditions.lift_breakdown - stores results here
            
        Assumptions:
            surrogate model returns total incompressible lift due to wings
            prandtl-glaurert compressibility correction on this
            fuselage contribution to lift correction as a factor
        
    """    
   
    # unpack
    fus_correction = configuration.fuselage_lift_correction
    Mc             = conditions.freestream.mach_number
    AoA            = conditions.aerodynamics.angle_of_attack
    
    # pack for interpolate
    X_interp = AoA    
    wings_lift = np.array([[0.0]] * len(Mc))
    wings_lift_comp = np.array([[0.0]] * len(Mc))
    compress_corr = np.array([[0.0]] * len(Mc))
    aircraft_lift_total = np.array([[0.0]] * len(Mc))
    vortex_cl = np.array([[0.0]] * len(Mc))
    #print aircraft_lift_total
    
    wing = geometry.Wings[0]
    
    for i in range(len(Mc)):

        if Mc[i] <= 1.05:
        
            # the lift surrogate model for wings only
            #print("Low Mach")
            wings_lift_model = configuration.surrogate_models_sub.lift_coefficient
    
        
            # compressibility correction
            if Mc[i] < 0.95:
                compress_corr[i] = 1./(np.sqrt(1.-Mc[i]**2.))
            else:
                compress_corr[i] = 1./(np.sqrt(1.-0.95**2)) # basic approximation for now
            
            # interpolate
            #print wings_lift_model(X_interp)
            wings_lift[i] = wings_lift_model(X_interp[i]) + vortex_lift(X_interp[i],configuration,wing)
            #print wings_lift[i]
            vortex_cl[i] = vortex_lift(X_interp[i],configuration,wing)
            
            # correct lift
            wings_lift_comp[i] = wings_lift[i] * compress_corr[i]            
    
        elif Mc[i] > 1.05:
            
            # Supersonic lift calculation - less accurate for low mach numbers
            #print("High Mach")
            wings_lift_model = configuration.surrogate_models_sup.lift_coefficient
            
            # compressibility correction
            compress_corr[i] = 1./(np.sqrt(Mc[i]**2.-1.))
            
            # interpolate
            #print wings_lift_model(X_interp)
            wings_lift[i] = wings_lift_model(X_interp[i])# + vortex_lift(X_interp[i],configuration,wing)
            vortex_cl[i] = 0.0
            #print wings_lift[i]
            
            # correct lift
            wings_lift_comp[i] = wings_lift[i] * compress_corr[i]                
            

        
        # total lift, accounting one fuselage
        aircraft_lift_total[i] = wings_lift_comp[i] * fus_correction 
    
    #print("From compute_aircraft_lift_supersonic")
    #print aircraft_lift_total
    #raw_input()
    # store results
    lift_results = Result(
        total                = aircraft_lift_total ,
        incompressible_wings = wings_lift          ,
        compressible_wings   = wings_lift_comp     ,
        compressibility_correction_factor = compress_corr  ,
        fuselage_correction_factor        = fus_correction ,
        vortex                            = vortex_cl ,
    )
    try:
        conditions.aerodynamics.lift_breakdown.update( lift_results )    #update
        
        conditions.aerodynamics.lift_coefficient= aircraft_lift_total
    except(AttributeError):
        print("Drag Polar Mode")

    return aircraft_lift_total


if __name__ == '__main__':   
    #test()
    raise RuntimeError , 'module test failed, not implemented'


#-------runn this caase  - have a local test case---------------------

