# propeller.py
# 
# Created:  E. Botero, Sep 2014

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

from SUAVE.Core import (
Data, Container,
)

import numpy as np
import copy, time
from SUAVE.Methods.Propulsion import propeller_design

def main():
    
    # This script could fail if either the design or analysis scripts fail,
    # in case of failure check both. The design and analysis powers will 
    # differ because of karman-tsien compressibility corrections in the 
    # analysis scripts
    
    # Design the Propeller
    prop                     = SUAVE.Components.Energy.Converters.Propeller()
    prop.number_blades       = 2.0 
    prop.freestream_velocity = 50.0
    prop.angular_velocity    = 2000.*(2.*np.pi/60.0)
    prop.tip_radius          = 1.5
    prop.hub_radius          = 0.05
    prop.design_Cl           = 0.7 
    prop.design_altitude     = 0.0 * Units.km
    prop.design_thrust       = 0.0
    prop.design_power        = 7000.
    prop                     = propeller_design(prop)    

    # Design a Rotor now
    rot  = SUAVE.Components.Energy.Converters.Rotor()
    rot.number_blades          = 2.0 
    rot.freestream_velocity    = 0#.1*Units.ft/Units.second
    rot.angular_velocity       = 2000.*(2.*np.pi/60.0)
    rot.tip_radius             = 1.5
    rot.hub_radius             = 0.05
    rot.design_Cl              = 0.7 
    rot.design_altitude        = 0.0 * Units.km
    rot.design_thrust          = 1000.0
    rot.induced_hover_velocity = 13.5 #roughly equivalent to a Chinook at SL

    rot  = propeller_design(prop) 
    
    # Find the operating conditions
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere_conditions =  atmosphere.compute_values(prop.design_altitude)
    
    V  = prop.freestream_velocity
    Vr = rot.freestream_velocity
    
    conditions = Data()
    conditions.freestream = Data()
    conditions.propulsion = Data()
    conditions.frames     = Data()
    conditions.frames.body     = Data()
    conditions.frames.inertial = Data()
    conditions.freestream.update(atmosphere_conditions)
    conditions.freestream.dynamic_viscosity = atmosphere_conditions.dynamic_viscosity
    conditions.frames.inertial.velocity_vector = np.array([[V,0,0]])
    conditions.propulsion.throttle = np.array([[1.0]])
    conditions.frames.body.transform_to_inertial = np.array([np.eye(3)])
    
    conditions_r = copy.deepcopy(conditions)
    conditions.frames.inertial.velocity_vector   = np.array([[V,0,0]])
    conditions_r.frames.inertial.velocity_vector = np.array([[0,Vr,0]])
    
    # Create and attach this propeller 
    prop.inputs.omega    = np.array(prop.angular_velocity,ndmin=2)
    rot.inputs.omega     = copy.copy(prop.inputs.omega)
    
    F, Q, P, Cplast ,output , etap       = prop.spin(conditions)
    Fr, Qr, Pr, Cplastr ,outputr , etapr = rot.spin(conditions_r)

    
    # Truth values
    F_truth      = 103.38703422
    Q_truth      = 29.79226982
    P_truth      = 6239.67840083
    Cplast_truth = 0.00056596
    
    Fr_truth      = 98.33229685
    Qr_truth      = 1.8647076
    Pr_truth      = 390.5434467
    Cplastr_truth = 3.54234419e-05
    
    error = Data()
    error.Thrust   = np.max(np.abs(F-F_truth))
    error.Power    = np.max(np.abs(P-P_truth))
    error.Torque   = np.max(np.abs(Q-Q_truth))
    error.Cp       = np.max(np.abs(Cplast-Cplast_truth))   
    error.Thrustr  = np.max(np.abs(Fr-Fr_truth))
    error.Powerr   = np.max(np.abs(Pr-Pr_truth))
    error.Torquer  = np.max(np.abs(Qr-Qr_truth))
    error.Cpr      = np.max(np.abs(Cplastr-Cplastr_truth)) 
    
    print('Errors:')
    print(error)
    
    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)
     
    return

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()