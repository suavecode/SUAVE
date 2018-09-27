# test_propeller.py
# 
# Created:  Emilio Botero, Sep 2014

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
    prop_attributes = Data()
    prop_attributes.number_blades       = 2.0 
    prop_attributes.freestream_velocity = 50.0
    prop_attributes.angular_velocity    = 2000.*(2.*np.pi/60.0)
    prop_attributes.tip_radius          = 1.5
    prop_attributes.hub_radius          = 0.05
    prop_attributes.design_Cl           = 0.7 
    prop_attributes.design_altitude     = 0.0 * Units.km
    prop_attributes.design_thrust       = 0.0
    prop_attributes.design_power        = 7000.
    prop_attributes                     = propeller_design(prop_attributes)    

    # Find the operating conditions
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere_conditions =  atmosphere.compute_values(prop_attributes.design_altitude)
    
    V = prop_attributes.freestream_velocity
    
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
    
    # Create and attach this propeller
    prop                 = SUAVE.Components.Energy.Converters.Propeller()
    prop.prop_attributes = prop_attributes    
    prop.inputs.omega    = np.array(prop_attributes.angular_velocity,ndmin=2)
    
    F, Q, P, Cplast = prop.spin(conditions)
    
    # Truth values
    F_truth      = 166.41590262
    Q_truth      = 45.21732911
    P_truth      = 9470.2952633 # Over 9000!
    Cplast_truth = 0.00085898
    
    error = Data()
    error.Thrust  = np.max(np.abs(F-F_truth))
    error.Power   = np.max(np.abs(P-P_truth))
    error.Torque  = np.max(np.abs(Q-Q_truth))
    error.Cp      = np.max(np.abs(Cplast-Cplast_truth))   
    
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