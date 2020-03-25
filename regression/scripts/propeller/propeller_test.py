# propeller_test.py
# 
# Created:  Sep 2014, E. Botero
# Modified: Feb 2020, M. Clarke  

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
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller

def main():
    
    # This script could fail if either the design or analysis scripts fail,
    # in case of failure check both. The design and analysis powers will 
    # differ because of karman-tsien compressibility corrections in the 
    # analysis scripts
    
    net                       = Battery_Propeller()   
    net.number_of_engines     = 2    
    
    # Design Gearbox 
    gearbox                   = SUAVE.Components.Energy.Converters.Gearbox()
    gearbox.gearwheel_radius1 = 1
    gearbox.gearwheel_radius2 = 1
    gearbox.efficiency        = 0.95
    gearbox.inputs.torque     = 800 # N-m
    gearbox.inputs.speed      = 209.43951023931953
    gearbox.inputs.power      = 7000 
    gearbox.compute()
    
    # Design the Propeller with airfoil  geometry defined 
    prop_a                         = SUAVE.Components.Energy.Converters.Propeller()
    prop_a.number_blades           = 2.0 
    prop_a.freestream_velocity     = 50.0
    prop_a.angular_velocity        = gearbox.inputs.speed # 209.43951023931953
    prop_a.tip_radius              = 1.5
    prop_a.hub_radius              = 0.05
    prop_a.design_Cl               = 0.7 
    prop_a.design_altitude         = 0.0 * Units.km 
    prop_a.airfoil_geometry        = ['NACA_4412_geo.txt']
    prop_a.airfoil_polars          = ['NACA_4412_polar.txt']
    prop_a.airfoil_polar_stations  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    prop_a.design_power            = gearbox.outputs.power # 7000  
    prop_a                         = propeller_design(prop_a)   
    
    # Design the Propeller with airfoil  geometry defined 
    prop                           = SUAVE.Components.Energy.Converters.Propeller()
    prop.number_blades             = 2.0 
    prop.freestream_velocity       = 50.0
    prop.angular_velocity          = gearbox.inputs.speed # 209.43951023931953
    prop.tip_radius                = 1.5
    prop.hub_radius                = 0.05
    prop.design_Cl                 = 0.7 
    prop.design_altitude           = 0.0 * Units.km 
    prop.design_power              = gearbox.outputs.power # 7000  
    prop                           = propeller_design(prop)  

    # Design a Rotor with airfoil  geometry defined 
    rot_a  = SUAVE.Components.Energy.Converters.Rotor()
    rot_a.number_blades            = 2.0 
    rot_a.freestream_velocity      = 1*Units.ft/Units.second
    rot_a.angular_velocity         = 2000.*(2.*np.pi/60.0)
    rot_a.tip_radius               = 1.5
    rot_a.hub_radius               = 0.05
    rot_a.design_Cl                = 0.7 
    rot_a.design_altitude          = 0.0 * Units.km
    rot_a.design_thrust            = 1000.0
    rot_a.airfoil_geometry         = ['NACA_4412_geo.txt']
    rot_a.airfoil_polars           = ['NACA_4412_polar.txt']
    rot_a.airfoil_polar_stations   = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]    
    rot_a.induced_hover_velocity   = 13.5 #roughly equivalent to a Chinook at SL
    rot_a                          = propeller_design(rot_a) 
    
    # Design a Rotor without airfoil geometry defined 
    rot  = SUAVE.Components.Energy.Converters.Rotor()
    rot.number_blades              = 2.0 
    rot.freestream_velocity        = 1*Units.ft/Units.second
    rot.angular_velocity           = 2000.*(2.*np.pi/60.0)
    rot.tip_radius                 = 1.5
    rot.hub_radius                 = 0.05
    rot.design_Cl                  = 0.7 
    rot.design_altitude            = 0.0 * Units.km
    rot.design_thrust              = 1000.0
    rot.induced_hover_velocity     = 13.5 #roughly equivalent to a Chinook at SL
    rot                            = propeller_design(rot) 
                                   
    # Find the operating conditions
    atmosphere            = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere_conditions =  atmosphere.compute_values(prop.design_altitude)
    
    V  = prop.freestream_velocity
    Vr = rot.freestream_velocity
    
    conditions                                          = Data()
    conditions.freestream                               = Data()
    conditions.propulsion                               = Data()
    conditions.frames                                   = Data()
    conditions.frames.body                              = Data()
    conditions.frames.inertial                          = Data()
    conditions.freestream.update(atmosphere_conditions)
    conditions.freestream.dynamic_viscosity             = atmosphere_conditions.dynamic_viscosity
    conditions.frames.inertial.velocity_vector          = np.array([[V,0,0]])
    conditions.propulsion.throttle                      = np.array([[1.0]])
    conditions.frames.body.transform_to_inertial        = np.array([np.eye(3)])
    
    conditions_r = copy.deepcopy(conditions)
    conditions.frames.inertial.velocity_vector   = np.array([[V,0,0]])
    conditions_r.frames.inertial.velocity_vector = np.array([[0,Vr,0]])
    
    # Create and attach this propeller 
    prop_a.inputs.omega  = np.array(prop.angular_velocity,ndmin=2)
    prop.inputs.omega    = np.array(prop.angular_velocity,ndmin=2)
    rot_a.inputs.omega   = copy.copy(prop.inputs.omega)
    rot.inputs.omega     = copy.copy(prop.inputs.omega)
    
    # propeller with airfoil results 
    F_a, Q_a, P_a, Cplast_a ,output_a , etap_a = prop_a.spin(conditions)
      
    # propeller without airfoil results 
    conditions.propulsion.pitch_command = np.array([[1.0]])*Units.degree
    F, Q, P, Cplast ,output , etap      = prop.spin_variable_pitch(conditions)
    
    # rotor with airfoil results 
    Fr_a, Qr_a, Pr_a, Cplastr_a ,outputr_a , etapr = rot_a.spin(conditions_r)
    
    # rotor with out airfoil results 
    conditions_r.propulsion.pitch_command = np.array([[1.0]])*Units.degree
    Fr, Qr, Pr, Cplastr ,outputr , etapr  = rot.spin_variable_pitch(conditions_r)
    
    # Truth values for propeller with airfoil geometry defined 
    F_a_truth       = 97.95830293
    Q_a_truth       = 28.28338146
    P_a_truth       = 5923.6575610
    Cplast_a_truth  = 0.00053729
    
    # Truth values for propeller without airfoil geometry defined 
    F_truth         = 179.27020984
    Q_truth         = 48.06170453
    P_truth         = 10066.0198578
    Cplast_truth    = 0.00091302
    
    # Truth values for rotor with airfoil geometry defined 
    Fr_a_truth      = 893.16859917
    Qr_a_truth      = 77.57705597
    Pr_a_truth      = 16247.70060823
    Cplastr_a_truth = 0.00147371
    
    # Truth values for rotor without airfoil geometry defined 
    Fr_truth        = 900.63698565
    Qr_truth        = 78.01972629
    Pr_truth        = 16340.41326374
    Cplastr_truth   = 0.00148212 
    
    # Store errors 
    error = Data()
    error.Thrust_a  = np.max(np.abs(F_a -F_a_truth))
    error.Torque_a  = np.max(np.abs(Q_a -Q_a_truth))    
    error.Power_a   = np.max(np.abs(P_a -P_a_truth))
    error.Cp_a      = np.max(np.abs(Cplast_a -Cplast_a_truth))  
    error.Thrust    = np.max(np.abs(F-F_truth))
    error.Torque    = np.max(np.abs(Q-Q_truth))    
    error.Power     = np.max(np.abs(P-P_truth))
    error.Cp        = np.max(np.abs(Cplast-Cplast_truth))  
    error.Thrustr_a = np.max(np.abs(Fr_a-Fr_a_truth))
    error.Torquer_a = np.max(np.abs(Qr_a-Qr_a_truth))    
    error.Powerr_a  = np.max(np.abs(Pr_a-Pr_a_truth))
    error.Cpr_a     = np.max(np.abs(Cplastr_a-Cplastr_a_truth))  
    error.Thrustr   = np.max(np.abs(Fr-Fr_truth))
    error.Torquer   = np.max(np.abs(Qr-Qr_truth))    
    error.Powerr    = np.max(np.abs(Pr-Pr_truth))
    error.Cpr       = np.max(np.abs(Cplastr-Cplastr_truth))     
    
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