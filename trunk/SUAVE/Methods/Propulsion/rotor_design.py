## @ingroup Methods-Propulsion
# propeller_design.py
# 
# Created:  Jan 2022, E. Botero
# Modified: 


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
import numpy as np
import scipy as sp
from SUAVE.Core import Units , Data
from scipy import optimize

from . import propeller_design
from SUAVE.Plots.Geometry import plot_propeller

import matplotlib.pyplot as plt    


# ----------------------------------------------------------------------
#  Propeller Design
# ----------------------------------------------------------------------

def rotor_design(rotor, number_of_stations=20):
    """ Optimizes propeller chord and twist given input parameters.
          
          Inputs:
          Either design power or thrust
          prop_attributes.
            hub radius                       [m]
            tip radius                       [m]
            rotation rate                    [rad/s]
            hover_speed                      [m/s]
            forward_speed                    [m/s]
            altitude                         [m]
            number of blades               
            number of stations
            design lift coefficient
            airfoil data
            
          Outputs:
          Twist distribution                 [array of radians]
          Chord distribution                 [array of meters]
              
          Assumptions/ Source:
          
    """    

    # Do a MIL design using propeller design. This will set a baseline
    rotor.freestream_velocity = np.sqrt(rotor.hover_speed**2 + rotor.forward_speed**2)
    
    #plot_propeller(rotor)
    
    #plt.show()
    
    # Setup conditions
    atmosphere            = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere_conditions = atmosphere.compute_values(rotor.design_altitude)    
    
    conditions                                          = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    conditions._size                                    = 1
    conditions.freestream                               = Data()
    conditions.propulsion                               = Data()
    conditions.frames                                   = Data()
    conditions.frames.body                              = Data()
    conditions.frames.inertial                          = Data()
    conditions.freestream.update(atmosphere_conditions)
    conditions.freestream.dynamic_viscosity             = atmosphere_conditions.dynamic_viscosity
    conditions.frames.inertial.velocity_vector          = np.array([[rotor.forward_speed,0,rotor.hover_speed]])
    conditions.propulsion.throttle                      = np.array([[1.0]])
    conditions.frames.body.transform_to_inertial        = np.array([np.eye(3)])
    
    rotor.inputs.omega                                  = np.array(rotor.angular_velocity,ndmin=2)
    
    
    # Setup variables and bounds
    variables = np.concatenate([rotor.twist_distribution,rotor.chord_distribution])    
    lb        = np.concatenate([np.ones_like(rotor.twist_distribution)*-np.inf,np.ones_like(rotor.twist_distribution)*1e-3])
    ub        = np.concatenate([np.ones_like(rotor.twist_distribution)*np.inf,np.ones_like(rotor.twist_distribution)*np.inf])
    #bnds      = optimize.Bounds(lb, ub,keep_feasible=True)
    bnds      = np.zeros((2*number_of_stations,2))
    
    for ii in range(2*number_of_stations):
        bnds[ii] = (lb[ii]),(ub[ii])
        
    
    # setup a wrapper to passs to the optimizer
    def optimization_wrapper(variables):
        
        # set the variables
        rotor.twist_distribution = variables[:number_of_stations]
        rotor.chord_distribution = variables[number_of_stations:]
        
        thrust_vector, torque, power, Cp, outputs , etap = rotor.spin(conditions)

        objective = - outputs.figure_of_merit[0,0]
        
        print(objective)

        return objective    
    
    # setup a wrapper to passs to the optimizer
    def constraint_wrapper(variables):
        
        # set the variables
        rotor.twist_distribution = variables[:number_of_stations]
        rotor.chord_distribution = variables[number_of_stations:]
        
        thrust_vector, torque, power, Cp, outputs , etap = rotor.spin(conditions)

        constraint = np.linalg.norm(thrust_vector) - rotor.design_thrust
        
        return constraint      

    # Run an optimization case with FoM
    
    res =  sp.optimize.fmin_slsqp(optimization_wrapper,variables,f_ieqcons=constraint_wrapper,bounds=bnds)    
    #res = optimize.minimize(optimization_wrapper, variables, method='SLSQP', constraints=constraint_wrapper, bounds=bnds)
    
    rotor.twist_distribution = res[:number_of_stations]
    rotor.chord_distribution = res[number_of_stations:]    
    
    plot_propeller(rotor)
    
    print(res)
    
    return rotor
