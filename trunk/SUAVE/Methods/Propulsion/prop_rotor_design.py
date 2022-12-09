## @ingroup Methods-Propulsion
# prop_rotor_design.py 
#
# Created: Feb 2022, M. Clarke

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  
# SUAVE Imports 
import SUAVE 
from SUAVE.Core                                                                              import Units, Data  
from SUAVE.Analyses.Mission.Segments.Segment                                                 import Segment 
from SUAVE.Methods.Noise.Fidelity_One.Propeller.propeller_mid_fidelity                       import propeller_mid_fidelity
import SUAVE.Optimization.Package_Setups.scipy_setup                                         as scipy_setup 
from SUAVE.Optimization                                                                      import Nexus       
from SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics                                 import Aerodynamics 
from SUAVE.Analyses.Process                                                                  import Process   

from SUAVE.Methods.Propulsion.Rotor_Design.optimization_setup           import optimization_setup
from SUAVE.Methods.Propulsion.Rotor_Design.set_optimized_rotor_planform import set_optimized_rotor_planform

# Python package imports  
from numpy import linalg as LA  
import numpy as np 
import scipy as sp 
import time 

# ----------------------------------------------------------------------
#  Rotor Design
# ----------------------------------------------------------------------
## @ingroup Methods-Propulsion
def prop_rotor_design(rotor,number_of_stations = 20,solver_name= 'SLSQP',iterations = 500,
                      solver_sense_step = 1E-3,solver_tolerance = 1E-2,print_iterations = False):  
    """ Optimizes prop-rotor chord and twist given input parameters to meet either design power or thurst. 
        This scrip adopts SUAVE's native optimization style where the objective function is expressed 
        as an aeroacoustic function, considering both efficiency and radiated noise.
          
          Inputs: 
          prop_attributes.
              hub radius                       [m]
              tip radius                       [m]
              rotation rate                    [rad/s]
              freestream velocity              [m/s]
              number of blades                 [None]       
              number of stations               [None]
              design lift coefficient          [None]
              airfoil data                     [None]
              optimization_parameters.
                 slack_constaint               [None]
                 ideal_SPL_dbA                 [dBA]
                 aeroacoustic_weight           [None]
            
          Outputs:
          Twist distribution                   [array of radians]
          Chord distribution                   [array of meters]
              
          Assumptions: 
             N/A 
        
          Source:
             None 
    """    

    # Unpack rotor geometry  
    rotor_tag     = rotor.tag
    rotor.tag     = 'rotor'
    
    # start optimization 
    ti                   = time.time()   
    optimization_problem = optimization_setup(rotor,number_of_stations,print_iterations)
    output               = scipy_setup.SciPy_Solve(optimization_problem,solver=solver_name, iter = iterations , sense_step = solver_sense_step,tolerance = solver_tolerance)    
    tf                   = time.time()
    elapsed_time         = round((tf-ti)/60,2)
    print('Lift-rotor Optimization Simulation Time: ' + str(elapsed_time) + ' mins')   
    
    # print optimization results 
    print (output)  
    
    # set remaining rotor variables using optimized parameters 
    rotor     = set_optimized_rotor_planform(rotor,optimization_problem)
    rotor.tag = rotor_tag
     
    return rotor
   