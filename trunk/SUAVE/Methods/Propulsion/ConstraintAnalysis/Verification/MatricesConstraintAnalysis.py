import SUAVE
import numpy as np
import matplotlib as plt
from SUAVE.Core import Data
from SUAVE.Methods.Propulsion.ConstraintAnalysis.Verification.Parameters import *
from SUAVE.Methods.Propulsion.ConstraintAnalysis.Verification.ConstraintAnalysisEquations import *

### EFFICIENCY AND POWER MANAGEMENT PARAMETERS

# Takeoff
eta_valve_TO = 0.99
eta_gasturbine_1_TO = 0.5
eta_gasturbine_2_TO = 0.4
eta_powermanagement_1_TO = 0.98
eta_powermanagement_2_TO = 0.98
eta_electricmotor_TO = 0.9
eta_propulsive_1_TO = 0.8
eta_propulsive_2_TO = 0.8
phi_TO=0.7
psi_TO=1
lambda_TO=0.1

# Climb
eta_valve_climb = 0.99
eta_gasturbine_1_climb = 0.5
eta_gasturbine_2_climb = 0.4
eta_powermanagement_1_climb = 0.98
eta_powermanagement_2_climb = 0.98
eta_electricmotor_climb = 0.9
eta_propulsive_1_climb = 0.8
eta_propulsive_2_climb = 0.8
phi_climb=0.7
psi_climb=1
lambda_climb=0.1

# Cruise
eta_valve_cruise = 0.99
eta_gasturbine_1_cruise = 0.5
eta_gasturbine_2_cruise = 0.4
eta_powermanagement_1_cruise = 0.98
eta_powermanagement_2_cruise = 0.98
eta_electricmotor_cruise = 0.9
eta_propulsive_1_cruise = 0.8
eta_propulsive_2_cruise = 0.8
phi_cruise=0.7
psi_cruise=1
lambda_cruise=0.1

# Descent
eta_valve_descent = 0.99
eta_gasturbine_1_descent = 0.5
eta_gasturbine_2_descent = 0.4
eta_powermanagement_1_descent = 0.98
eta_powermanagement_2_descent = 0.98
eta_electricmotor_descent = 0.9
eta_propulsive_1_descent = 0.8
eta_propulsive_2_descent = 0.8
phi_descent=0.7
psi_descent=1
lambda_descent=0.1

# Landing
eta_valve_landing = 0.99
eta_gasturbine_1_landing = 0.5
eta_gasturbine_2_landing = 0.4
eta_powermanagement_1_landing = 0.98
eta_powermanagement_2_landing = 0.98
eta_electricmotor_landing = 0.9
eta_propulsive_1_landing = 0.8
eta_propulsive_2_landing = 0.8
phi_landing=0.7
psi_landing=1
lambda_landing=0.1

def matrices_constraint_analysis():
    # Matrices with no system failure
    matrix_takeoff=np.array([-eta_valve_TO, 1, 1, 0, 0,0,0,0,0,0,0,0,0],[0, -eta_gasturbine_1_TO, 0, 1, 0,-eta_gasturbine_2_TO,0,0,0,0,0,0,0],[0, 0, 0, -eta_propulsive_1_TO, 1,0,0,0,0,0,0,0,0],[0, 0, -1, 0, 0,1,1,0,0,0,0,0,0],[0, 0, 0, 0, 0,0,-eta_powermanagement_1_TO,-eta_powermanagement_1_TO,1,0,0,0,0],[0, 0, 0, 0, 0,0,0,0,-eta_powermanagement_2_TO,1,1,0,0],[0, 0, 0, 0, 0,0,0,0,0,0,-eta_electricmotor_TO,1,0],[0, 0, 0, 0, 0,0,0,0,0,0,0,-eta_propulsive_2_TO,1],[0, 1-phi_TO, phi_TO, 0, 0,0,0,0,0,0,0,0,0],[0, 0, 0, 0, 0,0,1-psi_TO,psi_TO,0,0,0,0,0],[0, 0, 0, 0, 0,0,0,0,0,1-lambda_TO,lambda_TO,0,0],[0, 0, -eta_SOFC_TO, 0, 0,0,1,0,0,0,0,0,0],[0, 0, 0, 0, 1,0,0,0,0,0,0,0,1])
    matrix_climb=np.array([-eta_valve_climb, 1, 1, 0, 0,0,0,0,0,0,0,0,0],[0, -eta_gasturbine_1_climb, 0, 1, 0,-eta_gasturbine_2_climb,0,0,0,0,0,0,0],[0, 0, 0, -eta_propulsive_1_climb, 1,0,0,0,0,0,0,0,0],[0, 0, -1, 0, 0,1,1,0,0,0,0,0,0],[0, 0, 0, 0, 0,0,-eta_powermanagement_1_climb,-eta_powermanagement_1_climb,1,0,0,0,0],[0, 0, 0, 0, 0,0,0,0,-eta_powermanagement_2_climb,1,1,0,0],[0, 0, 0, 0, 0,0,0,0,0,0,-eta_electricmotor_climb,1,0],[0, 0, 0, 0, 0,0,0,0,0,0,0,-eta_propulsive_2_climb,1],[0, 1-phi_climb, phi_climb, 0, 0,0,0,0,0,0,0,0,0],[0, 0, 0, 0, 0,0,1-psi_climb,psi_climb,0,0,0,0,0],[0, 0, 0, 0, 0,0,0,0,0,1-lambda_climb,lambda_climb,0,0],[0, 0, -eta_SOFC_climb, 0, 0,0,1,0,0,0,0,0,0],[0, 0, 0, 0, 1,0,0,0,0,0,0,0,1])
    matrix_cruise=np.array([-eta_valve_cruise, 1, 1, 0, 0,0,0,0,0,0,0,0,0],[0, -eta_gasturbine_1_cruise, 0, 1, 0,-eta_gasturbine_2_cruise,0,0,0,0,0,0,0],[0, 0, 0, -eta_propulsive_1_cruise, 1,0,0,0,0,0,0,0,0],[0, 0, -1, 0, 0,1,1,0,0,0,0,0,0],[0, 0, 0, 0, 0,0,-eta_powermanagement_1_cruise,-eta_powermanagement_1_cruise,1,0,0,0,0],[0, 0, 0, 0, 0,0,0,0,-eta_powermanagement_2_cruise,1,1,0,0],[0, 0, 0, 0, 0,0,0,0,0,0,-eta_electricmotor_cruise,1,0],[0, 0, 0, 0, 0,0,0,0,0,0,0,-eta_propulsive_2_cruise,1],[0, 1-phi_cruise, phi_cruise, 0, 0,0,0,0,0,0,0,0,0],[0, 0, 0, 0, 0,0,1-psi_cruise,psi_cruise,0,0,0,0,0],[0, 0, 0, 0, 0,0,0,0,0,1-lambda_cruise,lambda_cruise,0,0],[0, 0, -eta_SOFC_cruise, 0, 0,0,1,0,0,0,0,0,0],[0, 0, 0, 0, 1,0,0,0,0,0,0,0,1])
    matrix_descent=np.array([-eta_valve_descent, 1, 1, 0, 0,0,0,0,0,0,0,0,0],[0, -eta_gasturbine_1_descent, 0, 1, 0,-eta_gasturbine_2_descent,0,0,0,0,0,0,0],[0, 0, 0, -eta_propulsive_1_descent, 1,0,0,0,0,0,0,0,0],[0, 0, -1, 0, 0,1,1,0,0,0,0,0,0],[0, 0, 0, 0, 0,0,-eta_powermanagement_1_descent,-eta_powermanagement_1_descent,1,0,0,0,0],[0, 0, 0, 0, 0,0,0,0,-eta_powermanagement_2_descent,1,1,0,0],[0, 0, 0, 0, 0,0,0,0,0,0,-eta_electricmotor_descent,1,0],[0, 0, 0, 0, 0,0,0,0,0,0,0,-eta_propulsive_2_descent,1],[0, 1-phi_descent, phi_descent, 0, 0,0,0,0,0,0,0,0,0],[0, 0, 0, 0, 0,0,1-psi_descent,psi_descent,0,0,0,0,0],[0, 0, 0, 0, 0,0,0,0,0,1-lambda_descent,lambda_descent,0,0],[0, 0, -eta_SOFC_descent, 0, 0,0,1,0,0,0,0,0,0],[0, 0, 0, 0, 1,0,0,0,0,0,0,0,1])
    matrix_landing=np.array([-eta_valve_landing, 1, 1, 0, 0,0,0,0,0,0,0,0,0],[0, -eta_gasturbine_1_landing, 0, 1, 0,-eta_gasturbine_2_landing,0,0,0,0,0,0,0],[0, 0, 0, -eta_propulsive_1_landing, 1,0,0,0,0,0,0,0,0],[0, 0, -1, 0, 0,1,1,0,0,0,0,0,0],[0, 0, 0, 0, 0,0,-eta_powermanagement_1_landing,-eta_powermanagement_1_landing,1,0,0,0,0],[0, 0, 0, 0, 0,0,0,0,-eta_powermanagement_2_landing,1,1,0,0],[0, 0, 0, 0, 0,0,0,0,0,0,-eta_electricmotor_landing,1,0],[0, 0, 0, 0, 0,0,0,0,0,0,0,-eta_propulsive_2_landing,1],[0, 1-phi_landing, phi_landing, 0, 0,0,0,0,0,0,0,0,0],[0, 0, 0, 0, 0,0,1-psi_landing,psi_landing,0,0,0,0,0],[0, 0, 0, 0, 0,0,0,0,0,1-lambda_landing,lambda_landing,0,0],[0, 0, -eta_SOFC_landing, 0, 0,0,1,0,0,0,0,0,0],[0, 0, 0, 0, 1,0,0,0,0,0,0,0,1])

    # Matrices with system failure
