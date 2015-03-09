# Tim Momose, January 2015
# Modified February 6, 2015

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import pylab as plt
import numpy as np
from copy import deepcopy

import sys

# SUAVE Imports
from SUAVE.Core        import Units
from SUAVE.Analyses.Aerodynamics import AVL
from SUAVE.Analyses.Missions.Segments.Conditions import Aerodynamics,State
# SUAVE test script imports
from full_setup_737800 import vehicle_setup
from test_callable_avl import run_avl_test


def main():

    # -------------------------------------------------------------
    #  Test Script
    # -------------------------------------------------------------

    # Time the process
    import time
    t0 = time.time()
    print "Instantiating:    " + time.ctime()
    
    avl_surrogate = AVL()
    vehicle       = vehicle_setup()
    
    print "Initializing:     " + time.ctime()
    
    avl_surrogate.initialize(vehicle)
    
    print "Cross Validation: " + time.ctime()
    
    test_set = run_avl_test()
    state    = State()
    # IS THE STATE STRUCTURE ACTUALLY LIKE THIS??? (IF NOT, FIX THE UNPACKING APPROACH IN AVL.EVALUATE() AND AVL.EVALUATE_LIFT() ALSO)
    state.conditions.aerodynamics = Aerodynamics()
    state.conditions.aerodynamics.freestream.angle_of_attack = np.linspace(-20,20,50)
    surrogate_results = avl_surrogate.evaluate(state)
    
    # Plot results
    alphas_te = test_set.aerodynamics.angle_of_attack
    CL_te     = test_set.aerodynamics.lift_coefficient
    CDi_te    = test_set.aerodynamics.drag_breakdown.induced.total
    CM_te     = test_set.aerodynamics.pitch_moment_coefficient
    
    alphas_su = state.conditions.aerodynamics.freestream.angle_of_attack
    CL_su     = surrogate_results.lift_coefficient
    CDi_su    = surrogate_results.induced_drag_coefficient
    CM_su     = surrogate_results.pitch_moment_coefficient
    
    plt.figure('Induced Drag Polar')
    axes  = plt.gca()
    axes.plot(CDi_te,CL_te,'bo-')
    axes.plot(CDi_su,CL_su,'r--')
    axes.set_xlabel('Total Drag Coefficient')
    axes.set_ylabel('Total Lift Coefficient')
    axes.grid(True)
	   
    fig = plt.figure('Aerodynamic Coefficients')
    axes = fig.add_subplot(3,1,1)
    axes.set_title('Pitching Moment Coefficient')
    axes.plot(alphas_te,CL_te,'bo-')
    axes.plot(alphas_su,CL_su,'r--')
    axes.plot()
    axes.plot()
    axes.set_xlabel('Angle of Attack')
    axes.set_ylabel('Lift')
    axes.grid(True)
    axes = fig.add_subplot(3,1,2)
    axes.set_title('Induced Drag Coefficient')
    axes.plot(alphas_te,CDi_te,'bo-')
    axes.plot(alphas_su,CDi_su,'r--')
    axes.plot()
    axes.set_xlabel('Angle of Attack')
    axes.set_ylabel('Induced Drag')
    axes.grid(True)    
    axes = fig.add_subplot(3,1,3)
    axes.set_title('Pitching Moment Coefficient')
    axes.plot(alphas_te,CM_te,'bo-')
    axes.plot(alphas_su,CM_su,'r--')
    axes.plot()
    axes.set_xlabel('Angle of Attack')
    axes.set_ylabel('Pitching Moment')
    axes.grid(True)
    
    tf = time.time()
    print "Finished:         " + time.ctime()
    print "({0:.2f} seconds)".format(tf-t0)

    return


# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    plt.show()