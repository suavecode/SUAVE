# Tim Momose, January 2015
# Modified March 2015

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import pylab as plt
import numpy as np
from copy import deepcopy

import sys

# SUAVE Imports
from SUAVE.Core        import Units
from full_setup_737800 import vehicle_setup
from SUAVE.Analyses.Missions.Segments.Conditions import Aerodynamics

# SUAVE-AVL Imports
from SUAVE.Analyses.Aerodynamics  import AVL as AVL_Callable


def main():

    # -------------------------------------------------------------
    #  Test Script
    # -------------------------------------------------------------

    # Time the process
    import time
    t0 = time.time()
    print "Start: " + time.ctime()
    
    results = run_avl_test()
    alphas  = results.aerodynamics.angle_of_attack / Units.deg
    CL      = results.aerodynamics.lift_coefficient
    CDi     = results.aerodynamics.drag_breakdown.induced.total
    CM      = results.aerodynamics.pitch_moment_coefficient

    # Plot results
    plt.figure('Induced Drag Polar')
    axes = plt.gca()
    axes.plot(CDi,CL,'bo-')
    axes.set_xlabel('Total Drag Coefficient')
    axes.set_ylabel('Total Lift Coefficient')
    axes.grid(True)

    plt.figure('Pitching Momoent')
    axes = plt.gca()
    axes.plot(alphas,CM,'bo-')
    axes.set_xlabel('Angle of Attack [deg]')
    axes.set_ylabel('Pitching Moment')
    axes.grid(True)

    tf = time.time()
    print "End:   " + time.ctime()
    print "({0:.2f} seconds)".format(tf-t0)

    return


def run_avl_test():
    # Set up test defaults
    vehicle        = vehicle_setup()
    avl            = AVL_Callable()
    avl.features   = vehicle
    avl.keep_files = True
    avl.initialize()
    
    avl.settings.filenames.log_filename = sys.stdout
    avl.settings.filenames.err_filename = sys.stderr

        # set up conditions    
    run_conditions = Aerodynamics()
    run_conditions.weights.total_mass[0,0]     = vehicle.mass_properties.max_takeoff
    run_conditions.freestream.mach_number[0,0] = 0.2
    run_conditions.freestream.velocity[0,0]    = 150 * Units.knots
    run_conditions.freestream.density[0,0]     = 1.225
    run_conditions.freestream.gravity[0,0]     = 9.81

    # Set up run cases
    alphas    = np.array([-10,-5,-2,0,2,5,10,20]) * Units.deg
    run_conditions.expand_rows(alphas.shape[0])
    run_conditions.aerodynamics.angle_of_attack[:,0] = alphas

    results = avl.evaluate_conditions(run_conditions)
    
    return results


# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    plt.show(block=True)
