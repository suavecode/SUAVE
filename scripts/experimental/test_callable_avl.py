# Tim Momose, January 2015
# Modified February 6, 2015

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import pylab as plt
import numpy as np
from copy import deepcopy
# SUAVE Imports
from SUAVE.Core        import Units
from full_setup_737800 import vehicle_setup
from SUAVE.Analyses.Missions.Segments.Conditions.Aerodynamics import Aerodynamics
# SUAVE-AVL Imports
from SUAVE.Methods.Aerodynamics.AVL.Data.Cases import Run_Case
from SUAVE.Analyses.Aerodynamics.AVL_Callable  import AVL_Callable


def main():
    
    # -------------------------------------------------------------
    #  Test Script
    # -------------------------------------------------------------
    
    # Time the process
    import time
    t0 = time.time()
    print "Start: " + time.ctime()
    
    # Set up test defaults
    vehicle        = vehicle_setup()
    #avl,base_case = setup_avl_test(vehicle)
    avl            = AVL_Callable()
    avl.keep_files = True
    avl.initialize(vehicle)    
    run_conditions = Aerodynamics()
    ones_1col       = run_conditions.ones_row(1)
    run_conditions.weights.total_mass = ones_1col*vehicle.mass_properties.max_takeoff
    run_conditions.freestream.mach_number = ones_1col * 0.2
    run_conditions.freestream.velocity    = ones_1col * 150 * Units.knots
    run_conditions.freestream.density     = ones_1col * 1.225
    run_conditions.freestream.gravity     = ones_1col * 9.81
    run_conditions.aerodynamics.angle_of_attack = ones_1col * 0.0
    run_conditions.aerodynamics.side_slip_angle = ones_1col * 0.0
	    
    # Set up run cases
    alphas    = np.array([[-10],[-5],[-2],[0],[2],[5],[10],[20]])
    run_conditions.expand_rows(alphas.shape[0])
    run_conditions.aerodynamics.angle_of_attack = alphas
    #avl_cases = Run_Case.Container()
    #for alpha in alphas:
        #case = deepcopy(base_case)
        #case.tag = ('alpha_{}'.format(alpha)).replace('-','neg')
        #case.conditions.aerodynamics.angle_of_attack = alpha
        #avl_cases.append_case(case)
    
    results = avl(run_conditions)
    
    # Results
    plt.figure('Drag Polar')
    axes = plt.gca()
    CL = results.aerodynamics.lift_coefficient
    CD = results.aerodynamics.drag_coefficient
    CM = results.aerodynamics.pitch_moment_coefficient
    axes.plot(CD,CL,'bo-')
    axes.set_xlabel('Total Drag Coefficient')
    axes.set_ylabel('Total Lift Coefficient')
    axes.grid(True)
    
    plt.figure('Pitching Momoent')
    axes = plt.gca()
    axes.plot(alphas,CM,'bo-')
    axes.set_xlabel('Angle of Attack')
    axes.set_ylabel('Pitching Moment')
    axes.grid(True)
    
    tf = time.time()
    print "End:   " + time.ctime()
    print "({0:.2f} seconds)".format(tf-t0)
    
    plt.show()
    
    return


# -------------------------------------------------------------
#  Setup function
# -------------------------------------------------------------

def setup_avl_test(vehicle):

    default_case = Run_Case()
    default_case.conditions.freestream.mach     = 0.2
    default_case.conditions.freestream.velocity = 150 * Units.knots
    default_case.conditions.aerodynamics.parasite_drag = 0.0177

    avl_instance = AVL_Callable()
    avl_instance.keep_files = True
    avl_instance.initialize(vehicle)

    for wing in vehicle.wings:
        for cs in wing.control_surfaces:
            default_case.append_control_deflection(cs.tag,0.0) # default all control surfaces to zero deflection

    return avl_instance, default_case


# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
