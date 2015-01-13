# Tim Momose, October 2014

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import pylab as plt
# SUAVE Imports
from SUAVE.Attributes import Units
from full_setup_737800 import full_setup_737800
# SUAVE-AVL Imports
from SUAVE.Methods.Aerodynamics.AVL.AVL_Data.AVL_Cases    import AVL_Run_Case, AVL_Cases
from SUAVE.Methods.Aerodynamics.AVL.AVL_Callable          import AVL_Callable


# -------------------------------------------------------------
#  Setup function
# -------------------------------------------------------------

def setup_avl_test(vehicle):
	#default_settings = AVL_Settings()
	
	default_case = AVL_Run_Case()
	default_case.conditions.freestream.mach     = 0.2
	default_case.conditions.freestream.velocity = 150 * Units.knots
	default_case.conditions.aerodynamics.parasite_drag = 0.0177
	
	avl_instance = AVL_Callable()
	avl_instance.initialize(vehicle,default_case)
	
	for wing in vehicle.wings:
		for cs in wing.control_surfaces:
			avl_instance.append_default_control(cs.tag,0.0) # default all control surfaces to zero deflection
	
	return avl_instance


# -------------------------------------------------------------
#  Test Script
# -------------------------------------------------------------

# Set up test defaults
vehicle,mission = full_setup_737800()
avl = setup_avl_test(vehicle)

# Set up run cases
alphas    = [-10,-5,-2,0,2,5,10,20]
avl_cases = AVL_Cases()
for alpha in alphas:
	case = AVL_Run_Case()
	case.tag = 'alpha={}'.format(alpha)
	case.conditions.aerodynamics.angle_of_attack = alpha
	avl_cases.append_case(case)
	#avl.append_cases(avl_cases)

results = avl(avl_cases)

# Results
plt.figure('Drag Polar')
axes = plt.gca()
CL = []
CD = []
CM = []
for res in results:
	CL.append(res.aerodynamics.total_lift_coefficient)
	CD.append(res.aerodynamics.total_drag_coefficient)
	CM.append(res.aerodynamics.pitch_moment_coefficient)
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

plt.show()



