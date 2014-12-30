# Tim Momose, October 2014

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import pylab as plt
# SUAVE Imports
from SUAVE.Structure  import Data
from SUAVE.Attributes import Units
from full_setup_737800 import full_setup_737800
# SUAVE-AVL Imports
from SUAVE.Methods.Aerodynamics.AVL.initialize_inputs import initialize_inputs
from SUAVE.Methods.Aerodynamics.AVL.run_analysis      import run_analysis
from SUAVE.Methods.Aerodynamics.AVL.read_results      import read_results
from SUAVE.Methods.Aerodynamics.AVL.purge_files       import purge_files

# -------------------------------------------------------------
#  Test Script
# -------------------------------------------------------------

# Setup test conditions < Make sure this mimics SUAVE Conditions() properly
conditions = Data()
conditions.freestream = Data()
conditions.freestream.mach     = 0.2
conditions.freestream.velocity = 150 * Units.knots
conditions.density = 1.225
conditions.g       = 9.81

vehicle,mission = full_setup_737800()
configuration = vehicle.configs['takeoff']

avl_inputs = initialize_inputs(vehicle,configuration,conditions)

# unpack inputs
avl_cases         = avl_inputs.cases
files_path        = avl_inputs.input_files.reference_path
results_files     = avl_inputs.input_files.results
geometry_filename = avl_inputs.input_files.geometry
cases_filename    = avl_inputs.input_files.cases
deck_filename     = avl_inputs.input_files.deck

run_analysis(avl_inputs)

# Results
CLCD   = Data()
num_cases = avl_cases.num_cases
for i in range(num_cases):
	filename = results_files[i]
	res = read_results(files_path,filename)
	CLCD.append(res)

plt.figure('Drag Polar')
axes = plt.gca()
CL = []
CD = []
for res in CLCD:
	CL.append(res.CL_total)
	CD.append(res.CD_total)
axes.plot(CD,CL,'bo-')
axes.set_xlabel('Total Drag Coefficient')
axes.set_ylabel('Total Lift Coefficient')
axes.grid(True)
plt.show()

#Purge old results and input files
results_files.extend([deck_filename]) #geometry_filename,cases_filename,
purge_files(results_files,files_path)
