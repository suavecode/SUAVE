# automatic_regression.py
#
# Created:  Jun 2014, T. Lukaczyk
# Modified: Jun 2014, SUAVE Team
#           Jul 2017, SUAVE Team
#           Jan 2018, SUAVE Team
#           May 2019, T. MacDonald
#           Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import SUAVE
from SUAVE.Core.DataOrdered import DataOrdered
import sys, os, traceback, time


# ----------------------------------------------------------------------
#   How This Works
# ----------------------------------------------------------------------

# The "modules" list contains the name of the file you would like to run.
# Each test script must include a main function, this will be called by
# this automatic regression script.
#
# For more information, see ../templates/example_test_script.py

# ----------------------------------------------------------------------
#   The Modules to Test
# ----------------------------------------------------------------------

modules = [

    # ----------------------- Regression List --------------------------
    'scripts/aerodynamics/aerodynamics.py',
    'scripts/aerodynamics/all_moving_surfaces_vlm.py',
    'scripts/aerodynamics/control_surfaces_vlm.py',
    'scripts/aerodynamics/sideslip_and_rotation_vlm.py',
    'scripts/airfoil_import/airfoil_import_test.py',
    'scripts/airfoil_import/airfoil_interpolation_test.py',
    'scripts/airfoil_analysis/airfoil_panel_method_test.py',  
    'scripts/atmosphere/atmosphere.py',
    'scripts/atmosphere/constant_temperature.py',
    'scripts/AVL/test_AVL.py',
    'scripts/B737/mission_B737.py', 
    'scripts/battery/aircraft_discharge_comparisons.py',
    'scripts/battery/battery_cell_discharge_tests.py', 
    'scripts/cmalpha/cmalpha.py',
    'scripts/cnbeta/cnbeta.py',
    'scripts/concorde/concorde.py',
    'scripts/ducted_fan/ducted_fan_network.py',
    'scripts/ducted_fan/battery_ducted_fan_network.py',
    'scripts/ducted_fan/serial_hybrid_ducted_fan_network.py',
    'scripts/dynamic_stability/dynamicstability.py',
    'scripts/electric_performance/propeller_single_point.py',
    'scripts/electric_performance/electric_V_h_diagram.py',
    'scripts/electric_performance/electric_payload_range.py',
    'scripts/Embraer_E190_constThr/mission_Embraer_E190_constThr.py',
    'scripts/fuel_cell/fuel_cell.py',
    'scripts/gasturbine_network/gasturbine_network.py',
    'scripts/geometry/NACA_airfoil_compute.py',
    'scripts/geometry/NACA_volume_compute.py',
    'scripts/geometry/wing_fuel_volume_compute.py',
    'scripts/geometry/fuselage_planform_compute.py',
    'scripts/industrial_costs/industrial_costs.py',
    'scripts/internal_combustion_propeller/ICE_Test.py',
    'scripts/internal_combustion_propeller/ICE_CS_Test.py',
    'scripts/lifting_line/lifting_line.py',
    'scripts/mission_range_and_weight_sizing/landing_field_length.py',
    'scripts/mission_range_and_weight_sizing/take_off_field_length.py',
    'scripts/mission_range_and_weight_sizing/take_off_weight_from_tofl.py',
    'scripts/motor/motor_test.py',
    'scripts/multifidelity/optimize_mf.py',
    'scripts/noise_optimization/Noise_Test.py', 
    'scripts/noise_fidelity_zero/DC_10_noise.py', 
    'scripts/noise_fidelity_one/propeller_noise.py',  
    'scripts/noise_fidelity_one/aircraft_noise.py',  
    'scripts/nonuniform_propeller_inflow/nonuniform_propeller_inflow.py',
    'scripts/optimization_packages/optimization_packages.py',
    'scripts/payload_range/payload_range.py',
    'scripts/plots/plot_test.py',
    'scripts/propeller/propeller_test.py',
    'scripts/propeller_speeds/range_endurance_speeds.py',
    'scripts/propulsion_surrogate/propulsion_surrogate.py',
    'scripts/ramjet_network/ramjet_network.py',
    'scripts/Regional_Jet_Optimization/Optimize2.py',
    'scripts/scramjet_network/scramjet_network.py',
    'scripts/rocket_network/Rocketdyne_F1.py',
    'scripts/rocket_network/Rocketdyne_J2.py',
    'scripts/segments/segment_test.py',
    'scripts/slipstream/slipstream_test.py',
    'scripts/slipstream/propeller_interactions.py',
    'scripts/solar_network/solar_network.py',
    'scripts/solar_network/solar_low_fidelity_network.py',
    'scripts/solar_radiation/solar_radiation.py',
    'scripts/aerodynamics/sears_haack.py',
    'scripts/SU2_surrogate/BWB-450.py',
    'scripts/sweeps/test_sweeps.py',
    'scripts/test_input_output/test_xml_read_write.py',
    'scripts/test_input_output/test_freemind_write.py',
    'scripts/variable_cruise_distance/variable_cruise_distance.py',
    'scripts/V_n_diagram/V_n_diagram_regression.py',
    'scripts/VTOL/test_Multicopter.py',
    'scripts/VTOL/test_Tiltwing.py',
    'scripts/VTOL/test_Stopped_Rotor.py',
    'scripts/weights/weights.py'
]

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():

    # preallocate test results
    results = DataOrdered()
    for module in modules:
        results[module] = 'Untested'

    sys.stdout.write('# --------------------------------------------------------------------- \n')
    sys.stdout.write('#   SUAVE Automatic Regression \n')
    sys.stdout.write('#   %s \n' % time.strftime("%B %d, %Y - %H:%M:%S", time.gmtime()) )
    sys.stdout.write('# --------------------------------------------------------------------- \n')
    sys.stdout.write(' \n')

    # run tests
    all_pass = True
    for module in modules:
        passed = test_module(module)
        if passed:
            results[module] = '  Passed'
        else:
            results[module] = '* FAILED'
            all_pass = False

    # final report
    sys.stdout.write('# --------------------------------------------------------------------- \n')
    sys.stdout.write('Final Results \n')
    for module,result in list(results.items()):
        sys.stdout.write('%s - %s\n' % (result,module))

    if all_pass:
        sys.exit(0)
    else:
        sys.exit(1)


# ----------------------------------------------------------------------
#   Module Tester
# ----------------------------------------------------------------------

def test_module(module_path):

    home_dir = os.getcwd()
    test_dir, module_name = os.path.split( os.path.abspath(module_path) )

    sys.stdout.write('# --------------------------------------------------------------------- \n')
    sys.stdout.write('# Start Test: %s \n' % module_path)
    sys.stdout.flush()

    tic = time.time()

    # try the test
    try:

        # see if file exists
        os.chdir(test_dir)
        if not os.path.exists(module_name) and not os.path.isfile(module_name):
            raise ImportError('file %s does not exist' % module_name)

        # add module directory
        sys.path.append(test_dir)

        # do the import
        name = os.path.splitext(module_name)[0]
        module = __import__(name)

        # run main function
        module.main()

        passed = True

    # catch an error
    except Exception as exc:

        # print traceback
        sys.stderr.write( 'Test Failed: \n' )
        sys.stderr.write( traceback.format_exc() )
        sys.stderr.write( '\n' )
        sys.stderr.flush()

        passed = False

    # final result
    if passed:
        sys.stdout.write('# Passed: %s \n' % module_name)
    else:
        sys.stdout.write('# FAILED: %s \n' % module_name)
    sys.stdout.write('# Test Duration: %.4f min \n' % ((time.time()-tic)/60) )
    sys.stdout.write('\n')

    # cleanup
    plt.close('all')
    os.chdir(home_dir)

    # make sure to write to stdout
    sys.stdout.flush()
    sys.stderr.flush()

    return passed

# ----------------------------------------------------------------------
#   Call Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()
