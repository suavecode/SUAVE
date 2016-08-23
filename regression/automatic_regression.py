# automatic_regression.py
# 
# Created:  Jun 2014, T. Lukaczyk
# Modified: Jun 2014, SUAVE Team

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  
        
import matplotlib 
matplotlib.use('Agg') 
        
import SUAVE
from SUAVE.Core.DataOrdered import DataOrdered
import sys, os, traceback, time
import matplotlib.pyplot as plt

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
    
    # regression
    'scripts/atmosphere/atmosphere.py',
    'scripts/dynamic_stability/dynamicstability.py',
    'scripts/weights/weights.py',
    'scripts/DC_10_noise/DC_10_noise.py',
    'scripts/B737/mission_B737.py',
    'scripts/Embraer_E190_constThr/mission_Embraer_E190_constThr.py',
    'scripts/concorde/concorde.py',
    #'regression/test_mission_AS2.py',
    'scripts/landing_field_length/landing_field_length.py',
    'scripts/take_off_field_length/take_off_field_length.py',
    'scripts/solar_network/solar_network.py',
    'scripts/solar_radiation/solar_radiation.py',
    'scripts/propeller/propeller.py',
    'scripts/aerodynamics/aerodynamics.py',
    #'scripts/aerodynamics_super/aerodynamics_super.py',
    'scripts/battery/battery.py',
    'scripts/cmalpha/cmalpha.py',
    'scripts/cnbeta/cnbeta.py',
    'scripts/gasturbine_network/gasturbine_network.py',
    'scripts/payload_range/payload_range.py',
    'scripts/test_input_output/test_xml_read_write.py',
    'scripts/test_input_output/test_freemind_write.py',
    'scripts/variable_cruise_distance/variable_cruise_distance.py',
    'scripts/noise_optimization/Noise_Test.py',
    'scripts/Regional_Jet_Optimization/Optimize2.py',
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
    for module,result in results.items():
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
            raise ImportError, 'file %s does not exist' % module_name
        
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