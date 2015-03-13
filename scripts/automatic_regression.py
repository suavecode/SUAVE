# automatic_regression.py
# 
# Created:  Trent Lukaczyk, Jun 2014
# Modified:     

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  
        
import matplotlib 
matplotlib.use('Agg') 
        
import SUAVE
from SUAVE.Core import Ordered_Bunch
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
    'regression/test_atmosphere.py',
    'regression/test_dynamicstability.py',
    'regression/test_weights.py',
    'regression/DC_10_noise.py',
    'regression/test_mission_B737.py',
    'regression/test_mission_Embraer_E190_constThr.py',
    #'regression/test_mission_AS2.py',
    'regression/test_landing_field_length.py',
    'regression/test_take_off_field_length.py',
    #'regression/test_solar_network.py',
    'regression/test_solar_radiation.py',
    'regression/test_propeller.py',
    'regression/test_aerodynamics.py',
    'regression/test_aerodynamics_super.py',
    'regression/test_battery.py',
    'regression/test_cmalpha.py',
    'regression/test_cnbeta.py',
    'regression/test_gasturbine_network.py',
    'regression/test_payload_range.py',
    'regression/test_input_output/test_xml_read_write.py',
    'regression/test_input_output/test_freemind_write.py',
    'regression/test_variable_cruise_distance.py',
    
    # tutorials
    'tutorials/test_mission_B737.py',
    'tutorials/test_mission_Embraer_E190_constThr.py',
    
]


# ----------------------------------------------------------------------        
#   Main
# ----------------------------------------------------------------------    

def main():
    
    # preallocate test results
    results = Ordered_Bunch()
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