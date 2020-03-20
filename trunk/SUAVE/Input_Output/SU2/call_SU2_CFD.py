## @ingroup Input_Output-SU2
# call_SU2_CFD.py
# 
# Created:  Oct 2016, T. MacDonald
# Modified: Jan 2017, T. MacDonald
#           Mar 2018, T. MacDonald

import subprocess
from SUAVE.Core import Data
import sys, os

## @ingroup Input_Output-SU2
def call_SU2_CFD(tag,parallel=False,processors=1):
    """This calls SU2 to perform an analysis according to the related .cfg file.

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    tag                          <string>  This determines what .cfg is used and what the output file is called.
    parallel   (optional)        <boolean> This determines if SU2 will be run in parallel. This setting requires that SU2 has been built to allow this.
    processors (optional)        [-]       The number of processors used for a parallel computation.

    Outputs:
    <tag>_history.dat            This file has the SU2 convergence history.
    CL                           [-]
    CD                           [-]

    Properties Used:
    N/A
    """       
    
    if parallel==True:
        sys.path.append(os.environ['SU2_HOME'])
        from parallel_computation import parallel_computation
        parallel_computation( tag+'.cfg', processors )
        pass
    else:
        subprocess.call(['SU2_CFD',tag+'.cfg'])
        
    f = open(tag + '_history.dat')
        
    SU2_results = Data()    
    
    lines = f.readlines()
    final_state = lines[-1].split(',')
    
    # Lift and Drag
    
    CL  = float(final_state[9])
    CD  = float(final_state[8])
    
    SU2_results.coefficient_of_lift  = CL
    SU2_results.coefficient_of_drag  = CD
    
    print('CL:',CL)
    print('CD:',CD)
    
    # Moments
    # Moments are currently not recorded since no
    # reasonable reference length has been chosen
    
    #CMx = float(final_state[4])
    #CMy = float(final_state[5])
    #CMz = float(final_state[6])   
    
    #SU2_results.moment_coefficient_x = CMx
    #SU2_results.moment_coefficient_y = CMy
    #SU2_results.moment_coefficient_z = CMz
    
    #print 'CMx:',CMx
    #print 'CMy:',CMy
    #print 'CMz:',CMz    
            
    return CL,CD

if __name__ == '__main__':
    call_SU2_CFD('cruise',parallel=True)
