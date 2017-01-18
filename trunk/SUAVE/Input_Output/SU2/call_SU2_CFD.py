# call_SU2_CFD.py
# 
# Created:  Oct 2016, T. MacDonald
# Modified: Jan 2017, T. MacDonald

import subprocess
from SUAVE.Core import Data
import sys, os

def call_SU2_CFD(tag,parallel=False,processors=1):
    
    if parallel==True:
        sys.path.append(os.environ['SU2_HOME'])
        from parallel_computation import parallel_computation
        parallel_computation( tag+'.cfg', processors )
        pass
    else:
        subprocess.call(['SU2_CFD',tag+'.cfg'])
        
    f = open(tag + '_forces_breakdown.dat')
        
    SU2_results = Data()    
    
    # only the total forces have the ":"
    for line in f:
        if line.startswith('Total CL:'):
            print 'CL:',line.split()[2]
            SU2_results.coefficient_of_lift = float(line.split()[2])
        elif line.startswith('Total CD:'):
            print 'CD:',line.split()[2]
            SU2_results.coefficient_of_drag = float(line.split()[2])
        elif line.startswith('Total CMx:'):
            print 'CMx:',line.split()[2]
            SU2_results.moment_coefficient_x = float(line.split()[2])
        elif line.startswith('Total CMy:'):
            print 'CMy:',line.split()[2]
            SU2_results.moment_coefficient_y = float(line.split()[2])
        elif line.startswith('Total CMz:'):
            print 'CMz:',line.split()[2]
            SU2_results.moment_coefficient_z = float(line.split()[2])
           
    CL = SU2_results.coefficient_of_lift
    CD = SU2_results.coefficient_of_drag
            
    return CL,CD

if __name__ == '__main__':
    call_SU2_CFD('cruise',parallel=True)