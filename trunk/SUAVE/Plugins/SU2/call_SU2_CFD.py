import subprocess


def call_SU2_CFD(tag):
    
    subprocess.call(['SU2_CFD',tag+'.cfg'])
    
    f = open('forces_breakdown.dat')
        
    SU2_results = Data()    
    
    # only the total forces have the ":"
    for line in f:
        if line.startswith('Total CL:'):
            print 'CL:',line.split()[2]
            SU2_results.coefficient_of_lift = line.split()[2]
        elif line.startswith('Total CD:'):
            print 'CD:',line.split()[2]
            SU2_results.coefficient_of_drag = line.split()[2]
        elif line.startswith('Total CMx:'):
            print 'CMx:',line.split()[2]
            SU2_results.moment_coefficient_x = line.split()[2]
        elif line.startswith('Total CMy:'):
            print 'CMy:',line.split()[2]
            SU2_results.moment_coefficient_y = line.split()[2]
        elif line.startswith('Total CMz:'):
            print 'CMz:',line.split()[2]
            SU2_results.moment_coefficient_z = line.split()[2]
           
    CL = SU2_results.coefficient_of_lift
    CD = SU2_results.coefficient_of_drag
            
    return CL,CD