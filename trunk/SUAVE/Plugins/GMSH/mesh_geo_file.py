import subprocess
import os

def mesh_geo_file(tag):
    
    #subprocess.call(['gmsh',tag+'.geo','-3','-optimize','-o',tag+'.su2','-format','su2','-saveall'])
    if os.path.isfile(tag+'.su2') == True:
        os.remove(tag+'.su2') # This prevents an leftover mesh from being used when SU2 is called
                          # This is important because otherwise the code will continue even if gmsh fails
    #subprocess.call(['gmsh',tag+'.geo','-3','-o',tag+'.su2','-format','su2','-saveall'])
    subprocess.call(['gmsh',tag+'.geo','-3','-o',tag+'.su2','-format','su2','-saveall'])
    
    pass