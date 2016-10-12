import subprocess

def mesh_geo_file(tag):
    
    subprocess.call(['gmsh',tag+'.geo','-3','-optimize','-o',tag+'.su2','-format','su2','-saveall'])
    #subprocess.call(['gmsh',tag+'.geo','-3','-o',tag+'.su2','-format','su2','-saveall'])
    