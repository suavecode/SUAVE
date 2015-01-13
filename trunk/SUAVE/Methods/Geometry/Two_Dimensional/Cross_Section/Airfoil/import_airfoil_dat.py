
from math import sqrt, sin, cos, atan

def import_airfoil_dat(filename):
    
    filein = open(filename,'r')
    data = {}   
    data['header'] = filein.readline().strip()

    filein.readline()
    
    sections = ['upper','lower']
    data['upper'] = []
    data['lower'] = []
    section = None
    
    while True:
        line = filein.readline()
        if not line: 
            break
        
        line = line.strip()
        
        if line and section:
            pass
        elif not line and not section:
            continue
        elif line and not section:
            section = sections.pop(0)
        elif not line and section:
            section = None
            continue
        
        point = map(float,line.split())
        data[section].append(point)
        
    return data