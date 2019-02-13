## @ingroup Components-Wings-Airfoils
# Airfoil.py
# 
# Created:  
# Modified: Sep 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Components import Lofted_Body
import numpy as np

# ------------------------------------------------------------
#   Airfoil
# ------------------------------------------------------------

## @ingroup Components-Wings-Airfoils
class Airfoil(Lofted_Body.Section):
    def __defaults__(self):
        """This sets the default values of a airfoil defined in SUAVE.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """         
        
        self.tag                = 'Airfoil'
        self.thickness_to_chord = 0.0
        self.coordinate_file    = None    # absolute path
        self.points             = []
        
    
    def import_airfoil_dat(self):
        """Imports airfoil data

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """         
        
        filename = self.coordinate_file
        
        filein = open(filename,'r')
        data = {}   
        data['header'] = filein.readline().strip() + filein.readline().strip()
    
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
            
            point = list(map(float,line.split()))
            data[section].append(point)
            
        for k,v in data.items():
            data[k] = np.array(v)            
        
        self.points = data['upper'] 