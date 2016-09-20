# Airfoil.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald
#           Sep 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Components import Lofted_Body
import numpy as np

# ------------------------------------------------------------
#   Airfoil
# ------------------------------------------------------------

class Airfoil(Lofted_Body.Section):
    def __defaults__(self):
        self.tag                = 'Airfoil'
        #self.type               = 0
        #self.inverted           = False
        #self.camber             = 0.0
        #self.camber_loc         = 0.0
        self.thickness_to_chord = 0.0
        #self.thickness_loc      = 0.0
        #self.radius_le          = 0.0
        #self.radius_te          = 0.0
        #self.six_series         = 0
        #self.ideal_cl           = 0.0
        #self.A                  = 0.0
        #self.slat_flag          = False
        #self.slat_shear_flag    = False
        #self.slat_chord         = 0.0
        #self.slat_angle         = 0.0
        #self.flap_flag          = False
        #self.flap_shear_flag    = False
        #self.flap_chord         = 0.0
        #self.flap_angle         = 0.0
        self.coordinate_file    = None    # absolute path
        self.points             = [[],[]]
        
    
    def import_airfoil_dat(self):
        
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
            
            point = map(float,line.split())
            data[section].append(point)
            
        for k,v in data.items():
            data[k] = np.array(v)            
        
        self.points = data['upper'] 