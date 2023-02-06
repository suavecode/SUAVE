## @ingroup Components-Airfoils
# Airfoil.py
# 
# Created:  
# Modified: Sep 2016, E. Botero
#           Mar 2020, M. Clarke
#           Oct 2021, M. Clarke


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from MARC.Core import Data
from MARC.Components import Lofted_Body

# ------------------------------------------------------------
#   Airfoil
# ------------------------------------------------------------

## @ingroup Components-Airfoils
class Airfoil(Lofted_Body.Section):
    def __defaults__(self):
        """This sets the default values of a airfoil defined in MARC.

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
        
        self.tag                        = 'Airfoil' 
        self.coordinate_file            = None    # absolute path  
        self.NACA_4_series_flag         = False   # Flag for NACA 4 series airfoil
        self.geometry                   = None
        self.polar_files                = None
        self.polars                     = None
        self.number_of_points           = 200
       