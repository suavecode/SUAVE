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

from SUAVE.Components import Lofted_Body

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
        
        self.tag                   = 'Airfoil'
        self.thickness_to_chord    = 0.0
        self.naca_4_series_airfoil = None    # string of 4 digits defining NACA 4 series airfoil"
        self.coordinate_file       = None    # absolute path
        self.points                = []
       