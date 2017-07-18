# Results.py
# 
# Created:  Jan 2015, T. Momose
# Modified: Jan 2016, E. Botero
#           Jul 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data

# ------------------------------------------------------------
#   Wing
# ------------------------------------------------------------

class Results(Data):
    def __defaults__(self):

        self.aerodynamics = Data()
        self.stability    = Data()
        
        self.stability.alpha_derivatives = Data()
        self.stability.beta_derivatives  = Data()
        