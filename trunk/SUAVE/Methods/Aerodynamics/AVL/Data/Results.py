# Results.py
# Tim Momose, January 2015

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