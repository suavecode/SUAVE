
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Components import Lofted_Body


# ------------------------------------------------------------
#   Airfoil
# ------------------------------------------------------------

class Airfoil(Lofted_Body.Section):
    def __defaults__(self):
        self.tag = 'Airfoil'
        self.type            = 0
        self.inverted        = False
        self.camber          = 0.0
        self.camber_loc      = 0.0
        self.thickness       = 0.0
        self.thickness_loc   = 0.0
        self.radius_le       = 0.0
        self.radius_te       = 0.0
        self.six_series      = 0
        self.ideal_cl        = 0.0
        self.A               = 0.0
        self.slat_flag       = False
        self.slat_shear_flag = False
        self.slat_chord      = 0.0
        self.slat_angle      = 0.0
        self.flap_flag       = False
        self.flap_shear_flag = False
        self.flap_chord      = 0.0
        self.flap_angle      = 0.0