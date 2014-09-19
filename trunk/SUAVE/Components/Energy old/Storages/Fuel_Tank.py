
""" SUAVE.Attrubtes.Components.Energy.Conversion
"""

# ------------------------------------------------------------
#  Imports 
# ------------------------------------------------------------

from SUAVE.Components.Energy import Energy
from Storage import Storage


# ------------------------------------------------------------
#  Fuel Tank
# ------------------------------------------------------------
    
class Fuel_Tank(Storage):
    def __defaults__(self):
        self.tag = 'Fuel_Tank'
        self.Propellant = Energy.Component()
        self.Tank       = Energy.Component()