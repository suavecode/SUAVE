
# ------------------------------------------------------------
#  Imports 
# ------------------------------------------------------------

from SUAVE.Components.Energy import Energy


# ------------------------------------------------------------
#  Energy Distributor
# ------------------------------------------------------------

class Distributor(Energy.Component):
    def __defaults__(self):
        self.tag = 'Energy Distribution Component'

class Container(Energy.Component.Container):
    pass
    

# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------
Distributor.Container = Container