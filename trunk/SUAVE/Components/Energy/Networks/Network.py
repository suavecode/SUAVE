
# ------------------------------------------------------------
#  Imports 
# ------------------------------------------------------------

from SUAVE.Components.Energy import Energy


# ------------------------------------------------------------
#  Energy Network
# ------------------------------------------------------------

class Network(Energy.Component):
    def __defaults__(self):
        self.tag = 'Energy Network Component'

class Container(Energy.Component.Container):
    pass
    
    
# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------
Network.Container = Container
