
# ------------------------------------------------------------
#  Imports
# ------------------------------------------------------------

from SUAVE.Components import Physical_Component
from Connections import Connections

# ------------------------------------------------------------
#  The Home Energy Container Class
# ------------------------------------------------------------
class Energy(Physical_Component):
    def __defaults__(self):
        
        from Storages     import Storage
        from Distributors import Distributor
        from Converters   import Converter
        from Networks     import Network

        self.tag = 'Energy'
        self.Storages      = Storage.Container()
        self.Distributors  = Distributor.Container()
        self.Converters    = Converter.Container()
        self.Networks      = Network.Container()


# ------------------------------------------------------------
#  Energy Component Classes
# ------------------------------------------------------------

class Component(Physical_Component):
    def __defaults__(self):
        self.tag = 'Energy Component'
        self.Connections = Connections()
    
    def provide_power():
        pass
    
class ComponentContainer(Physical_Component.Container):
    pass

# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------
Energy.Component = Component
Energy.Component.Container = ComponentContainer


