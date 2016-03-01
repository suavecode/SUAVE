# __init__.py
# 
# Created:  Aug 2014, E. Botero
# Modified: Feb 2016, T. MacDonald

# ------------------------------------------------------------
#  Imports
# ------------------------------------------------------------

from SUAVE.Components import Physical_Component

# ------------------------------------------------------------
#  The Home Energy Container Class
# ------------------------------------------------------------
class Energy(Physical_Component):
    def __defaults__(self):
        pass


# ------------------------------------------------------------
#  Energy Component Classes
# ------------------------------------------------------------

class Component(Physical_Component):
    def __defaults__(self):
        self.tag = 'Energy Component'
    
    def provide_power():
        pass
    
class ComponentContainer(Physical_Component.Container):
    pass

# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------
Energy.Component = Component
Energy.Component.Container = ComponentContainer


