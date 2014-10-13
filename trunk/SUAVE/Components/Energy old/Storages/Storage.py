
""" SUAVE.Attrubtes.Components.Energy.Conversion
"""

# ------------------------------------------------------------
#  Imports 
# ------------------------------------------------------------

from SUAVE.Components.Energy import Energy

# ------------------------------------------------------------
#  Energy Storage
# ------------------------------------------------------------

class Storage(Energy.Component):
    """ SUAVE.Components.Energy.Storages.Storage()
    """
    def __defaults__(self):
        self.tag = 'Energy Storage Component'

class Container(Energy.Component.Container):    
    """ SUAVE.Attributes.Components.Energy.Storage.Container()
    """    
    pass
    

# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------
Storage.Container = Container
