#Created by M. Vegh 7/25/13     #modified by M. Vegh 7/25/13


""" SUAVE.Attrubtes.Components.Energy.Conversion
"""

# ------------------------------------------------------------
#  Imports 
# ------------------------------------------------------------

from SUAVE.Components.Energy import Energy


# ------------------------------------------------------------
#  Energy Conversion
# ------------------------------------------------------------

class Converter(Energy.Component):
    """ SUAVE.Attributes.Components.Energy.Conversion.Component
    """
    def __defaults__(self):
        self.tag = 'Energy Conversion Component'

class Container(Energy.Component.Container):
    """ SUAVE.Attributes.Components.Energy.Conversion.Container
    """
    pass


# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------
Converter.Container = Container
