# System.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald

""" SUAVE Vehicle container class 
    with database + input / output functionality 
"""


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Components import Component


# ----------------------------------------------------------------------
#  Payload Base Class
# ----------------------------------------------------------------------
        
class System(Component):
    def __defaults__(self):
        self.tag             = 'System'
        self.mass_properties = mass_properties()
        self.position        = [0.0,0.0,0.0]
        self.control         = None
        self.accessories     = None

class Container(Component.Container):
    pass

System.Container = Container