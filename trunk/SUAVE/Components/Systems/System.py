# Vehicle.py
#
# Created By:       T. Lukaczyk
# Updated:          M. Colonno  4/20/13

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
        self.tag = 'System'

class Container(Component.Container):
    pass

System.Container = Container