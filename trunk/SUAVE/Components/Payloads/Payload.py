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

from SUAVE.Components import Physical_Component


# ----------------------------------------------------------------------
#  Payload Base Class
# ----------------------------------------------------------------------
        
class Payload(Physical_Component):
    def __defaults__(self):
        self.tag = 'Payload'
        
class Container(Physical_Component.Container):
    pass

Payload.Container = Container

