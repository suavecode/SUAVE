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

from Payload import Payload

# ----------------------------------------------------------------------
#  Cargo Data Class
# ----------------------------------------------------------------------

class Cargo(Payload):
    def __defaults__(self):
        self.tag = 'Cargo'
