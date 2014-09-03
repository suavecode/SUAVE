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
#  Person Data Class
# ----------------------------------------------------------------------
        
class Person(Payload):
    def __defaults__(self):
        self.tag = 'Person'
        self.Mass_Properties.mass = 90.718474     # kg, = 200 lb
