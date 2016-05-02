# Person.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald

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
        self.mass_properties.mass = 90.718474     # kg, = 200 lb
