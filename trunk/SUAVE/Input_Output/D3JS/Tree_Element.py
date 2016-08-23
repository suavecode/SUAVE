# Tree_Element.py
#
<<<<<<< HEAD
# Created: T. Lukaczyk Feb 2015
# Updated: Carlos Ilario, Feb 2016
=======
# Created:  Feb 2015, T. Lukaczyk 
# Modified: Jul 2016, E. Botero 
>>>>>>> original/develop

""" SUAVE Methods for IO """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import DataOrdered

#----------------------------------------------------------------------
# Tree Element
# ----------------------------------------------------------------------

class Tree_Element(DataOrdered):
    def __init__(self,name):
        self.name = name
        
    def append(self,element):
        if not 'children' in self:
            self.children = []
        self.children.append(e)