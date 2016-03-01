# Tree_Element.py
#
# Created: T. Lukaczyk Feb 2015
# Updated: Carlos Ilario, Feb 2016

""" SUAVE Methods for IO """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Ordered_Bunch


# ----------------------------------------------------------------------
#  Tree Element
# ----------------------------------------------------------------------

class Tree_Element(Ordered_Bunch):
    def __init__(self,name):
        self.name = name
        
    def append(self,element):
        if not 'children' in self:
            self.children = []
        self.children.append(e)