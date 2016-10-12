# Created:  Feb 2015, T. Lukaczyk 
# Modified: Jul 2016, E. Botero 


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