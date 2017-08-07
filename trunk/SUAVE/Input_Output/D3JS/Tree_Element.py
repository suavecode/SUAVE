## @ingroup Input_Output-D3JS
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
## @ingroup Input_Output-D3JS
class Tree_Element(DataOrdered):
    """This is a tree element used in D3JS trees.
    
    Assumptions:
    None

    Source:
    N/A
    """       
    def __init__(self,name):
        """This sets default values.
        
        Assumptions:
        None
    
        Source:
        N/A
    
        Inputs:
        name    <string>
    
        Outputs:
        None
    
        Properties Used:
        N/A
        """           
        self.name = name
        
    def append(self,element):
        """This adds an element to self.children
        
        Assumptions:
        None
    
        Source:
        N/A
    
        Inputs:
        element - an element to be added to self.children
    
        Outputs:
        None
    
        Properties Used:
        self.children (created if not already available)
        """           
        if not 'children' in self:
            self.children = []
        self.children.append(e)