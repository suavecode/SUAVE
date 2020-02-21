## @ingroup Components
# Lofted_Body.py
# 
# Created:  
# Modified: Dec 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .Component          import Component
from .Physical_Component import Physical_Component
from SUAVE.Core         import DataOrdered


# ------------------------------------------------------------
#  Lofted Body
# ------------------------------------------------------------

## @ingroup Components
class Lofted_Body(Physical_Component):
    def __defaults__(self):
        """This sets the default values.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """         
        self.tag = 'Lofted_Body'
        self.Segments = DataOrdered() # think edges
    
   
# ------------------------------------------------------------
#  Segment
# ------------------------------------------------------------

## @ingroup Components
class Segment(Component):
    """ A class that stubs out what a segment is
    
    Assumptions:
    None
    
    Source:
    None
    """      
    def __defaults__(self):
        """This sets the default values.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """         
        self.tag = 'Segment'
        
        self.prev = None
        self.next = None # for connectivity

# ------------------------------------------------------------
#  Section
# ------------------------------------------------------------

## @ingroup Components
class Section(Component):
    """ A class that stubs out what a section is
    
    Assumptions:
    None
    
    Source:
    None
    """     
    def __defaults__(self):
        """This sets the default values.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """         
        self.tag = 'Section'
        
        self.Curves = Curve_Container()
        
        self.prev = None
        self.next = None
        
# ------------------------------------------------------------
#  Curve
# ------------------------------------------------------------

## @ingroup Components
class Curve(Component):
    """ A class that stubs out what a curve is
    
    Assumptions:
    None
    
    Source:
    None
    """       
    def __defaults__(self):
        """This sets the default values.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """         
        self.tag = 'Curve'
        self.points = []
        
# ------------------------------------------------------------
#  Containers
# ------------------------------------------------------------

## @ingroup Components
class Section_Container(Component.Container):
    """ This does nothing
    
    Assumptions:
    None
    
    Source:
    None
    """    
    def get_children(self):
        """ Returns the components that can go inside
        
        Assumptions:
        None
    
        Source:
        N/A
    
        Inputs:
        None
    
        Outputs:
        None
    
        Properties Used:
        N/A
        """       
        
        return []

## @ingroup Components
class Curve_Container(Component.Container):
    """ This does nothing
    
    Assumptions:
    None
    
    Source:
    None
    """      
    pass


# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Section.Curve       = Curve
Section.Container   = Section_Container
Curve.Container     = Curve_Container
Lofted_Body.Section = Section
Lofted_Body.Segment = Segment

