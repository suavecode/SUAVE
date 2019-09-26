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
        self.Sections = SectionContainer() # think nodes
    
   
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
        
        self.Curves = CurveContainer()
        
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
class SectionContainer(Component.Container):
    """ This does nothing
    
    Assumptions:
    None
    
    Source:
    None
    """    
    pass

## @ingroup Components
class CurveContainer(Component.Container):
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
Section.Container   = SectionContainer
Curve.Container     = CurveContainer
Lofted_Body.Section = Section
Lofted_Body.Segment = Segment



