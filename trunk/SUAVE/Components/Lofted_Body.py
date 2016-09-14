# Lofted_Body.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Component          import Component
from Physical_Component import Physical_Component
from SUAVE.Core         import DataOrdered


# ------------------------------------------------------------
#  Lofted Body
# ------------------------------------------------------------

class Lofted_Body(Physical_Component):
    def __defaults__(self):
        self.tag = 'Lofted_Body'
        #self.Segments = SegmentContainer() # think edges
        self.Segments = DataOrdered()
        self.Sections = SectionContainer() # think nodes
    
   
# ------------------------------------------------------------
#  Segment
# ------------------------------------------------------------

class Segment(Component):
    def __defaults__(self):
        self.tag = 'Segment'
        
        self.prev = None
        self.next = None # for connectivity

        
# ------------------------------------------------------------
#  Section
# ------------------------------------------------------------

class Section(Component):
    def __defaults__(self):
        self.tag = 'Section'
        
        self.Curves = CurveContainer()
        
        self.prev = None
        self.next = None
        
# ------------------------------------------------------------
#  Curve
# ------------------------------------------------------------

class Curve(Component):
    def __defaults__(self):
        self.tag = 'Curve'
        self.points = []
        
# ------------------------------------------------------------
#  Containers
# ------------------------------------------------------------

#class SegmentContainer(Component.Container):
    #pass   

class SectionContainer(Component.Container):
    pass

class CurveContainer(Component.Container):
    pass


# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Section.Curve      = Curve
Section.Container  = SectionContainer
#Segment.Container  = SegmentContainer
Curve.Container    = CurveContainer
Lofted_Body.Section = Section
Lofted_Body.Segment = Segment
