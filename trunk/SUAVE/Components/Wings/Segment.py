# Wing.py
# 
# Created:  Sep 2016, E. Botero
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Components import Component, Lofted_Body, Mass_Properties

# ------------------------------------------------------------
#  Wing Segments
# ------------------------------------------------------------

class Segment(Lofted_Body.Segment):
    def __defaults__(self):
        self.tag = 'segment'
        self.percent_span_location = 0.0
        self.twist                 = 0.0
        self.root_chord_percent    = 0.0
        self.dihedral_outboard     = 0.0
        self.sweeps                = Data()
        self.sweeps.quarter_chord  = 0.0

class SegmentContainer(Lofted_Body.Segment.Container):
    pass