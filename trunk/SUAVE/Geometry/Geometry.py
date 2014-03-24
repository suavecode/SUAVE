# Geoemtry.py
#

""" SUAVE Methods for Geoemtry Generation
"""

# TODO:
# object placement, wing location
# tail: placed at end of fuselage, or pull from volume
# engines: number of engines, position by 757

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy
from math import pi, sqrt
from SUAVE.Structure  import Data
#from SUAVE.Attributes import Constants

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def fuselage_crosssection(Fuselage):
    """ err = SUAVE.Geometry.fuselage_planform(Fuselage)
        calculate relevent fuselage crosssection dimensions
        
        Assumptions:
            wall_thickness = 0.04 * seat_width
        
        Inputs:
            Fuselage.seat_width
            Fuselage.seat_layout_lower
            Fuselage.aisle_width
            Fuselage.fuse_hw
            
        Outputs:
            Fuselage.wall_thickness
            Fuselage.height
            Fusealge.width
        
    """
    
    # assumptions
    wall_thickness_ratio = 0.04 # ratio of inner diameter to wall thickness
    
    # unpack main floor
    seat_width     = Fuselage.seat_width
    layout         = Fuselage.seat_layout_lower
    aisle_width    = Fuselage.aisle_width
    fuselage_hw    = Fuselage.fuse_hw
    
    # calculate
    total_seat_width = sum(layout)*seat_width + \
                      (len(layout)-1)*aisle_width
    
    # needs verification
    wall_thickness  = total_seat_width * wall_thickness_ratio
    fuselage_width  = total_seat_width + 2*wall_thickness
    fuselage_height = fuselage_hw * fuselage_width
    
    # update
    Fuselage.wall_thickness  = wall_thickness
    Fuselage.width  = fuselage_width
    Fuselage.height = fuselage_height
    
    return 0

def fuselage_planform(Fuselage):
    """ err = SUAVE.Geometry.fuselage_planform(Fuselage)
    
        Assumptions:
            fuselage cross section is an ellipse
            ellipse circumference approximated
            
        Inputs:
            Fuselage.num_coach_seats
            Fuselage.seat_pitch
            Fuselage.fineness_nose
            Fuselage.fineness_tail
            Fuselage.fwdspace
            Fuselage.aftspace
            Fuselage.width
            Fuselage.height            
            
        Outputs:
            Fuselage.length_nose
            Fuselage.length_tail
            Fuselage.length_cabin
            Fuselage.length_total
            Fuselage.area_wetted
            
    """
    
    # unpack
    number_seats    = Fuselage.num_coach_seats
    seat_pitch      = Fuselage.seat_pitch
    nose_fineness   = Fuselage.fineness_nose
    tail_fineness   = Fuselage.fineness_tail
    forward_extra   = Fuselage.fwdspace
    aft_extra       = Fuselage.aftspace
    fuselage_width  = Fuselage.width
    fuselage_height = Fuselage.height
    
    # process
    nose_length  = nose_fineness * fuselage_width
    tail_length  = tail_fineness * fuselage_width
    cabin_length = number_seats * seat_pitch + \
                   forward_extra + aft_extra
    fuselage_length = cabin_length + nose_length + tail_length
    
    wetted_area = 0.0
    
    # model constant fuselage cross section as an ellipse
    # approximate circumference http://en.wikipedia.org/wiki/Ellipse#Circumference
    a = fuselage_width/2.
    b = fuselage_height/2.
    R = (a-b)/(a+b)
    C = pi*(a+b)*(1.+ ( 3*R**2 )/( 10+sqrt(4.-3.*R**2) ))
    wetted_area += C * cabin_length
    
    # approximate nose and tail wetted area
    # http://adg.stanford.edu/aa241/drag/wettedarea.html
    Deff = (a+b)*(64.-3.*R**4)/(64.-16.*R**2)
    wetted_area += 0.75*pi*Deff * (nose_length + tail_length)
    
    # update
    Fuselage.length_nose  = nose_length
    Fuselage.length_tail  = tail_length
    Fuselage.length_cabin = cabin_length
    Fuselage.length_total = fuselage_length
    Fuselage.area_wetted  = wetted_area
    
    return 0

def wing_planform(Wing):
    """ err = SUAVE.Geometry.wing_planform(Wing)
    
        basic wing planform calculation
        
        Assumptions:
            trapezoidal wing
            no leading/trailing edge extensions
            
        Inputs:
            Wing.sref
            Wing.ar
            Wing.taper
            Wing.sweep
            
        Outputs:
            Wing.chord_root
            Wing.chord_tip
            Wing.chord_mac
            Wing.area_wetted
            Wing.span
        
    """
    
    # unpack
    sref  = Wing.sref
    ar    = Wing.ar
    taper = Wing.taper
    sweep = Wing.sweep
    
    # calculate
    span = sqrt(ar*sref)
    
    chord_root = 2*sref/span/(1+taper)
    chord_tip  = taper * chord_root
    
    swet = 2*span/2*(chord_root+chord_tip)

    mac = 2/3*( chord_root+chord_tip - chord_root*chord_tip/(chord_root+chord_tip) )
    
    # update
    Wing.chord_root  = chord_root
    Wing.chord_tip   = chord_tip
    Wing.chord_mac   = mac
    Wing.area_wetted = swet
    Wing.span        = span
    
    return 0

def main_wing_planform(Wing):
    """ err = SUAVE.Geometry.main_wing_planform(Wing)
        
        main wing planform
        
        Assumptions:
            cranked wing with leading and trailing edge extensions
        
        Inputs:
            Wing.sref
            Wing.ar
            Wing.taper
            Wing.sweep
            Wing.span
            Wing.lex
            Wing.tex
            Wing.span_chordext
    
        Outputs:
            Wing.chord_root
            Wing.chord_tip
            Wing.chord_mid
            Wing.chord_mac
            Wing.area_wetted
            Wing.span
    
    """
    
    # unpack
    span          = Wing.span
    lex           = Wing.lex
    tex           = Wing.tex
    span_chordext = Wing.span_chordext    
    
    # run basic wing planform
    # mac assumed on trapezoidal reference wing
    err = wing_planform(Wing)
    
    # unpack more
    chord_root    = Wing.chord_root
    chord_tip     = Wing.chord_tip
    
    # calculate
    chord_mid = chord_root + span_chordext*(chord_tip-chord_root)
    
    swet = 2*span/2*(span_chordext*(chord_root+lex+tex + chord_mid) +
                     (1-span_chordext)*(chord_mid+chord_tip))    
    
    # update
    Wing.chord_mid = chord_mid
    Wing.swet      = swet

    return 0

def vertical_tail_planform(Wing):
    """ results = SUAVE.Geometry.vertical_tail_planform(Wing)
        
        see SUAVE.Geometry.wing_planform()
    """
    wing_planform(Wing)
    return 0
    
def horizontal_tail_planform(Wing):
    """ results = SUAVE.Geometry.horizontal_tail_planform(Wing)
    
        see SUAVE.Geometry.wing_planform()
    """
    wing_planform(Wing)
    return 0
            
def airfoil_generator(AV,number=0000):
    pass

def nacelle_planform(Nacelle):
    pass






# -----------------------------------------------------------
#  High Fidelity Geometry
# -----------------------------------------------------------

class Body(object):
    ''' SUAVE.Geometry.Body()
        abstract class for geoemtry object
    '''
    
    Length = 0.0
    RefLength = 0.0
    TopArea = 0.0
    SideArea = 0.0
    FrontArea = 0.0
    WettedArea = 0.0
    Volume = 0.0
    Weight = 0.0
    CG_xyz = 0.0
    
    # connectivity
    parents  = []
    children = []
    
    def __init__(self):
        
        return
    
    def get_Handbook():
        ''' returns data needed for Handbook methods
        '''
        
        return {}
    
    def get_AVL():
        ''' returns data needed for AVL
        '''
        
        return {}
    
    def get_CAD():
        ''' returns data needed for CAD
        '''
        
        return {}
    
    def get_Mesh():
        ''' returns data needed for surface mesh
        '''
        
        return {}    


class Wing(Body):
    ''' SUAVE.Geometry.Wing()
    '''
    
    # includes attributes from Body()
    Span = 0.0
    AspectRatio = 0.0
    DihedralAngle = 0.0
    SweepAngle = 0.0
    TapeRatio = 0.0
    MeanAeroChord = 0.0
    MeanGeoChord = 0.0
    TipChord = 0.0
    RootChord = 0.0
    
    def __init__(self):
        ''' initialize a Wing
        '''
        
        Body.__init__(self)
        
        self.PlanformArea = TopArea
        
        return
    


# simple interfaces for common wing types
class Canard(Wing):
    pass
class CrankedWing(Wing):
    pass
class Fin(Wing):
    pass
class ObliqueWing(Wing):
    pass
class Pylon(Wing): # maybe tell it two points to connect
    pass
class VerticalTail(Wing):
    pass
class Fin(Wing):
    pass


class Nacelle(Body):
    ''' SUAVE.Geometry.Nacelle()
    '''
    
    InletArea = 0.0  # zero for closed nacelle
    OutletArea = 0.0
    
    
    def __init__(self):
        ''' initialize a Wing
        '''
        
        Body.__init__(self)
        
        self.PlanformArea = TopArea
        
        return
        

class Fuselage(Body):
    '''
    '''
    
    pass


# for higher fidelity geometry

class CrossSection(object):
    ''' SUAVE.Geometry.CrossSection()
        defines cross section needed for lofted surface
    '''
    
    
    def __init__(self,raw=[],spline=[],naca=''):
        ''' initializes a cross section
            points assumed relative to user defined origin
            accepts raw 2d points, spline control points, or naca airfoil number
        '''
        
        return
    
class Airfoil(CrossSection):
    ''' SUAVE.Geoemtry.CrossSection.Airfoil()
    '''
    
    Chord = 0.0
    ThicknessRatio = 0.0
    
    def __init__(self):
        pass
    
class LoftedSurface(object):
    ''' SUAVE.Geoemtry.LoftedSurface()
    '''
    
    def __init__(self,sections=[]):
        ''' initialize a lofted surface defined by
            a collection of parallel cross-sections
            sections = [ [section,(x,y,z)] , [,]... ]
        '''
        
        self.sections = sections
        
        return
    
    def insert_sections(self,sections=[]):
        ''' inserts one or more sections
        '''
        
        return
    
    def projected_area(self,direction):
        ''' calculate projected area in specifed direction vector
        '''
        # maybe assume some 
        area = 0
        
        return area