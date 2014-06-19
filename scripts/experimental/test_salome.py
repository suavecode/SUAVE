# test_geometry.py

import SUAVE
import numpy as np
import copy, sys, pickle

# MAIN
def main():
    
    test()
    
    return
    

# TEST GEOMETRY
def test():
    
    vsp_geometry_filename = 'Vehicles/CAD/nasa_bwbconcept/vsp_nasa_bwbconcept.vsp'
    
    Vehicle = SUAVE.Vehicle()
    
    SUAVE.Methods.IO.ImportFromVSP(Vehicle, vsp_geometry_filename)
    
    print Vehicle.Wings['Wing_Body'].Sections[0]
    print Vehicle.Wings['Wing_Body'].Segments[0]

    dump_wing(Vehicle)
    
    return


def dump_wing(Vehicle):
    
    import salome, geompy, smesh
    
    Sections = Vehicle.Wings['Wing_Body'].Sections
    Segments = Vehicle.Wings['Wing_Body'].Segments
    
    origin = Vehicle.Wings['Wing_Body'].origin
    for i,seg in enumerate(Segments.values()):
        sec = Sections[i]
        sec.chord = seg.RC
        sec.origin = copy.deepcopy(origin)
        
        origin[0] += seg.span * np.tan( seg.sweep    * np.pi/180. )
        origin[1] += seg.span * np.tan( seg.dihedral * np.pi/180. )
        origin[2] += seg.span
        
    Sections[-1].chord = Segments[-1].TC
    Sections[-1].origin = origin
    
    wires = []
    origins = []
    
    seg = None
    for sec in Sections.values():
        
        # build vertices
        origin, points = build_foil(sec)
        verts = [ geompy.MakeVertex(v[0],v[1],v[2]) for v in points ]
        
        # sketch sections
        wire = draw_airfoil_section(sec.tag,verts)
        
        # store
        origins.append(origin)
        wires.append(wire)
    
    # make wires
    
    data = {
        'wires'   : wires   ,
        'origins' : origins ,
    }
    
    # dump
    fileout = open('sections.pkl')
    pickle.dump(fileout,data)
    
    
    
    
        
        
def build_foil(sec):
    
    chord = sec.chord
    origin = np.array(sec.origin)
    
    upper = sec.Curves['upper'].points
    lower = sec.Curves['lower'].points
    
    upper = [ v+[0.0] for v in upper ]
    lower = [ v+[0.0] for v in lower ]
    upper.insert(0,lower[0])
    lower = lower.reverse()
    
    upper = np.array(upper)*chord + origin
    lower = np.array(lower)*chord + origin
    
    points = lower + upper    
    
    return origin, points

    
def draw_airfoil_section(name,verts):
    wire = geompy.MakePolyline(verts,theIsClosed=0)
    geompy.addToStudy(wire,name)
    return wire



# call main
if __name__ == '__main__':
    main()
