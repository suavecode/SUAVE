# test_geometry.py

import SUAVE
import pySolidWorks
import numpy as np
import copy, time
from collections import OrderedDict as odict

from SUAVE.Geometry.Three_Dimensional import angle_to_dcm

# MAIN
def main():
    
    import_BWB_H2()
    #import_B737_H2()
    
    return
    
def import_BWB_H2():
    
    vsp_geometry_filename = '../trunk/Vehicles/CAD/adl_BWB-H2/geom_adl_BWB-H2.vsp'
    cad_geometry_filename = '../trunk/Vehicles/CAD/adl_BWB-H2/geom_adl_BWB-H2.sldprt'
    
    Vehicle = SUAVE.Vehicle()
    
    SUAVE.Methods.Input_Output.import_from_VSP(Vehicle, vsp_geometry_filename)
    
    SW = pySolidWorks.SolidWorks()
    SW.visible()
    SW_Model = SW.new_part()
    #SW_Model = SW.open_model(cad_geometry_filename)
    
    cad_wing(Vehicle.Wings['Wing_Body'],SW_Model)
    cad_wing(Vehicle.Wings['Vertical_Tails'],SW_Model)
 
def import_B737_H2():
    
    hrm_geometry_filename = 'Vehicles/CAD/adl_B737-H2/xsec_adl_B737-H2.hrm'
    cad_geometry_filename = 'Vehicles/CAD/adl_B737-H2/geom_adl_B737-H2.sldprt'
    
    components = read_hrm(hrm_geometry_filename)
    
    # trim wing sections
    isecs_1 = range(0,5+1,1)
    isecs_2 = range(5,11,1) + range(11,47,6) + range(47,56,2) + range(56,98,6) + range(98,105+1,1)
    wing_1 = odict()
    wing_2 = odict()
    for i in isecs_1:
        key = components['Wing'].keys()[i]
        wing_1[key] = components['Wing'][key]
    for i in isecs_2:
        key = components['Wing'].keys()[i]
        wing_2[key] = components['Wing'][key]        
    components['Wing_1'] = wing_1
    components['Wing_2'] = wing_2
        
        
    
    SW = pySolidWorks.SolidWorks()
    SW.visible()
    SW_Model = SW.new_part()
    #SW_Model = SW.open_model(cad_geometry_filename)
    
    
    cad_component(components,'Wing_1',SW_Model,True)
    cad_component(components,'Wing_2',SW_Model,True)
    cad_component(components,'Tails',SW_Model,True)
    cad_component(components,'Fuselage',SW_Model)
    
    
    
def cad_component(Components,component_name,Model,split=False):
    
    Component = Components[component_name]
    sketches = []
    
    for section,points in Component.items():
        
        section_name = component_name + ' - ' + section
        
        if split:
            i_split = 50            
            upper = points[:i_split+1]
            lower = points[i_split:]
            
            # insert sketch
            sketch = Model.enter_sketch(section_name,dim=3)
            Model.insert_spline(upper)
            Model.insert_spline(lower)
            Model.exit_sketch()            

            
        else:
            
            # insert sketch
            sketch = Model.enter_sketch(section_name,dim=3)
            Model.insert_spline(points)
            Model.exit_sketch()
        
        Model.clear_selection()
        
        sketches.append(sketch)
    
    # make solid
    Model.rebuild()
    Model.insert_boundary_surface(sketches,[])    
    
    time.sleep(5.0)
    
    return 
    
def import_vsp():
    
    #vsp_geometry_filename = 'Vehicles/OpenVSP/NASA_N2A_Hybrid_Wing_Body.vsp'
    vsp_geometry_filename = 'Vehicles/CAD/adl_B737-H2/geom_adl_B737-H2.vsp'
    cad_geometry_filename = 'Vehicles/CAD/adl_B737-H2/geom_adl_B737-H2.sldprt'
    
    Vehicle = SUAVE.Vehicle()
    
    SUAVE.Methods.IO.ImportFromVSP(Vehicle, vsp_geometry_filename)
    
    #print Vehicle.Wings['Wing_Body'].Sections[0]
    #print Vehicle.Wings['Wing_Body'].Segments[0]
    
    SW = pySolidWorks.SolidWorks()
    SW.visible()
    SW_Model = SW.new_part()
    #SW_Model = SW.open_model(cad_geometry_filename)
    
    cad_wing(Vehicle.Wings['Wing'],SW_Model)
    cad_wing(Vehicle.Wings['Tails'],SW_Model)
    cad_fuselage(Vehicle.Fuselages['Fuselage'],SW_Model)
    #cad_wing(Vehicle.Wings['Wing_Body'],SW_Model)
    #cad_wing(Vehicle.Wings['Vertical_Tails'],SW_Model)
    
    return


def cad_wing(Wing,SW_Model):
    
    Sections = Wing.Sections
    Segments = Wing.Segments
    
    origin = copy.deepcopy(Wing.origin)
    for i,seg in enumerate(Segments.values()):
        sec = Sections[i]
        sec.chord = seg.RC
        sec.origin = origin
        
        # scaling
        sec.scaling = [sec.chord,1.0,sec.chord]
        
        # origin sweeping
        #d_origin = [ seg.span * np.tan( seg.sweep    * np.pi/180. ) ,
                     #seg.span                                       ,
                     #seg.span * np.tan( seg.dihedral * np.pi/180. )  ]
        #if seg.rot_sec_dihed:
        d_origin = [ seg.span * np.tan( seg.sweep    * np.pi/180. ) ,
                     seg.span * np.cos( seg.dihedral * np.pi/180. ) ,
                     seg.span * np.sin( seg.dihedral * np.pi/180. )  ]
        origin = [ o+d for o,d in zip(origin,d_origin) ]
        
        # euler rotation vector
        rotation = [0.0,0.0,0.0] 
        rotation_order = 'XZY'
        if seg.rot_sec_dihed:
            rotation[0] = seg.dihedral
            if i > 0 and i < len(Segments):
                rotation[0] = ( rotation[0] + Segments[i-1].dihedral ) / 2
        rotation[2] = seg.twist
        sec.rotation = rotation
        sec.rotation_order = rotation_order
        
    # last section
    sec = Sections[-1]
    seg = Segments[-1]    
    
    sec.chord = seg.TC
    sec.origin = origin
    sec.scaling = [sec.chord,1.0,sec.chord]
    
    rotation = [0.0,0.0,0.0] 
    rotation_order = 'XZY'
    if seg.rot_sec_dihed:
        rotation[0] = seg.dihedral
    rotation[2] = seg.twist
    sec.rotation = rotation
    sec.rotation_order = rotation_order    
    
    sec = None
    seg = None
    
    sketches = []
    for sec in Sections.values():
        
        sketch = draw_airfoil_section(SW_Model,Wing,sec)
        sketches.append(sketch)
    
    # make solid
    SW_Model.insert_boundary_surface(sketches,[],Wing.tag)
        
        
    
def cad_fuselage(Fuselage,SW_Model):
    
    Segments = Fuselage.Segments
    Sections = Fuselage.Sections
    
    draw_map = { 
        'POINT'   : draw_point_section ,
        'LINE'    : draw_line_section ,
        'CIRCLE'  : draw_circle_section ,
        'ELLIPSE' : draw_ellipse_section ,
    }
    
    sketches = []
    for tag,sec in Sections.items():
        
        draw_func = draw_map[sec.type]
        
        sketch = draw_func(SW_Model,Fuselage,sec)
        sketches.append(sketch)
        
    # make solid
    SW_Model.insert_boundary_surface(sketches,[],Fuselage.tag)
    
    
    
def draw_airfoil_section(Model,Wing,Section):
    
    # unpack
    name     = Wing.tag + ' - ' + Section.tag    
    chord    = Section.chord
    origin   = np.array(Section.origin)
    scaling  = np.array(Section.scaling)
    rotation = np.array(Section.rotation)
    rotation_order = Section.rotation_order
    
    upper = Section.Curves['upper'].points
    lower = Section.Curves['lower'].points
    upper = np.array(upper)
    lower = np.array(lower)        

    # scaling
    upper = upper*scaling
    lower = lower*scaling

    # rotation
    transform = angle_to_dcm(rotation,rotation_order,'degrees')
    upper = np.dot(upper,transform)
    lower = np.dot(lower,transform)
    
    # translation
    upper = upper + origin
    lower = lower + origin

    # insert sketch
    sketch = Model.enter_sketch(name,dim=3)
    Model.insert_spline(upper)
    Model.insert_spline(lower)
    Model.exit_sketch()
    
    Model.clear_selection()
    #Model.rebuild()
    
    return sketch

def draw_circle_section(Model,Fuselage,Section):
    
    name   = Fuselage.tag + ' - ' + Section.tag  
    center = Section.origin
    radius = section.radius
    
    point1 = copy.deepcopy(center)
    point1[2] = center[2] + radius
    
    point2 = copy.deepcopy(center)
    point2[2] = center[2] - radius
    
    # fuselage origin
    center = [ p+o for p,o in zip(center,Fuselage.origin) ]
    point1 = [ p+o for p,o in zip(point1,Fuselage.origin) ]
    point2 = [ p+o for p,o in zip(point2,Fuselage.origin) ]
    
    sketch = Model.enter_sketch(name,dim=3)
    Model.insert_circle_arc(center,point1,point2,+1.)
    Model.insert_sketch_line(point1,point2)
    Model.exit_sketch()
    
    Model.clear_selection()
    #Model.rebuild()
    
    return sketch

def draw_line_section(Model,Fuselage,Section):
    
    name   = Fuselage.tag + ' - ' + Section.tag  
    point1 = Section.point1
    point2 = Section.point2
    
    # section and fuselage origin
    point1 = [ p+s+o for p,s,o in zip(point1,Section.origin,Fuselage.origin) ]
    point2 = [ p+s+o for p,s,o in zip(point2,Section.origin,Fuselage.origin) ]
    
    sketch = Model.enter_sketch(name,dim=3)
    Model.insert_sketch_line(point1,point2)
    Model.exit_sketch()
    
    Model.clear_selection()    
    #Model.rebuild()
    
    return sketch

def draw_ellipse_section(Model,Fuselage,Section):
    
    name   = Fuselage.tag + ' - ' + Section.tag  
    center = Section.origin
    height = Section.height
    width  = Section.width

    point1 = copy.deepcopy(center)
    point1[2] = center[2] + height/2.
    
    point2 = copy.deepcopy(center)
    point2[2] = center[2] - height/2.
    
    p_major = copy.deepcopy(point1)
    
    p_minor = copy.deepcopy(center)
    p_minor[1] = center[1] + width/2.
    
    # fuselage origin
    origin = Fuselage.origin
    center  = [ p+o for p,o in zip(center,origin) ]
    p_major = [ p+o for p,o in zip(p_major,origin) ]
    p_minor = [ p+o for p,o in zip(p_minor,origin) ]
    point1  = [ p+o for p,o in zip(point1,origin) ]
    point2  = [ p+o for p,o in zip(point2,origin) ]
    
    # insert plane
    plane_name = name + ' - Plane'
    
    plane = Model.insert_reference_plane('Right Plane',center[0],plane_name)
    Model.clear_selection()
    Model.set_selection_byname(plane_name,'PLANE')
    sketch = Model.enter_sketch(name,dim=2)
    
    # plane orientation
    order = [2,1,0]
    sign  = [-1,-1,1]
    center = [ center[i]*s for i,s in zip(order,sign) ]
    p_major = [ p_major[i]*s for i,s in zip(order,sign) ]
    p_minor = [ p_minor[i]*s for i,s in zip(order,sign) ]
    point1 = [ point1[i]*s for i,s in zip(order,sign) ]
    point2 = [ point2[i]*s for i,s in zip(order,sign) ]

    Model.insert_ellipse_arc(center,p_major,p_minor,point1,point2,-1)
    Model.insert_sketch_line(point1,point2)
    #sketch = Model.get_active_sketch()
    Model.exit_sketch()
    Model.hide(plane)
    
    Model.clear_selection()
    #Model.rebuild()
    
    return sketch

def draw_point_section(Model,Fuselage,Section):
    
    name     = Fuselage.tag + ' - ' + Section.tag  
    position = Section.origin
    
    # fuselage origin
    position  = [ p+o for p,o in zip(position,Fuselage.origin) ]
    
    sketch = Model.enter_sketch(name,dim=3)
    Model.insert_sketch_point(position)
    Model.exit_sketch()
    
    Model.clear_selection()
    #Model.rebuild()
    
    return sketch


def read_hrm(filename):
    
    filein = open(filename,'r')
    
    components = {}
    
    component = None
    
    while True:
        
        line = filein.readline()
        if not line: 
            break        
        
        if not component :
            if line.startswith(' ') or not line.strip():
                continue
            
            else:
                component = line.strip()
                
                components[component] = odict()
                
                inputs = {}
                for k in ['GROUP','TYPE','SECTIONS','POINTS']:
                    line = filein.readline().strip().split('=')[-1].strip()
                    inputs[k] = int(line)
                
                sections = inputs['SECTIONS']
                points = inputs['POINTS']
                
                isec = 0
                ipnt = 0
                
                continue
            
            #: if line
             
        else:
            line = line.strip()
            if not line:
                component = None
                continue
            
            else:
                
                assert isec < sections
                    
                line = line.split()
                coord = map(float,line)
                                
                section = 'Section_%i' % isec
                
                if ipnt == 0:
                    components[component][section] = []
                
                components[component][section].append(coord)
                
                ipnt += 1
                if ipnt >= points:
                    ipnt = 0
                    isec+=1
                    
            #: if line
                    
        #: if section
                    
    return components
                

            
            
    


# call main
if __name__ == '__main__':
    main()
