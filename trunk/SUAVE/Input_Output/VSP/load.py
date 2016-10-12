# load.py
#
# Created By:       M. Colonno  4/15/13
# Updated:          M. Colonno  4/24/13
#                   T. Lukaczyk 12/19/13
#                   C. Ilario   Feb/16

""" Import from VSP File """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# standard imports
import copy
from warnings import warn

# suave imports
import SUAVE.Components
from SUAVE.Input_Output.XML import load as import_from_xml
from SUAVE.Core  import Data, Container


# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------


def load(vehicle,input_file,mass_props_file=""):

    """ SUAVE.Methods.Input_Output.import_from_VSP(vehicle,input_file,mass_props_file)
        Import OpenVSP model and mass properties

        Inputs:     vehcile             Vehicle class instance (required) 
        Inputs:     input_file          OpenVSP model file (required) 
                    mass_props_file     OpenVSP mass properties file (optional) 
        Outputs:    None
    """
    

    # preprocess vsp vehicle file
    try:
        vehicle_vsp_data = import_from_xml(input_file)
    except IOError:
        raise IOError, "Model file %s was not found." % input_file
    
    # preprocess vsp mass properties file
    if mass_props_file:        
        try:
            mass_props_data = read_vsp_massprops(mass_props_file)  
        except IOError:
            raise IOError, "Mass-Props file %s was not found." % input_file
    else:
        mass_props = {}

    # map VSP data to SUAVE setting function
    function_type_map = {
        'fuselage' : set_fuselage,
        'Mswing'   : set_mswing,
        'Hwb'      : set_hwb,
        'Engine'   : set_engine,
        #'Blank'    : set_system, 
    }

    # sort VSP data into SUave database
    # process by component type, pull setting function from map
    for component in vehicle_vsp_data.Component_List.Component:
        
        # check function map
        if function_type_map.has_key(component.Type):
            
            # get the setting function
            set_function = function_type_map[component.Type]
            
            # set the vehicle
            set_function(vehicle,component,mass_props)
        
        # unhandled component type
        else:
            print "Unknown component %s," % component.Type
            print "Component ignored."        
        

    return
    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Setting Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
# ----------------------------------------------------------------------
#   Wings
# ----------------------------------------------------------------------
    
def set_mswing(vehicle,component,mass_props = None):
    
    # general params
    wing = set_wing(component,mass_props)
    
    # wing properties 
    wing.totals.area      = float(component.Mswing_Parms.Total_Area)
    wing.totals.span      = float(component.Mswing_Parms.Total_Span)
    wing.totals.proj_span = float(component.Mswing_Parms.Total_Proj_Span)
    wing.avg_chord        = float(component.Mswing_Parms.Avg_Chord)    
    
    # attach to vehcile
    vehicle.append_component(wing)    
    
    return

def set_hwb(vehicle,component,mass_props = None):
    
    # general params
    wing = set_wing(component,mass_props)
    
    # wing properties 
    wing.totals.area      = float(component.Hwb_Parms.Total_Area)
    wing.totals.span      = float(component.Hwb_Parms.Total_Span)
    wing.totals.proj_span = float(component.Hwb_Parms.Total_Proj_Span)
    wing.avg_chord        = float(component.Hwb_Parms.Avg_Chord)    
    
    # attach to vehcile
    vehicle.append_component(wing)    
    
    return
    
def set_wing(component,mass_props = None):
    
    # new wing
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = str(component.General_Parms.Name)
    
    # geometric properties 
    wing.origin      = [ float(component.General_Parms.Tran_X) , 
                         float(component.General_Parms.Tran_Y) , 
                         float(component.General_Parms.Tran_Z)  ]
    wing.aero_center = [ float(component.General_Parms.AeroCenter_X) , 
                         float(component.General_Parms.AeroCenter_Y) ,  
                         float(component.General_Parms.AeroCenter_Z)  ]
    wing.ref.area    = float(component.General_Parms.RefArea)
    wing.ref.span    = float(component.General_Parms.RefSpan)
    wing.ref.chord   = float(component.General_Parms.RefCbar)
    
    # segments
    # vsp section = suave segment
    segments = wing.segments
    for i,section in enumerate(component.Section_List.Section):
        
        # new wing segment
        s = SUAVE.Components.Wings.Wing.Segment()
        
        # geometric properies
        s.tag        = 'Segment_%i'%i
        s.AR         = float(section["AR"])
        s.TR         = float(section["TR"])
        s.area       = float(section["Area"])
        s.span       = float(section["Span"])
        s.TC         = float(section["TC"])
        s.RC         = float(section["RC"])
        s.sweep      = float(section["Sweep"])
        s.sweep_loc  = float(section["SweepLoc"])
        s.twist      = float(section["Twist"])
        s.twist_loc  = float(section["TwistLoc"])
        s.dihedral   = float(section["Dihedral"]) 
        s.rot_sec_dihed = float(section["DihedRotFlag"])==1
        
        # append to wing
        segments.append(s)
        
    #: for each segment
    
    # airfoils
    # vsp cross section = suave section
    sections = wing.Sections
    for i,airfoil in enumerate(component.Airfoil_List.Airfoil):
        
        a = SUAVE.Components.Wings.Wing.Airfoil()
        
        # geometric properies
        a.tag             = 'Airfoil_%i'%i
        a.type            = int(airfoil["Type"])
        a.inverted        = bool(int(airfoil["Inverted_Flag"]))
        a.camber          = float(airfoil["Camber"])
        a.camber_loc      = float(airfoil["Camber_Loc"])
        a.thickness       = float(airfoil["Thickness"])
        a.thickness_loc   = float(airfoil["Thickness_Loc"])
        a.radius_le       = float(airfoil["Radius_Le"])
        a.radius_te       = float(airfoil["Radius_Te"])
        a.six_series      = int(airfoil["Six_Series"])
        a.ideal_cl        = float(airfoil["Ideal_Cl"])
        a.A               = float(airfoil["A"])
        
        # section points
        if a.type == 1:
            set_naca_4series(a,airfoil)        
        elif a.type == 4:
            set_custom_airfoil(a,airfoil)
        elif a.type == 5:
            set_naca_6series(a,airfoil)
        
        # control surfaces
        a.slat_flag       = bool(int(airfoil["Slat_Flag"]))
        a.slat_shear_flag = bool(int(airfoil["Slat_Shear_Flag"]))
        a.slat_chord      = float(airfoil["Slat_Chord"])
        a.slat_angle      = float(airfoil["Slat_Angle"])
        a.flap_flag       = bool(int(airfoil["Flap_Flag"]))
        a.flap_shear_flag = bool(int(airfoil["Flap_Shear_Flag"]))
        a.flap_chord      = float(airfoil["Flap_Chord"])
        a.flap_angle      = float(airfoil["Flap_Angle"])
                        
        sections.append(a)
    
    #: for each section
    
    # check segment-section alignment
    if not len(sections)==(len(segments)+1):
        raise Exception , 'Segment-Section mismatch'    
    
    # mass properties
    if mass_props and mass_props.has_key(wing.tag):
        wing.Mass_Properties.update(mass_props[wing.tag])
        wing.Mass_Properties.density       = float(component.General_Parms.Density)
        wing.Mass_Properties.shell_density = float(component.General_Parms.ShellMassArea)         
    
    return wing
    
# ----------------------------------------------------------------------
#   Airfoils
# ----------------------------------------------------------------------
    
def set_custom_airfoil(airfoil,input_vsp):
    
    upper = read_airfoil_points( str(input_vsp['Upper_Pnts']) )
    lower = read_airfoil_points( str(input_vsp['Lower_Pnts']) )
    upper.insert(0,lower[0])
    upper = [ [v[0],0.0,v[1]] for v in upper ]
    lower = [ [v[0],0.0,v[1]] for v in lower ]
    
    curves = airfoil.Curves
    upper = SUAVE.Components.Wings.Wing.Section.Curve(tag='upper',points=upper)
    lower = SUAVE.Components.Wings.Wing.Section.Curve(tag='lower',points=lower) 
    curves.append(upper)
    curves.append(lower)    

def read_airfoil_points(points):
    
    points = ''.join(points.split())
    points = points.split(',')
    if not points[-1]: del points[-1]
    points = map(float,points)
    
    # restack list
    if len(points) % 2 > 0 : raise Exception , 'jagged airfoil point list'
    points = zip(*[iter(points)]*2)
    points = map(list,points)
    
    return points        
    
def set_naca_4series(airfoil,input_vsp):
    
    # shorten name...
    from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil import compute_naca_4series
    
    # airfoil params
    camber        = float(input_vsp["Camber"])
    camber_loc    = float(input_vsp["Camber_Loc"])
    thickness     = float(input_vsp["Thickness"])    
    
    # get airfoil points
    upper,lower = compute_naca_4series(camber,camber_loc,thickness)
    upper = [ [v[0],0.0,v[1]] for v in upper ]
    lower = [ [v[0],0.0,v[1]] for v in lower ]
    
    # make curves
    curves = airfoil.Curves
    upper = SUAVE.Components.Wings.Wing.Section.Curve(tag='upper',points=upper)
    lower = SUAVE.Components.Wings.Wing.Section.Curve(tag='lower',points=lower)             
    curves.append(upper)
    curves.append(lower)
    
    return

def set_naca_6series(airfoil,input_vsp):
    
    six_series    = int(input_vsp["Six_Series"])
    thickness     = float(input_vsp["Thickness"])      
    ideal_CL      = float(input_vsp["Ideal_Cl"])
    
    thickness = int( thickness * 100 )
    ideal_CL  = int( ideal_CL  * 10  )
    
    name = 'NACA_%i%d%02d' % (six_series,ideal_CL,thickness)
    
    try:
        airfoil_in = getattr( SUAVE.Components.Wings.Airfoils , name )
        airfoil_in = copy.deepcopy(airfoil_in)
    except AttributeError:
        raise AttributeError , 'Cannot handle airfoil type %s' % name
    
    airfoil.Curves = airfoil_in.Curves
    
    return



# ----------------------------------------------------------------------
#   Fuselages
# ----------------------------------------------------------------------
    
def set_fuselage(vehicle,component,mass_props=None):
    
    tag = str(component.General_Parms.Name)

    # name can be tagged as battery or fuel tank
    # battery
    if tag.lower().startswith('battery'):
        fuselage = SUAVE.Components.Energy.Storages.Battery(tag=tag)
    # fuel tank
    elif tag.lower().startswith(('fuel_tank','fueltank','fuel tank')):
        fuselage = SUAVE.Components.Energy.Storages.Fuel_Tank(tag=name)
    # fuselage
    else:
        fuselage = SUAVE.Components.Fuselages.Fuselage(tag=tag)

    # geometric properties
    fuselage.origin = [ float(component.General_Parms.Tran_X) , 
                        float(component.General_Parms.Tran_Y) , 
                        float(component.General_Parms.Tran_Z)  ]
    fuselage.aero_center = [ float(component.General_Parms.AeroCenter_X) , 
                             float(component.General_Parms.AeroCenter_Y) , 
                             float(component.General_Parms.AeroCenter_Z)  ]
    fuselage.length = float(component.Fuse_Parms.Fuse_Length)
    
    # cross sections
    # vsp cross section = suave section
    sections = fuselage.Sections
    for i,cross_section in enumerate(component.Cross_Section_List.Cross_Section):
        section = read_fuselage_cross(fuselage,cross_section)
        section.tag = 'Section_%i' % i
        sections.append(section)
    
    # mass properties
    if mass_props and mass_props.has_key(name):
        fuselage.Mass_Properties.update(mass_props[name])
        fuselage.Mass_Properties.density = float(component.General_Parms.Density)
        fuselage.Mass_Properties.shell_density = float(component.General_Parms.ShellMassArea)         

    # add to vehicle
    vehicle.append_component(fuselage)    

    return

FUSELAGE_TYPE_MAP = {
    0:'POINT',
    1:'CIRCLE',
    2:'ELLIPSE',
    3:'ROUND_BOX',
    4:'GENERAL'
}    
    
def read_fuselage_cross(Fuselage,Cross_Section):
        
    fuse_length = Fuselage.length
    z_offset = float(Cross_Section.Z_Offset)
    x_location = float(Cross_Section.Spine_Location)*fuse_length
    
    section_type = int(Cross_Section.OML_Parms.Type)
    section_type = FUSELAGE_TYPE_MAP[section_type]
    
    Section = SUAVE.Components.Fuselages.Fuselage.Section()
    
    Section.type = section_type
    Section.origin = [ x_location, 0.0, z_offset ]    
    
    if section_type == 'POINT':
        pass
    
    elif section_type == 'CIRCLE':
        radius = float(Cross_Section.OML_Parms.Height)
        if radius > 0.:
            Section.radius = float(Cross_Section.OML_Parms.Height) / 2.
        else:
            Section.type = 'POINT'
    
    elif section_type == 'ELLIPSE':
        height = float(Cross_Section.OML_Parms.Height)
        width = float(Cross_Section.OML_Parms.Width)
        
        if height <= 0:
            height = 1e-3
        if width <= 0:
            width = 1e-3            
            
        Section.height = height
        Section.width = width
        
        #if height > 0 and width > 0:
            #Section.height = height
            #Section.width = width
        #elif height > 0:
            #Section.type = 'LINE'
            #Section.point1 = [0.,0.,-height/2.]
            #Section.point2 = [0.,0.,+height/2.]
        #elif width > 0:
            #Section.type = 'LINE'
            #Section.point1 = [0.,0.,0.]
            #Section.point2 = [0.,width/2.,0.]            
    
    elif section_type == 'ROUND_BOX':
        raise NotImplementedError
    
    elif section_type == 'GENERAL':
        raise NotImplementedError
        
    return Section


# ----------------------------------------------------------------------
#   Engines
# ----------------------------------------------------------------------

def set_engine(vehicle,component,mass_props):
    # geometric properties 
    engine = SUAVE.Components.Propulsors.Turbofan()
    tag = str(component.General_Parms.Name)
    
    engine.origin = [float(component.General_Parms.Tran_X), 
                     float(component.General_Parms.Tran_Y), 
                     float(component.General_Parms.Tran_Z)]
    engine.aero_center = [float(component.General_Parms.AeroCenter_X), 
                          float(component.General_Parms.AeroCenter_Y), 
                          float(component.General_Parms.AeroCenter_Z)]

    # engine parameters 
    engine.type                    = int(component.Engine_Parms.Engine_Type)
    engine.length                  = float(component.Engine_Parms.Eng_Length)
    engine.cowl_length             = float(component.Engine_Parms.Cowl_Length)
    engine.eng_thrt_ratio          = float(component.Engine_Parms.Eng_Thrt_Ratio)
    engine.hilight_thrt_ratio      = float(component.Engine_Parms.Hilight_Thrt_Ratio)
    engine.lip_finess_ratio        = float(component.Engine_Parms.Lip_Finess_Ratio)
    engine.height_width_ratio      = float(component.Engine_Parms.Height_Width_Ratio)
    engine.upper_surf_shape_factor = float(component.Engine_Parms.Upper_Surf_Shape_Factor)
    engine.lower_surf_shape_factor = float(component.Engine_Parms.Lower_Surf_Shape_Factor)
    engine.dive_flap_ratio         = float(component.Engine_Parms.Dive_Flap_Ratio)

    # engine tip parameters
    engine.tip = Data()
    engine.tip.r   = float(component.Engine_Parms.Radius_Tip)
    engine.tip.max = float(component.Engine_Parms.Max_Tip)
    engine.tip.hub = float(component.Engine_Parms.Hub_Tip)        

    # engine inlet parameters
    engine.inlet = Data()
    engine.inlet.type            = int(component.Engine_Parms.Inlet_Type)
    engine.inlet.xy_sym_flag     = bool(component.Engine_Parms.Inlet_XY_Sym_Flag)
    engine.inlet.half_split_flag = bool(component.Engine_Parms.Inlet_Half_Split_Flag)
    engine.inlet.x_axis_rot      = float(component.Engine_Parms.Inlet_X_Axis_Rot)
    engine.inlet.scarf_angle     = float(component.Engine_Parms.Inlet_Scarf_Angle)
    
    engine.inlet.duct = Data()
    engine.inlet.duct.on_off       = bool(component.Engine_Parms.Inlet_Duct_On_Off)
    engine.inlet.duct.offset       = [ float(component.Engine_Parms.Inlet_Duct_X_Offset) ,
                                       float(component.Engine_Parms.Inlet_Duct_Y_Offset) ]
    engine.inlet.duct.shape_factor = float(component.Engine_Parms.Inlet_Duct_Shape_Factor)

    # engine diverter parameters
    engine.divertor = Data()
    engine.divertor.on_off = bool(component.Engine_Parms.Divertor_On_Off)
    engine.divertor.height = float(component.Engine_Parms.Divertor_Height)
    engine.divertor.length = float(component.Engine_Parms.Divertor_Length)

    # engine nozzle parameters
    engine.nozzle = Data()
    engine.nozzle.type               = int(component.Engine_Parms.Nozzle_Type)
    engine.nozzle.length             = float(component.Engine_Parms.Nozzle_Length)
    engine.nozzle.exit_area_ratio    = float(component.Engine_Parms.Exit_Area_Ratio)
    engine.nozzle.height_width_ratio = float(component.Engine_Parms.Nozzle_Height_Width_Ratio)
    engine.nozzle.exit_throat_ratio  = float(component.Engine_Parms.Exit_Throat_Ratio)     
    
    engine.nozzle.duct = Data()
    engine.nozzle.duct.on_off        = bool(component.Engine_Parms.Nozzle_Duct_On_Off)
    engine.nozzle.duct.offset        = [ float(component.Engine_Parms.Nozzle_Duct_X_Offset), 
                                         float(component.Engine_Parms.Nozzle_Duct_Y_Offset)]
    engine.nozzle.duct.shape_factor  = float(component.Engine_Parms.Nozzle_Duct_Shape_Factor)


    # mass properties
    if mass_props and mass_props.has_key(name):
        fuselage.Mass_Properties.update(mass_props[name])
        fuselage.Mass_Properties.density = float(component.General_Parms.Density)
        fuselage.Mass_Properties.shell_density = float(component.General_Parms.ShellMassArea)         

    # attach to vehcile
    vehicle.append_component(engine)
    
# ----------------------------------------------------------------------
#   Systems
# ----------------------------------------------------------------------

def set_systems(vehicle,component,mass_props=None):
    raise NotImplementedError


# ----------------------------------------------------------------------
#   Mass Properties
# ----------------------------------------------------------------------

def read_vsp_massprops(filename):

    # data for import
    data_out = Container()
    
    # open file
    file_in = open(filename)
    
    # process each line
    for line in file_in:
        
        # split line
        data = line.split()
        
        # dump data
        mass_props = SUAVE.Components.Mass_Properties()
        mass_props['tag'] = this_data[0]
        
        data = map(float,data[1:])
        
        mass_props["mass"]   = data[1]
        mass_props["volume"] = data[11]
        mass_props["center_of_gravity"] = data[2:4+1]
        mass_props["Moments_Of_Inertia"]["tensor"] = [ [ data[i] for i in (5,8,9)  ] ,
                                 [ data[i] for i in (8,6,10) ] ,
                                 [ data[i] for i in (9,10,7) ] ]
    
    #: for each line
    
    # done
    file_in.close()
    return mass_props

    