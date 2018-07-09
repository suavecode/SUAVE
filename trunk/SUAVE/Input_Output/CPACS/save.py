## @ingroup Input_Output-CPACS
# save.py
#
# Created: J. Jepsen Jul 2018
# Updated:  

""" SUAVE Methods for IO """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Input_Output.CPACS.airfoils import NACA0009
from SUAVE.Input_Output.CPACS.profiles import create_circle_list

from warnings import warn
from datetime import datetime
from math import pi, tan

# Try to load lxml library for handling XML data
try:
    import lxml
    from lxml import objectify

    E = objectify.E
except ImportError:
    warn('CPACS.save.py: No suitable XML package found. Install lxml package.', ImportWarning)


# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
## @ingroup Input_Output-XML
def save(data, filename):
    """This creates a CPACS file based on an python dict data structure.

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    data   The python dict data structure to be saved
    filename   The name of the saved file

    Outputs:
    CPACS file with name as specified by filename

    Properties Used:
    N/A
    """
    ## collecting all required data
    aircraft_name = data.base._base.tag
    reference_area = data.base._base.reference_area
    # collect wings data
    wings = []
    for wing in data.base._base.wings:
        entry = {'name': wing.tag,
                 'origin': [val for val in wing.origin],
                 'position': [val for val in wing.position],
                 'dihedral': wing.dihedral,
                 'sweeps': {'quarter_chord': wing.sweeps.quarter_chord},
                 'chords': {'root': wing.chords.root,
                            'tip': wing.chords.tip},
                 'taper': wing.taper,
                 'span': wing.spans.projected,
                 'twists': {'root': wing.twists.root,
                            'tip': wing.twists.tip},
                 'symmetric': wing.symmetric,
                 'vertical': wing.vertical}
        wings.append(entry)

    ## putting data in CPACS structure
    # create new CPACS node with filled header
    root = E.cpacs(E.header(E.cpacsVersion("2.3.1"),
                            E.creator("SUAVE exporter"),
                            E.name(aircraft_name),
                            E.timestamp(datetime.strftime(datetime.now(), '%Y-%m-%dT%H:%M:%S')),
                            E.version("1.0")))

    # add aircraft model to the structure
    ac_path = objectify.ObjectPath('cpacs.vehicles.aircraft.model')
    ac_model = E.model(E.reference(E.area(reference_area)),
                       uID=aircraft_name)
    ac_path.addattr(root, ac_model)

    # create a standard wing airfoil
    airfoil_name = 'NACA0009'
    wing_airfoils_path = objectify.ObjectPath('cpacs.vehicles.profiles.wingAirfoils.wingAirfoil')
    wing_airfoils_path.addattr(root, E.wingAirfoil(E.name(airfoil_name),
                                                   E.pointList(E.x(';'.join([str(p[0]) for p in NACA0009]),
                                                                   mapType="vector"),
                                                               E.y(';'.join([str(0.0) for _ in range(len(NACA0009))]),
                                                                   mapType="vector"),
                                                               E.z(';'.join([str(p[1]) for p in NACA0009]),
                                                                   mapType="vector")),
                                                   uID=airfoil_name))

    # create a circular fuselage profile
    profile_name = 'Circle'
    xs, ys, zs = create_circle_list()
    fuselage_profiles_path = objectify.ObjectPath('cpacs.vehicles.profiles.fuselageProfiles.fuselageProfile')
    fuselage_profiles_path.addattr(root, E.fuselageProfile(E.name(profile_name),
                                                           E.pointList(E.x(';'.join([str(val) for val in xs]),
                                                                           mapType="vector"),
                                                                       E.y(';'.join([str(val) for val in ys]),
                                                                           mapType="vector"),
                                                                       E.z(';'.join([str(val) for val in zs]),
                                                                           mapType="vector")),
                                                           uID=profile_name))

    # create wings
    wing_path = objectify.ObjectPath('cpacs.vehicles.aircraft.model.wings.wing')
    for wing in wings:
        section_1_uid = wing['name'] + '_Sec1'
        section_2_uid = wing['name'] + '_Sec2'
        element_1_uid = wing['name'] + '_Sec1_Elem1'
        element_2_uid = wing['name'] + '_Sec2_Elem1'
        segment_uid = wing['name'] + '_Seg1'
        half_span = wing['span'] / 2.
        if not wing['symmetric']:
            half_span = wing['span']
        dx = 0.25 * wing['chords']['root'] + tan(wing['sweeps']['quarter_chord']) * half_span \
             - 0.25 * wing['chords']['root'] * wing['taper']
        dz = tan(wing['dihedral']) * half_span
        wing_node = E.wing(E.name(wing['name']),
                           E.parentUID(),
                           E.transformation(E.translation(E.x(wing['origin'][0] + wing['position'][0]),
                                                          E.y(wing['origin'][1] + wing['position'][1]),
                                                          E.z(wing['origin'][2] + wing['position'][2])),
                                            E.rotation(E.x(90.0 if wing['vertical'] else 0.0),
                                                       E.y(0.0),
                                                       E.z(0.0))),
                           E.sections(E.section(E.name(section_1_uid),
                                                E.transformation(E.translation(E.x(0.0),
                                                                               E.y(0.0),
                                                                               E.z(0.0))),
                                                E.elements(E.element(E.name(element_1_uid),
                                                                     E.transformation(E.translation(E.x(0.0),
                                                                                                    E.y(0.0),
                                                                                                    E.z(0.0)),
                                                                                      E.rotation(E.x(0.0),
                                                                                                 E.y(wing['twists'][
                                                                                                         'root']),
                                                                                                 E.z(0.0)),
                                                                                      E.scaling(
                                                                                          E.x(wing['chords']['root']),
                                                                                          E.y(1.0),
                                                                                          E.z(wing['chords']['root']))),
                                                                     E.airfoilUID(airfoil_name),
                                                                     uID=element_1_uid)),
                                                uID=section_1_uid),
                                      E.section(E.name(section_2_uid),
                                                E.transformation(E.translation(E.x(0.0),
                                                                               E.y(0.0),
                                                                               E.z(0.0))),
                                                E.elements(E.element(E.name(element_2_uid),
                                                                     E.transformation(
                                                                         E.translation(E.x(dx),
                                                                                       E.y(half_span),
                                                                                       E.z(dz)),  # calc from dihedral
                                                                         E.rotation(E.x(0.0),
                                                                                    E.y(wing['twists']['tip']),
                                                                                    E.z(0.0)),
                                                                         E.scaling(E.x(wing['chords']['root']*
                                                                                       wing['taper']),
                                                                                   E.y(1.0),
                                                                                   E.z(wing['chords']['root']*
                                                                                       wing['taper']))),
                                                                     E.airfoilUID(airfoil_name),
                                                                     uID=element_2_uid)),
                                                uID=section_2_uid), ),
                           E.segments(E.segment(E.name(segment_uid),
                                                E.fromElementUID(element_1_uid),
                                                E.toElementUID(element_2_uid),
                                                uID=segment_uid)),
                           uID=wing['name'])
        if wing['symmetric']:
            wing_node.attrib['symmetry'] = 'x-z-plane'
        wing_path.addattr(root, wing_node)

    # TODO: create fuselages

    # deannotate tree structure
    objectify.deannotate(root, xsi_nil=True)

    # save CPACS structure to file
    with open(filename, 'w') as outfile:
        outfile.write(lxml.etree.tostring(root, pretty_print=True))

    return
