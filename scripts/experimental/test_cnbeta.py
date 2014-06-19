# test_cnbeta.py

import SUAVE
import numpy as np
from SUAVE.Attributes import Units as Units
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Tube_Wing.taw_cnbeta import taw_cnbeta
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.datcom import datcom
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.extend_to_ref_area import extend_to_ref_area
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.trapezoid_ac_x import trapezoid_ac_x
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.trapezoid_mac import trapezoid_mac
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)


#Parameters Required
#Using values for a Beechcraft Model 99
#MODEL DOES NOT ACCOUNT FOR DESTABILIZING EFFECTS OF PROPELLERS!
wing               = SUAVE.Components.Wings.Wing()
wing.area          = 280.0 * Units.feet**2
wing.span          = 46.0  * Units.feet
wing.sweep_le      = 3.0   * Units.deg
wing.z_position    = 2.2   * Units.feet
wing.taper         = 0.46
wing.aspect_ratio  = wing.span**2/wing.area
wing.symmetric     = True

fuselage           = SUAVE.Components.Fuselages.Fuselage()
fuselage.side_area = 185.36 * Units.feet**2
fuselage.length    = 44.0   * Units.feet
fuselage.h_max     = 6.0    * Units.feet
fuselage.w_max     = 5.4    * Units.feet
fuselage.height_at_vroot_quarter_chord   = 2.9 * Units.feet
fuselage.height_at_quarter_length        = 4.8 * Units.feet
fuselage.height_at_three_quarters_length = 4.3 * Units.feet

nacelle           = SUAVE.Components.Fuselages.Fuselage()
nacelle.side_area = 34.45 * Units.feet**2
nacelle.x_front   = 7.33  * Units.feet
nacelle.length    = 14.13 * Units.feet
nacelle.h_max     = 3.68  * Units.feet
nacelle.w_max     = 2.39  * Units.feet
nacelle.height_at_quarter_length        = 3.08 * Units.feet
nacelle.height_at_three_quarters_length = 2.12 * Units.feet

other_bodies      = [nacelle,nacelle]

vertical              = SUAVE.Components.Wings.Wing()
vertical.span         = 6.6  * Units.feet
vertical.root_chord   = 8.2  * Units.feet
vertical.tip_chord    = 3.6  * Units.feet
vertical.sweep_le     = 47.0 * Units.deg
vertical.x_root_LE1   = 34.8 * Units.feet
vertical.symmetric    = False
dz_centerline         = 2.0  * Units.feet
ref_vertical          = extend_to_ref_area(vertical,dz_centerline)
vertical.span         = ref_vertical.ref_span
vertical.area         = ref_vertical.ref_area
vertical.aspect_ratio = ref_vertical.ref_aspect_ratio
vertical.x_root_LE    = vertical.x_root_LE1 + ref_vertical.root_LE_change
vertical.taper        = vertical.tip_chord/ref_vertical.ref_root_chord
vertical.effective_aspect_ratio = 1.57
vertical.x_ac_LE      = trapezoid_ac_x(vertical)

aircraft              = SUAVE.Vehicle()
aircraft.wing         = wing
aircraft.fuselage     = fuselage
aircraft.other_bodies = other_bodies
aircraft.vertical     = vertical
aircraft.Mass_Props.pos_cg[0] = 17.2 * Units.feet

segment            = SUAVE.Attributes.Missions.Segments.Segment()
segment.M          = 0.152
segment.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
altitude           = 0.0 * Units.feet
segment.compute_atmosphere(altitude / Units.km)
segment.v_inf      = segment.M * segment.a

#Method Test
expected = 0.12
print 'Beech 99 at M = {0} and h = {1} meters'.format(segment.M, altitude)
cn_b = taw_cnbeta(aircraft,segment)

print 'Cn_beta        = {0:.4f}'.format(cn_b)
print 'Expected value = {}'.format(expected)
print 'Percent Error  = {0:.2f}%'.format(100.0*(cn_b-expected)/expected)
print ' '  


#Parameters Required
#Using values for a Boeing 747-200
#Parameters Required
#Using values for a Boeing 747-200
wing                = SUAVE.Components.Wings.Wing()
wing.area           = 5500.0 * Units.feet**2
wing.span           = 196.0  * Units.feet
wing.sweep_le       = 42.0   * Units.deg
wing.z_position     = 3.6    * Units.feet
wing.symmetric      = True
wing.taper          = 14.7/54.5
wing.aspect_ratio   = wing.span**2/wing.area

fuselage            = SUAVE.Components.Fuselages.Fuselage()
fuselage.side_area  = 4696.16 * Units.feet**2
fuselage.length     = 229.7   * Units.feet
fuselage.h_max      = 26.9    * Units.feet
fuselage.w_max      = 20.9    * Units.feet
fuselage.height_at_vroot_quarter_chord   = 15.8 * Units.feet
fuselage.height_at_quarter_length        = 26   * Units.feet
fuselage.height_at_three_quarters_length = 19.7 * Units.feet

other_bodies        = []

vertical              = SUAVE.Components.Wings.Wing()
vertical.span         = 32.4   * Units.feet
vertical.root_chord   = 38.7   * Units.feet
vertical.tip_chord    = 13.4   * Units.feet
vertical.sweep_le     = 50.0   * Units.deg
vertical.x_root_LE1   = 180.0  * Units.feet
vertical.symmetric    = False
dz_centerline         = 13.3   * Units.feet
ref_vertical          = extend_to_ref_area(vertical,dz_centerline)
vertical.span         = ref_vertical.ref_span
vertical.area         = ref_vertical.ref_area
vertical.aspect_ratio = ref_vertical.ref_aspect_ratio
vertical.x_root_LE    = vertical.x_root_LE1 + ref_vertical.root_LE_change
vertical.taper        = vertical.tip_chord/ref_vertical.ref_root_chord
vertical.effective_aspect_ratio = 2.2
vertical.x_ac_LE      = trapezoid_ac_x(vertical)

aircraft              = SUAVE.Vehicle()
aircraft.wing         = wing
aircraft.fuselage     = fuselage
aircraft.other_bodies = other_bodies
aircraft.vertical     = vertical
aircraft.Mass_Props.pos_cg[0] = 112. * Units.feet

segment            = SUAVE.Attributes.Missions.Segments.Segment()
segment.M          = 0.198
segment.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
altitude           = 0.0 * Units.feet
segment.compute_atmosphere(altitude / Units.km)
segment.v_inf      = segment.M * segment.a

#Method Test
print ' '
print 'Boeing 747 at M = {0} and h = {1} meters'.format(segment.M, altitude)

cn_b = taw_cnbeta(aircraft,segment)

expected = 0.184
print 'Cn_beta        = {0:.4f}'.format(cn_b)
print 'Expected value = {}'.format(expected)
print 'Percent Error  = {0:.2f}%'.format(100.0*(cn_b-expected)/expected)
print ' '  


#Parameters Required
#Using values for an SIAI Marchetti S-211
wing               = SUAVE.Components.Wings.Wing()
wing.area          = 136.0 * Units.feet**2
wing.span          = 26.3  * Units.feet
wing.sweep_le      = 19.5  * Units.deg
wing.z_position    = -1.1  * Units.feet
wing.taper         = 3.1/7.03
wing.aspect_ratio  = wing.span**2/wing.area

fuselage           = SUAVE.Components.Fuselages.Fuselage()
fuselage.side_area = 116.009 * Units.feet**2
fuselage.length    = 30.9    * Units.feet
fuselage.h_max     = 5.1     * Units.feet
fuselage.w_max     = 5.9     * Units.feet
fuselage.height_at_vroot_quarter_chord   = 4.1 * Units.feet
fuselage.height_at_quarter_length        = 4.5 * Units.feet
fuselage.height_at_three_quarters_length = 4.3 * Units.feet

other_bodies       = []

vertical              = SUAVE.Components.Wings.Wing()
vertical.span         = 5.8   * Units.feet
vertical.root_chord   = 5.7   * Units.feet
vertical.tip_chord    = 2.0   * Units.feet
vertical.sweep_le     = 40.2  * Units.deg
vertical.x_root_LE1   = 22.62 * Units.feet
vertical.symmetric    = False
dz_centerline         = 2.9   * Units.feet
ref_vertical          = extend_to_ref_area(vertical,dz_centerline)
vertical.span         = ref_vertical.ref_span
vertical.area         = ref_vertical.ref_area
vertical.aspect_ratio = ref_vertical.ref_aspect_ratio
vertical.x_root_LE    = vertical.x_root_LE1 + ref_vertical.root_LE_change
vertical.taper        = vertical.tip_chord/ref_vertical.ref_root_chord
vertical.effective_aspect_ratio = 2.65
vertical.x_ac_LE      = trapezoid_ac_x(vertical)

##TEST STUFF
#mac = trapezoid_mac(vertical_symm)
#h = SUAVE.Components.Wings.Wing()
#dzh = 0.8*Units.feet
#h.span = (13.3/2-dzh)*Units.feet
#h.root_chord = 3.5*Units.feet
#h.tip_chord = 1.6*Units.feet
#h.sweep = 18.5*Units.deg
#h.x_root_LE1 = 0.0
#ref_h = extend_to_ref_area(h,dzh)
#h.span = ref_h.ref_span
#h.area = ref_h.ref_area
#h.aspect_ratio = ref_h.ref_aspect_ratio
#h.x_root_LE = ref_h.root_LE_change
#h.taper = h.tip_chord/ref_h.ref_root_chord
#h_full = copy.deepcopy(h)
#h_full.span = 2.0*h.span
#h_full.area = 2.*h.area
#h_full.aspect_ratio = h_full.span**2/h_full.area
#h_full.x_ac_LE = trapezoid_ac_x(h_full)
#zh = 2.4*Units.feet

#print "cv: {0}; x1: {1}".format((ref_vertical.ref_root_chord-ref_vertical.ref_root_chord*(1-vertical.taper)*zh/vertical.span)/Units.feet,(h_full.x_ac_LE+h.x_root_LE)/Units.feet)
#print mac/Units.feet
#print ref_vertical.ref_root_chord/Units.feet
#print ref_vertical.root_LE_change/Units.feet

#print "SIAI S-211 Vertical Tail Geometry"
#print "Span: {0} feet; Area: {1} sq. ft.; AR_geom: {2}; taper: {3}; dx_ac: {4} feet".format(vertical.span/Units.feet,vertical.area/Units.feet**2,vertical.aspect_ratio,vertical.taper,vertical.x_ac_LE/Units.feet)

aircraft              = SUAVE.Vehicle()
aircraft.wing         = wing
aircraft.fuselage     = fuselage
aircraft.other_bodies = other_bodies
aircraft.vertical     = vertical
aircraft.Mass_Props.pos_cg[0] = 16.6 * Units.feet

segment            = SUAVE.Attributes.Missions.Segments.Segment()
segment.M          = 0.111
segment.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
altitude           = 0.0 * Units.feet
segment.compute_atmosphere(altitude / Units.km)
segment.v_inf      = segment.M * segment.a

#Method Test
print ' '
print 'SIAI Marchetti S-211 at M = {0} and h = {1} meters'.format(segment.M, altitude)

cn_b = taw_cnbeta(aircraft,segment)

expected = 0.160
print 'Cn_beta        = {0:.4f}'.format(cn_b)
print 'Expected value = {}'.format(expected)
print 'Percent Error  = {0:.2f}%'.format(100.0*(cn_b-expected)/expected)
print ' '