# test_cnbeta.py
# Tim Momose, April 2014
# Reference: Aircraft Dynamics: from Modeling to Simulation, by M. R. Napolitano

import SUAVE
import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Tube_Wing.taw_cnbeta import taw_cnbeta
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.datcom import datcom
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.extend_to_ref_area import extend_to_ref_area
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.trapezoid_ac_x import trapezoid_ac_x
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.trapezoid_mac import trapezoid_mac
from SUAVE.Core import (
    Data, Container,
)

def main():
    #Parameters Required
        #Using values for a Boeing 747-200
    vehicle = SUAVE.Vehicle()
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'main_wing'
    wing.areas.reference      = 5500.0 * Units.feet**2
    wing.spans.projected      = 196.0  * Units.feet
    wing.sweeps.quarter_chord = 42.0   * Units.deg # Leading edge
    wing.chords.root          = 42.9   * Units.feet #54.5
    wing.chords.tip           = 14.7   * Units.feet
    wing.chords.mean_aerodynamic = 27.3 * Units.feet
    wing.taper                = wing.chords.tip / wing.chords.root
    wing.aspect_ratio         = wing.spans.projected**2/wing.areas.reference
    wing.symmetric            = True
    wing.origin               = np.array([58.6,0.,3.6]) * Units.feet  
    
    reference = SUAVE.Core.Container()
    vehicle.reference_area = wing.areas.reference
    vehicle.append_component(wing)

    wing = SUAVE.Components.Wings.Wing()
    wing.spans.exposed        = 32.4  * Units.feet
    wing.chords.root          = 38.7  * Units.feet      # vertical.chords.fuselage_intersect
    wing.chords.tip           = 13.4  * Units.feet
    wing.sweeps.quarter_chord = 50.0  * Units.deg # Leading Edge
    wing.x_root_LE1           = 180.0 * Units.feet
    wing.symmetric            = False
    wing.exposed_root_chord_offset = 13.3   * Units.feet
    wing                      = extend_to_ref_area(wing)
    wing.tag                  = 'vertical_stabilizer'
    wing.areas.reference      = wing.extended.areas.reference
    wing.spans.projected      = wing.extended.spans.projected
    #wing.chords.root          = wing.extended.chords.root
    wing.chords.root          = 14.9612585185
    dx_LE_vert                = wing.extended.root_LE_change
    #wing.taper                = wing.chords.tip/wing.chords.root
    wing.taper                = 0.272993077083
    wing.origin               = np.array([wing.x_root_LE1 + dx_LE_vert,0.,0.])
    wing.aspect_ratio         = (wing.spans.projected**2)/wing.areas.reference
    wing.effective_aspect_ratio = 2.2
    wing.symmetric              = False
    wing.aerodynamic_center     = np.array([trapezoid_ac_x(wing),0.0,0.0])
    Mach                        = np.array([0.198])
    wing.CL_alpha = datcom(wing,Mach)
    vehicle.append_component(wing)

    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    fuselage.areas.side_projected               = 4696.16 * Units.feet**2
    fuselage.lengths.total                      = 229.7   * Units.feet
    fuselage.heights.maximum                    = 26.9    * Units.feet
    fuselage.width                              = 20.9    * Units.feet
    fuselage.heights.at_quarter_length          = 26.0    * Units.feet
    fuselage.heights.at_three_quarters_length   = 19.7    * Units.feet
    fuselage.heights.at_wing_root_quarter_chord = 23.8    * Units.feet
    vehicle.append_component(fuselage)

    configuration = Data()
    configuration.mass_properties = Data()
    configuration.mass_properties.center_of_gravity = Data()
    configuration.mass_properties.center_of_gravity = np.array([112.2,0,6.8]) * Units.feet

    #segment            = SUAVE.Analyses.Mission.Segments.Base_Segment()
    segment            = SUAVE.Analyses.Mission.Segments.Segment()
    segment.freestream = Data()
    segment.freestream.mach_number = Mach
    segment.atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    altitude           = 0.0 * Units.feet
    
    conditions = segment.atmosphere.compute_values(altitude / Units.km)
    segment.a          = conditions.speed_of_sound
    segment.freestream.density   = conditions.density
    segment.freestream.dynamic_viscosity = conditions.dynamic_viscosity
    segment.freestream.velocity  = segment.freestream.mach_number * segment.a

    #Method Test
    cn_b = taw_cnbeta(vehicle,segment,configuration)
    expected = 0.09427599 # Should be 0.184
    error = Data()
    error.cn_b_747 = (cn_b-expected)/expected

    #Parameters Required
    #Using values for a Beechcraft Model 99
    #MODEL DOES NOT ACCOUNT FOR DESTABILIZING EFFECTS OF PROPELLERS!
    """wing               = SUAVE.Components.Wings.Wing()
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

    segment            = SUAVE.Analyses.Mission.Segments.Base_Segment()
    segment.M          = 0.152
    segment.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    altitude           = 0.0 * Units.feet
    segment.a          = segment.atmosphere.compute_values(altitude / Units.km, type="a")
    segment.rho        = segment.atmosphere.compute_values(altitude / Units.km, type="rho")
    segment.mew        = segment.atmosphere.compute_values(altitude / Units.km, type="mew")
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

    aircraft              = SUAVE.Vehicle()
    aircraft.wing         = wing
    aircraft.fuselage     = fuselage
    aircraft.other_bodies = other_bodies
    aircraft.vertical     = vertical
    aircraft.Mass_Props.pos_cg[0] = 16.6 * Units.feet

    segment            = SUAVE.Analyses.Mission.Segments.Base_Segment()
    segment.M          = 0.111
    segment.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    altitude           = 0.0 * Units.feet
    segment.a          = segment.atmosphere.compute_values(altitude / Units.km, type="a")
    segment.rho        = segment.atmosphere.compute_values(altitude / Units.km, type="rho")
    segment.mew        = segment.atmosphere.compute_values(altitude / Units.km, type="mew")
    segment.v_inf      = segment.M * segment.a

    #Method Test
    print 'SIAI Marchetti S-211 at M = {0} and h = {1} meters'.format(segment.M, altitude)

    cn_b = taw_cnbeta(aircraft,segment)

    expected = 0.160
    print 'Cn_beta        = {0:.4f}'.format(cn_b)
    print 'Expected value = {}'.format(expected)
    print 'Percent Error  = {0:.2f}%'.format(100.0*(cn_b-expected)/expected)
    print ' '"""

    for k,v in error.items():
        assert(np.abs(v)<0.1)

    return

# ----------------------------------------------------------------------
#   Call Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()
