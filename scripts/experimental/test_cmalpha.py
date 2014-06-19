# test_cmalpha.py
# Tim Momose, April 2014
# Reference: Aircraft Dynamics: from Modeling to Simulation, by M. R. Napolitano

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.datcom import datcom
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.trapezoid_mac import trapezoid_mac
#from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.convert_sweep import convert_sweep
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.trapezoid_ac_x import trapezoid_ac_x
#from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approsimations.Supporting_Functions.extend_to_ref_area import extend_to_ref_area
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Tube_Wing.taw_cmalpha import taw_cmalpha
from SUAVE.Attributes import Units as Units
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)

#Parameters Required
#Using values for a Boeing 747-200  
wing                = SUAVE.Components.Wings.Wing()
wing.area           = 5500.0 * Units.feet**2
wing.span           = 196.0  * Units.feet
wing.sweep_le       = 42.0   * Units.deg
wing.taper          = 14.7/54.5
wing.aspect_ratio   = wing.span**2/wing.area
wing.symmetric      = True
wing.x_LE           = 58.6   * Units.feet
wing.x_ac_LE        = 112. * Units.feet - wing.x_LE
wing.eta            = 1.0
wing.downwash_adj   = 1.0

Mach                    = 0.198
reference               = SUAVE.Structure.Container()
reference.area          = wing.area
reference.mac           = 27.3 * Units.feet
reference.CL_alpha_wing = datcom(wing,Mach)
wing.CL_alpha           = reference.CL_alpha_wing

horizontal          = SUAVE.Components.Wings.Wing()
horizontal.area     = 1490.55* Units.feet**2
horizontal.span     = 71.6   * Units.feet
horizontal.sweep_le = 44.0   * Units.deg
horizontal.taper    = 7.5/32.6
horizontal.aspect_ratio = horizontal.span**2/horizontal.area
horizontal.x_LE     = 187.0  * Units.feet
horizontal.symmetric= True
horizontal.eta      = 0.95
horizontal.downwash_adj = 1.0 - 2.0*reference.CL_alpha_wing/np.pi/wing.aspect_ratio
horizontal.x_ac_LE  = trapezoid_ac_x(horizontal)
horizontal.CL_alpha = datcom(horizontal,Mach)

Lifting_Surfaces    = []
Lifting_Surfaces.append(wing)
Lifting_Surfaces.append(horizontal)

fuselage            = SUAVE.Components.Fuselages.Fuselage()
fuselage.x_root_quarter_chord = 77.0 * Units.feet
fuselage.length     = 229.7  * Units.feet
fuselage.w_max      = 20.9   * Units.feet 

aircraft                  = SUAVE.Vehicle()
aircraft.reference        = reference
aircraft.Lifting_Surfaces = Lifting_Surfaces
aircraft.fuselage         = fuselage
aircraft.Mass_Props.pos_cg[0] = 112. * Units.feet    

#Method Test
print '<<Test run of the taw_cmalpha() method>>'
print '--Boeing 747 at Mach {0}--'.format(Mach)

cm_a = taw_cmalpha(aircraft,Mach)

expected = -1.45
print 'Cm_alpha       = {0:.4f}'.format(cm_a)
print 'Expected value = {}'.format(expected)
print 'Percent Error  = {0:.2f}%'.format(100.0*(cm_a-expected)/expected)
print 'Static Margin  = {0:.4f}'.format(-cm_a/reference.CL_alpha_wing)
print ' '


#Parameters Required
#Using values for a Beech 99  
wing                = SUAVE.Components.Wings.Wing()
wing.area           = 280.0 * Units.feet**2
wing.span           = 46.0  * Units.feet
wing.sweep_le       = 3.0   * Units.deg
wing.taper          = 0.47
wing.aspect_ratio   = wing.span**2/wing.area
wing.x_LE           = 14.0  * Units.feet
wing.symmetric      = True
wing.eta            = 1.0
wing.downwash_adj   = 1.0
wing.x_ac_LE        = trapezoid_ac_x(wing)

Mach                    = 0.152
reference               = SUAVE.Structure.Container()
reference.area          = wing.area
reference.mac           = 6.5  * Units.feet
reference.CL_alpha_wing = datcom(wing,Mach)
wing.CL_alpha           = reference.CL_alpha_wing

horizontal              = SUAVE.Components.Wings.Wing()
horizontal.area         = 100.5 * Units.feet**2
horizontal.span         = 22.5  * Units.feet
horizontal.sweep_le     = 21.0  * Units.deg
horizontal.taper        = 3.1/6.17
horizontal.aspect_ratio = horizontal.span**2/horizontal.area
horizontal.x_LE         = 36.3  * Units.feet
horizontal.symmetric    = True
horizontal.eta          = 0.95
horizontal.downwash_adj = 1 - 2*reference.CL_alpha_wing/np.pi/wing.aspect_ratio
horizontal.x_ac_LE      = trapezoid_ac_x(horizontal)
horizontal.CL_alpha     = datcom(horizontal,Mach)

Lifting_Surfaces    = []
Lifting_Surfaces.append(wing)
Lifting_Surfaces.append(horizontal)

fuselage            = SUAVE.Components.Fuselages.Fuselage()
fuselage.x_root_quarter_chord = 5.4 * Units.feet
fuselage.length     = 44.0  * Units.feet
fuselage.w_max      = 5.4   * Units.feet

aircraft                  = SUAVE.Vehicle()
aircraft.reference        = reference
aircraft.Lifting_Surfaces = Lifting_Surfaces
aircraft.fuselage         = fuselage
aircraft.Mass_Props.pos_cg[0] = 17.2 * Units.feet    

#Method Test
print '--Beech 99 at Mach {0}--'.format(Mach)

cm_a = taw_cmalpha(aircraft,Mach)

expected = -2.08
print 'Cm_alpha       = {0:.4f}'.format(cm_a)
print 'Expected value = {}'.format(expected)
print 'Percent Error  = {0:.2f}%'.format(100.0*(cm_a-expected)/expected)
print 'Static Margin  = {0:.4f}'.format(-cm_a/reference.CL_alpha_wing)
print ' '



#Parameters Required
#Using values for an SIAI Marchetti S-211
wing               = SUAVE.Components.Wings.Wing()
wing.area          = 136.0 * Units.feet**2
wing.span          = 26.3  * Units.feet
wing.sweep_le      = 19.5  * Units.deg
wing.taper         = 3.1/7.03
wing.aspect_ratio  = wing.span**2/wing.area
wing.x_LE          = 12.11 * Units.feet
wing.symmetric     = True
wing.eta           = 1.0
wing.downwash_adj  = 1.0
wing.x_ac_LE       = 16.6 * Units.feet - wing.x_LE

Mach                    = 0.111
reference               = SUAVE.Structure.Container()
reference.area          = wing.area
reference.mac           = 5.4 * Units.feet
reference.CL_alpha_wing = datcom(wing,Mach) 
wing.CL_alpha           = reference.CL_alpha_wing

fuselage           = SUAVE.Components.Fuselages.Fuselage()
fuselage.length    = 30.9    * Units.feet
fuselage.w_max     = ((2.94+5.9)/2) * Units.feet
fuselage.x_root_quarter_chord = 12.67 * Units.feet

horizontal              = SUAVE.Components.Wings.Wing()
horizontal.area         = 36.46 * Units.feet**2
horizontal.span         = 13.3 * Units.feet
horizontal.sweep_le     = 18.5 * Units.deg
horizontal.taper        = 1.6/3.88
horizontal.aspect_ratio = horizontal.span**2/horizontal.area
horizontal.x_LE         = 26.07 * Units.feet
horizontal.symmetric    = True
horizontal.eta          = 0.9
horizontal.downwash_adj = 1 - 2*reference.CL_alpha_wing/np.pi/wing.aspect_ratio
horizontal.x_ac_LE      = trapezoid_ac_x(horizontal)
horizontal.CL_alpha     = datcom(horizontal,Mach)

Lifting_Surfaces    = []
Lifting_Surfaces.append(wing)
Lifting_Surfaces.append(horizontal)

aircraft                  = SUAVE.Vehicle()
aircraft.reference        = reference
aircraft.Lifting_Surfaces = Lifting_Surfaces
aircraft.fuselage         = fuselage
aircraft.Mass_Props.pos_cg[0] = 16.6 * Units.feet 


#Method Test
print '--SIAI Marchetti S-211 at Mach {0}--'.format(Mach)

cm_a = taw_cmalpha(aircraft,Mach)

expected = -0.6
print 'Cm_alpha       = {0:.4f}'.format(cm_a)
print 'Expected value = {}'.format(expected)
print 'Percent Error  = {0:.2f}%'.format(100.0*(cm_a-expected)/expected)
print 'Static Margin  = {0:.4f}'.format(-cm_a/reference.CL_alpha_wing)
print ' '