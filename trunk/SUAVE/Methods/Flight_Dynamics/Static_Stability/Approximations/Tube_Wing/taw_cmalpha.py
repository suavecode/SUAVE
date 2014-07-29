# taw_cmalpha.py
#
# Created:  Tim Momose, April 2014
# Modified: Andrew Wendorff, July 2014

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.datcom import datcom
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.convert_sweep import convert_sweep
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.trapezoid_ac_x import trapezoid_ac_x
from SUAVE.Attributes import Units as Units
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

def taw_cmalpha(geometry,mach,conditions,configuration):
    """ cm_alpha = SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Tube_Wing.taw_cmalpha(configuration,conditions)
        This method computes the static longitudinal stability derivative for a
        standard Tube-and-Wing aircraft configuration.
        
        Inputs:
            configuration - a data dictionary with the fields:
                reference - a data dictionary with the fields:
                    area - the reference wing area [meters**2]
                    mac  - the main wing mean aerodynamic chord [meters]
                    CL_alpha_wing - the main wing lift curve slope [dimensionless]
                Mass_Props - a data dictionary with the field:
                    pos_cg - A vector in 3-space indicating CG position [meters]
                Lifting_Surfaces - a list containing data dictionaries
                representing all non-vertical lifting surfaces. Each of these
                lifting surface data dictionaries has the fields:
                    x_LE - the x-coordinate of the surface root LE position 
                    [meters] (root at fuselage symmetry plane)
                    x_ac_LE - the x-coordinate of the surface's aerodynamic 
                    center measured relative to the root leading edge (root
                    of reference area - at symmetry plane) [meters]
                    area - lifting surface planform area [meters**2]
                    span - span of the lifting surface [meters]
                    sweep_le - sweep of the leading edge [radians]
                    taper - taper ratio [dimensionless]
                    aspect_ratio - aspect ratio [dimensionless]
                    CL_alpha - the surface's lift curve slope [dimensionless]
                    eta - lifting efficiency. Use to indicate effect of surfaces
                    ahead of the lifting surface [dimensionless]
                fuselage - a data dictionary with the fields:
                    x_root_quarter_chord - x coordinate of the quarter-chord of 
                    the wing root (here, the wing root is where the wing 
                    intersects the body) [meters]
                    w_max - maximum width of the fuselage [meters]
                    length - length of the fuselage [meters]
            mach - flight Mach number
    
        Outputs:
            cm_alpha - a single float value: The static longidutinal stability
            derivative (d(Cm_cg)/d(alpha))
                
        Assumptions:
            -This method assumes a tube-and-wing configuration
            -April 8, 2014 - The current version only accounts for the effect of
            downwash on the lift curve slopes of lifting surfaces behind other
            lifting surfaces by an efficiency factor.
    """   

    # Unpack inputs
    Sref  = geometry.reference_area
    mac   = geometry.Wings['Main Wing'].chord_mac
    C_Law = conditions.lift_curve_slope
    x_cg  = configuration.mass_props.pos_cg[0]
    x_rqc = geometry.Wings['Main Wing'].origin[0]
    w_f   = geometry.Fuselages.Fuselage.width
    l_f   = geometry.Fuselages.Fuselage.length_total
    M     = mach
    
    #Evaluate the effect of each lifting surface in turn
    CmAlpha_surf = []
    for surf in geometry.Wings:
        #Unpack inputs
        s         = surf.sref
        x_surf    = surf.origin[0]
        x_ac_surf = surf.aero_center[0]
        eta       = surf.eta
        downw     = 1 - surf.ep_alpha
        CL_alpha  = surf.CL_alpha
        #Calculate Cm_alpha contributions
        l_surf    = x_surf + x_ac_surf - x_cg
        Cma       = -l_surf*s/(mac*Sref)*(CL_alpha)*eta*downw
        CmAlpha_surf.append(Cma)
        #For debugging
        #print "Cmalpha_surf: {:.4f}".format(Cma)
    
    #Evaluate the effect of the fuselage on the stability derivative
    p  = x_rqc/l_f
    Kf = 1.5012*p**2. + 0.538*p + 0.0331
    CmAlpha_body = Kf*w_f**2*l_f/Sref/mac   #NEGLECTS TAIL EFFECT ON CL_ALPHA
    
    cm_alpha = sum(CmAlpha_surf) + CmAlpha_body
    
    return cm_alpha

# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':
    from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.extend_to_ref_area import extend_to_ref_area
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
    print 'Boeing 747 at Mach {0}'.format(Mach)
    
    cm_a = taw_cmalpha(aircraft,Mach)
    
    expected = -1.45
    print 'Cm_alpha       = {0:.4f}'.format(cm_a)
    print 'Expected value = {}'.format(expected)
    print 'Percent Error  = {0:.2f}%'.format(100.0*(cm_a-expected)/expected)
    print 'Static Margin  = {0:.4f}'.format(-cm_a/reference.CL_alpha_wing)
    print ' '