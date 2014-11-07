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
    mac   = geometry.wings['Main Wing'].chords.mean_aerodynamic
    c_root= geometry.wings['Main Wing'].chords.root
    taper = geometry.wings['Main Wing'].taper
    c_tip = taper*c_root
    span  = geometry.wings['Main Wing'].spans.projected
    sweep = geometry.wings['Main Wing'].sweep
    C_Law = conditions.lift_curve_slope
    w_f   = geometry.fuselages.Fuselage.width
    l_f   = geometry.fuselages.Fuselage.lengths.total
    x_cg  = configuration.mass_properties.center_of_gravity[0]
    x_rqc = geometry.wings['Main Wing'].origin[0] + 0.5*w_f*np.tan(sweep) + 0.25*c_root*(1 - (w_f/span)*(1-taper))
    M     = mach
    
    #Evaluate the effect of each lifting surface in turn
    CmAlpha_surf = []
    for surf in geometry.wings:
        #Unpack inputs
        s         = surf.areas.reference
        x_surf    = surf.origin[0]
        x_ac_surf = surf.aerodynamic_center[0]
        eta       = surf.dynamic_pressure_ratio
        downw     = 1 - surf.ep_alpha
        CL_alpha  = surf.CL_alpha
        vertical  = surf.vertical
        #Calculate Cm_alpha contributions
        l_surf    = x_surf + x_ac_surf - x_cg
        Cma       = -l_surf*s/(mac*Sref)*(CL_alpha*eta*downw)*(1. - vertical)
        CmAlpha_surf.append(Cma)
        ##For debugging
        ##print "Cmalpha_surf: {:.4f}".format(Cma[len(Cma)-1])
    
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
    #Parameters Required
    #Using values for a Boeing 747-200  
    vehicle = SUAVE.Vehicle()
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Main Wing'
    wing.areas.reference           = 5500.0 * Units.feet**2
    wing.spans.projected           = 196.0  * Units.feet
    wing.chords.mean_aerodynamic   = 27.3 * Units.feet
    wing.chords.root               = 44. * Units.feet  #54.5ft
    wing.sweep          = 42.0   * Units.deg # Leading edge
    wing.taper          = 13.85/44.  #14.7/54.5
    wing.aspect_ratio   = wing.spans.projected**2/wing.areas.reference
    wing.symmetric      = True
    wing.vertical       = False
    wing.origin         = np.array([59.,0,0]) * Units.feet  
    wing.aerodynamic_center     = np.array([112.2*Units.feet,0.,0.])-wing.origin#16.16 * Units.meters,0.,0,])np.array([trapezoid_ac_x(wing),0., 0.])#
    wing.dynamic_pressure_ratio = 1.0
    wing.ep_alpha               = 0.0
    
    Mach                        = np.array([0.198])
    conditions                  = Data()
    conditions.lift_curve_slope = datcom(wing,Mach)
    wing.CL_alpha               = conditions.lift_curve_slope
    vehicle.reference_area      = wing.areas.reference
    vehicle.append_component(wing)
    
    main_wing_CLa = wing.CL_alpha
    main_wing_ar  = wing.aspect_ratio
    
    wing                     = SUAVE.Components.Wings.Wing()
    wing.tag = 'Horizontal Stabilizer'
    wing.areas.reference     = 1490.55* Units.feet**2
    wing.spans.projected     = 71.6   * Units.feet
    wing.sweep               = 44.0   * Units.deg # leading edge
    wing.taper               = 7.5/32.6
    wing.aspect_ratio        = wing.spans.projected**2/wing.areas.reference
    wing.origin              = np.array([187.0,0,0])  * Units.feet
    wing.symmetric           = True
    wing.vertical            = False
    wing.dynamic_pressure_ratio = 0.95
    wing.ep_alpha            = 2.0*main_wing_CLa/np.pi/main_wing_ar    
    wing.aerodynamic_center  = [trapezoid_ac_x(wing), 0.0, 0.0]
    wing.CL_alpha            = datcom(wing,Mach)
    vehicle.append_component(wing)
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'Fuselage'
    fuselage.x_root_quarter_chord = 77.0 * Units.feet
    fuselage.lengths.total     = 229.7  * Units.feet
    fuselage.width      = 20.9   * Units.feet 
    vehicle.append_component(fuselage)
    
    configuration = Data()
    configuration.mass_properties = Data()
    configuration.mass_properties.center_of_gravity = Data()
    configuration.mass_properties.center_of_gravity = np.array([112.2,0,0]) * Units.feet  
    
    #Method Test
    print '<<Test run of the taw_cmalpha() method>>'
    print 'Boeing 747 at Mach {0}'.format(Mach[0])
    
    cm_a = taw_cmalpha(vehicle,Mach,conditions,configuration)
    
    expected = -1.45
    print 'Cm_alpha       = {0:.4f}'.format(cm_a[0])
    print 'Expected value = {}'.format(expected)
    print 'Percent Error  = {0:.2f}%'.format(100.0*(cm_a[0]-expected)/expected)
    print 'Static Margin  = {0:.4f}'.format(-cm_a[0]/vehicle.wings['Main Wing'].CL_alpha[0])
    print ' '