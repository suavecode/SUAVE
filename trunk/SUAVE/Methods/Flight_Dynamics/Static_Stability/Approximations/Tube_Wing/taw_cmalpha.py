## @ingroup Methods-Flight_Dynamics-Static_Stability-Approximations-Tube_wing
# taw_cmalpha.py
#
# Created:  Apr 2014, T. Momose
# Modified: Nov 2015, M. Vegh
#           Jan 2016, E. Botero
#           Jul 2017, M. Clarke
#           Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np
from SUAVE.Methods.Center_of_Gravity.compute_mission_center_of_gravity import compute_mission_center_of_gravity

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

## @ingroup Methods-Flight_Dynamics-Static_Stability-Approximations-Tube_wing
def taw_cmalpha(geometry,mach,conditions,configuration):
    """ This method computes the static longitudinal stability derivative for a
    standard Tube-and-Wing aircraft configuration.
            
    Assumptions:
        -This method assumes a tube-and-wing configuration
        -April 8, 2014 - The current version only accounts for the effect of
        downwash on the lift curve slopes of lifting surfaces behind other
        lifting surfaces by an efficiency factor.
        
    Source: 
        Unknown
    
    Inputs:
        configuration - a data dictionary with the fields:
            reference - a data dictionary with the fields: 
                area - the reference wing area                                  [meters**2]
                mac  - the main wing mean aerodynamic chord                     [meters]
                CL_alpha_wing - the main wing lift curve slope                  [dimensionless]
            Mass_Props - a data dictionary with the field:
                pos_cg - A vector in 3-space indicating CG position             [meters]
            Lifting_Surfaces - a list containing data dictionaries
            representing all non-vertical lifting surfaces. Each of these
            lifting surface data dictionaries has the fields:
                x_LE - the x-coordinate of the surface root LE position 
                                                                                [meters] (root at fuselage symmetry plane)
                x_ac_LE - the x-coordinate of the surface's aerodynamic 
                center measured relative to the root leading edge (root
                of reference area - at symmetry plane)                          [meters]
                area - lifting surface planform area                            [meters**2]
                span - span of the lifting surface                              [meters]
                sweep_le - sweep of the leading edge                            [radians]
                taper - taper ratio                                             [dimensionless]
                aspect_ratio - aspect ratio                                     [dimensionless]
                CL_alpha - the surface's lift curve slope                       [dimensionless]
                eta - lifting efficiency. Use to indicate effect of surfaces
                ahead of the lifting surface                                    [dimensionless]
            fuselage - a data dictionary with the fields:
                x_root_quarter_chord - x coordinate of the quarter-chord of 
                the wing root (here, the wing root is where the wing 
                intersects the body)                                            [meters]
                w_max - maximum width of the fuselage                           [meters]
                length - length of the fuselage                                 [meters]
        mach - flight Mach number
    
    Outputs:
        cm_alpha - a single float value: The static longidutinal stability
        derivative (d(Cm_cg)/d(alpha))
        
    Properties Used:
        N/A              
    """

    # Unpack inputs
    Sref   = geometry.reference_area
    mac    = geometry.wings['main_wing'].chords.mean_aerodynamic
    c_root = geometry.wings['main_wing'].chords.root
    taper  = geometry.wings['main_wing'].taper
    c_tip  = taper*c_root
    span   = geometry.wings['main_wing'].spans.projected
    sweep  = geometry.wings['main_wing'].sweeps.quarter_chord
    C_Law  = conditions.lift_curve_slope
    alpha  = conditions.aerodynamics.angle_of_attack 
    M      = mach
    
    weights      = conditions.weights.total_mass
    fuel_weights = weights-configuration.mass_properties.max_zero_fuel
    cg           = compute_mission_center_of_gravity(configuration,fuel_weights)		
    x_cg         = np.atleast_2d(cg[:,0]).T # get cg location at every point in the mission    
    

    #Evaluate the effect of the fuselage on the stability derivative
    if 'fuselage' in geometry.fuselages:
        w_f   = geometry.fuselages['fuselage'].width
        l_f   = geometry.fuselages['fuselage'].lengths.total
        x_rqc = geometry.wings['main_wing'].origin[0][0] + 0.5*w_f*np.tan(sweep) + 0.25*c_root*(1 - (w_f/span)*(1-taper))    
        
            
        p  = x_rqc/l_f
        Kf = 1.5012*p**2. + 0.538*p + 0.0331
        CmAlpha_body = Kf*w_f*w_f*l_f/Sref/mac   #NEGLECTS TAIL EFFECT ON CL_ALPHA
    else:
        CmAlpha_body = 0.

    #Evaluate the effect of each lifting surface in turn
    CmAlpha_surf = []
    Cm0_surf     = []
    for surf in geometry.wings:
        #Unpack inputs
        s         = surf.areas.reference
        x_surf    = surf.origin[0][0]
        x_ac_surf = surf.aerodynamic_center[0]
        eta       = surf.dynamic_pressure_ratio
        downw     = 1 - surf.ep_alpha
        CL_alpha  = surf.CL_alpha
        vertical  = surf.vertical
        twist_r   = surf.twists.root
        twist_t   = surf.twists.tip     
        taper     = surf.taper
        if 'Airfoil' in surf:
            if 'zero_angle_lift_coefficient' in surf.Airfoil:
                al0 = surf.Airfoil.zero_angle_lift_coefficient
            else:
                al0 = 0.
            if 'zero_angle_moment_coefficient' in surf.Airfoil:
                cmac = surf.Airfoil.zero_angle_moment_coefficient
            else:
                cmac = 0.
        else:
            al0  = 0.
            cmac = 0.
        
        # Average out the incidence angles to ge the zero angle lift
        CL0_surf   = CL_alpha * ((twist_r+taper*twist_t)/2. -al0)
        
        #Calculate Cm_alpha contributions
        l_surf    = x_surf + x_ac_surf - x_cg
        Cma       = -l_surf*s/(mac*Sref)*(CL_alpha*eta*downw)*(1. - vertical)
        cmo       = cmac+ s*eta*CL0_surf*l_surf*downw*(1. - vertical)/(mac*Sref)
        CmAlpha_surf.append(Cma)
        Cm0_surf.append(cmo)
        
    
    cm_alpha = sum(CmAlpha_surf) + CmAlpha_body
    
    CM0 = sum(Cm0_surf)
    
    CM = cm_alpha*alpha + CM0
    
    return cm_alpha, CM0, CM

# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':
    from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.datcom import datcom
    from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.trapezoid_ac_x import trapezoid_ac_x
    from SUAVE.Core import Units
    from SUAVE.Core import (
        Data, Container,
    )    
    
    #Parameters Required
    #Using values for a Boeing 747-200  
    vehicle = SUAVE.Vehicle()
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'main_wing'
    wing.areas.reference           = 5500.0 * Units.feet**2
    wing.spans.projected           = 196.0  * Units.feet
    wing.chords.mean_aerodynamic   = 27.3 * Units.feet
    wing.chords.root               = 44. * Units.feet  #54.5ft
    wing.sweeps.leading_edge       = 42.0   * Units.deg # Leading edge
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
    wing.tag = 'horizontal_stabilizer'
    wing.areas.reference     = 1490.55* Units.feet**2
    wing.spans.projected     = 71.6   * Units.feet
    wing.sweeps.leading_edge = 44.0   * Units.deg # leading edge
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
    fuselage.tag = 'fuselage'
    fuselage.x_root_quarter_chord = 77.0 * Units.feet
    fuselage.lengths.total     = 229.7  * Units.feet
    fuselage.width      = 20.9   * Units.feet 
    vehicle.append_component(fuselage)
    
    configuration = Data()
    configuration.mass_properties = Data()
    configuration.mass_properties.center_of_gravity = Data()
    configuration.mass_properties.center_of_gravity = np.array([112.2,0,0]) * Units.feet  
    
    #Method Test
    print('<<Test run of the taw_cmalpha() method>>')
    print('Boeing 747 at Mach {0}'.format(Mach[0]))
    
    cm_a = taw_cmalpha(vehicle,Mach,conditions,configuration)
    
    expected = -1.45
    print('Cm_alpha       = {0:.4f}'.format(cm_a[0]))
    print('Expected value = {}'.format(expected))
    print('Percent Error  = {0:.2f}%'.format(100.0*(cm_a[0]-expected)/expected))
    print('Static Margin  = {0:.4f}'.format(-cm_a[0]/vehicle.wings['main_wing'].CL_alpha[0]))
    print(' ')
