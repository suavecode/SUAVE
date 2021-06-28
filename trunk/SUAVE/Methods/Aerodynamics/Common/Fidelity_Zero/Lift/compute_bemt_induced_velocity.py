## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# compute_bemt_induced_velocity.py
# 
# Created:  Jun 2021, R. Erhard 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np 

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift 
def compute_bemt_induced_velocity(prop,geometry,conditions, plot_induced_v=False):  
    """ This computes the velocity induced by the fixed helical wake
    on lifting surface control points

    Assumptions:  
    
    Source:   
    
    Inputs: 
    WD     - helical wake distribution points               [Unitless] 
    VD     - vortex distribution points on lifting surfaces [Unitless] 
    cpts   - control points in segment                      [Unitless] 

    Properties Used:
    N/A
    """    
    # extract the propeller data structure
    VD   = geometry.vortex_distribution
    
    # contraction factor
    R = prop.tip_radius
    s = geometry.wings.main_wing.origin[0][0] - prop.origin[0][0]
    kd = 1 + s/(np.sqrt(s**2 + R**2)) # formula from McCormick 1969, Aerodynamics of V/STOL Flight

    # extract radial and azimuthal velocities at blade
    va = kd*prop.outputs.blade_axial_induced_velocity[0]
    vt = kd*prop.outputs.blade_tangential_induced_velocity[0]
    r = prop.outputs.disc_radial_distribution[0][0]
    
    Vinf = conditions.freestream.velocity[0][0]
    
    if plot_induced_v:
        import pylab as plt
        ls = ['-','--','-.']
        lc = ['black','mediumblue','firebrick']
        plt.rcParams['axes.linewidth'] = 2.
        plt.rcParams["font.family"]    = "Times New Roman"
        plt.rcParams.update({'font.size': 14})                    
        plt.figure()
        plt.plot(r/R,va/Vinf,ls[0],color=lc[0],label="$\\frac{V_a}{V_\infty}$")
        plt.plot(r/R,vt/Vinf,ls[0],color=lc[2],label="$\\frac{V_t}{V_\infty}$")
        plt.xlabel("$\\frac{r}{R}$")
        plt.ylabel("Normalized Velocity")
        plt.title("Contracted Induced Velocities at Wing")
        plt.legend()
        plt.show()
    
    hub_y_center = prop.origin[0][1]
    prop_y_min = hub_y_center - r[-1]
    prop_y_max = hub_y_center + r[-1]
    
    
    prop_V_wake_ind = np.zeros((1,VD.n_cp,3)) # one mission control point, n_cp wing control points, 3 velocity components
    from scipy.interpolate import interp1d
    
    ir = prop_y_min+r
    ro = np.flipud(prop_y_max-r)
    prop_y_range = np.append(ir, ro)
    

    # within this range, add an induced x- and z- velocity from propeller wake
    bool_in_range = (abs(VD.YC)>ir[0])*(abs(VD.YC)<ro[-1])
    YC_in_range = VD.YC[bool_in_range]
    
    va_y_range   = np.append(np.flipud(va), va)
    vt_y_range   = np.append(np.flipud(vt), vt)*prop.rotation[0]
    va_interp = interp1d(prop_y_range, va_y_range)
    vt_interp = interp1d(prop_y_range, vt_y_range)
    
    for i in range(len(YC_in_range)):
        y = YC_in_range[i]
        cp_idx = np.where(VD.YC==YC_in_range[i])
        # determine if point is inboard or outboard of the propeller hub
        if abs(y)<hub_y_center:
            inboard_pt = True
        else:
            inboard_pt = False
        # va acts in x-direction
        # vt acts in y-direction (sign effected by rotation)                        
        if y<0: # Left wing
            # symmetry, flip
            va_new = va_interp(-y)
            if inboard_pt:
                vt_new = -vt_interp(-y)
            else:
                vt_new = vt_interp(-y)                  
        else: # Right wing
            va_new = va_interp(y)
            if inboard_pt:
                vt_new = -vt_interp(y)
            else:
                vt_new = vt_interp(y)
            
        prop_V_wake_ind[0,cp_idx,0] = va_new  # axial induced velocity
        prop_V_wake_ind[0,cp_idx,1] = 0       # spanwise induced velocity; in line with prop, so 0
        prop_V_wake_ind[0,cp_idx,2] = vt_new  # vertical induced velocity                           
                    
    
    return prop_V_wake_ind
  
  