## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# compute_bemt_induced_velocity.py
# 
# Created:  Jun 2021, R. Erhard 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np 
from scipy.interpolate import interp1d
from SUAVE.Components import Wings

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift 
def compute_bemt_induced_velocity(props,geometry,cpts,conditions,identical_flag,wing_instance=None):  
    """ This computes the velocity induced by the BEMT wake
    on lifting surface control points

    Assumptions:  
       The wake contracts following formulation by McCormick.
    
    Source:   
       Contraction factor: McCormick 1969, Aerodynamics of V/STOL Flight
       
    Inputs: 
       prop        - propeller or rotor data structure             [Unitless] 
       geometry    - SUAVE vehicle                                 [Unitless] 
       cpts        - control points in segment                     [Unitless]
    Properties Used:
       N/A
    """

    # extract vortex distribution
    VD = geometry.vortex_distribution
    
    # initialize propeller wake induced velocities
    prop_V_wake_ind = np.zeros((cpts,VD.n_cp,3))
    
    for i,prop in enumerate(props):
        if identical_flag:
            idx = 0
        else:
            idx = i
        prop_key     = list(props.keys())[idx]
        prop_outputs = conditions.noise.sources.propellers[prop_key]
        R            = prop.tip_radius
        
        # contraction factor by McCormick
        if wing_instance == None:
            nmw = 0
            for wing in geometry.wings:
                if not isinstance(wing,Wings.Main_Wing): continue
                nmw = nmw+1
                s  = wing.origin[0][0] - prop.origin[0][0]
                kd = 1 + s/(np.sqrt(s**2 + R**2))
            if nmw ==1:
                print("No wing specified for wake analysis in compute_bemt_induced_velocity. Main wing is used.")
            elif nmw>1:
                print("No wing specified for wake analysis in compute_bemt_induced_velocity. Multiple main wings, using the last one.")
            else:
                print("No wing specified for wake analysis in compute_bemt_induced_velocity. No main_wing defined! Using last wing found.")
                s  = wing.origin[0][0] - prop.origin[0][0]
                kd = 1 + s/(np.sqrt(s**2 + R**2))            
                    
        else:
            s  = wing_instance.origin[0][0] - prop.origin[0][0]
            kd = 1 + s/(np.sqrt(s**2 + R**2))        

        # extract radial and azimuthal velocities at blade
        va = kd*prop_outputs.blade_axial_induced_velocity[0]
        vt = kd*prop_outputs.blade_tangential_induced_velocity[0]
        r  = prop_outputs.disc_radial_distribution[0,:,0]
        
        hub_y_center = prop.origin[0][1]
        prop_y_min   = hub_y_center - r[-1]
        prop_y_max   = hub_y_center + r[-1]
        
        ir = prop_y_min+r
        ro = np.flipud(prop_y_max-r)
        prop_y_range = np.append(ir, ro)
    
        # within this range, add an induced x- and z- velocity from propeller wake
        bool_in_range = (abs(VD.YC)>ir[0])*(abs(VD.YC)<ro[-1])
        YC_in_range   = VD.YC[bool_in_range]
        
        va_y_range  = np.append(np.flipud(va), va)
        vt_y_range  = np.append(np.flipud(vt), vt)*prop.rotation
        va_interp   = interp1d(prop_y_range, va_y_range)
        vt_interp   = interp1d(prop_y_range, vt_y_range)
        
        y_vals  = YC_in_range
        val_ids = np.where(bool_in_range==True)
    
        # check if y values are inboard of propeller axis
        inboard_bools = abs(y_vals)<hub_y_center
        
        # preallocate va_new and vt_new
        va_new = np.zeros(np.size(val_ids))
        vt_new = np.zeros(np.size(val_ids))
    
        # take absolute y values (symmetry)
        va_new = va_interp(abs(y_vals))
        
        # inboard vt values
        vt_new[inboard_bools]        = -vt_interp(abs(y_vals[inboard_bools]))
        vt_new[inboard_bools==False] = vt_interp(abs(y_vals[inboard_bools==False]))
        
        prop_V_wake_ind[0,val_ids,0] = va_new  # axial induced velocity
        prop_V_wake_ind[0,val_ids,1] = 0       # spanwise induced velocity; in line with prop, so 0
        prop_V_wake_ind[0,val_ids,2] = vt_new  # vertical induced velocity      
    
       
    return prop_V_wake_ind
  
  