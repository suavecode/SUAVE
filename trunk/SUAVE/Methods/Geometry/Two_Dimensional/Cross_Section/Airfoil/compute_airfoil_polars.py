## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# compute_airfoil_polars.py
# 
# Created:  Mar 2019, M. Clarke
# Modified: Mar 2020, M. Clarke
#           Jan 2021, E. Botero
#           Jan 2021, R. Erhard
#           Nov 2021, R. Erhard
#           Aug 2022, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core               import Data , Units
from SUAVE.Methods.Aerodynamics.AERODAS.pre_stall_coefficients import pre_stall_coefficients
from SUAVE.Methods.Aerodynamics.AERODAS.post_stall_coefficients import post_stall_coefficients 
from .import_airfoil_geometry import import_airfoil_geometry 
from .import_airfoil_polars   import import_airfoil_polars 
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator


## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def compute_airfoil_polars(a_geo, a_polar, npoints = 200, use_pre_stall_data=True):
    """This computes the lift and drag coefficients of an airfoil in stall regimes using pre-stall
    characterstics and AERODAS formation for post stall characteristics. This is useful for 
    obtaining a more accurate prediction of wing and blade loading. Pre stall characteristics 
    are obtained in the form of a text file of airfoil polar data.
    
    Assumptions:
    Uses AERODAS formulation for post stall characteristics 

    Source:
    Models of Lift and Drag Coefficients of Stalled and Unstalled Airfoils in Wind Turbines and Wind Tunnels
    by D Spera, 2008

    Inputs:
    a_geo                  <string>
    a_polar                <string>
    use_pre_stall_data     [Boolean]
           

    Outputs:
    airfoil_data.
        cl_polars          [unitless]
        cd_polars          [unitless]      
        aoa_sweep          [unitless]
    
    Properties Used:
    N/A
    """  
    
    num_airfoils = len(a_geo)
    
    # check number of polars per airfoil in batch
    num_polars   = 0
    for i in range(num_airfoils): 
        n_p = len(a_polar[i])
        if n_p < 3:
            raise AttributeError('Provide three or more airfoil polars to compute surrogate')
        
        num_polars = max(num_polars, n_p)        

    # read airfoil geometry  
    airfoil_data = import_airfoil_geometry(a_geo, npoints = npoints)

    # Get all of the coefficients for AERODAS wings
    AoA_sweep_deg = np.linspace(-90, 90, 180 * 4 + 1)
    AoA_sweep_radians = AoA_sweep_deg*Units.degrees
    CL = np.zeros((num_airfoils, num_polars, len(AoA_sweep_deg)))
    CD = np.zeros((num_airfoils, num_polars, len(AoA_sweep_deg)))
    aoa0 = np.zeros((num_airfoils, num_polars))
    cl0 = np.zeros((num_airfoils, num_polars))
    
    # Create an infinite aspect ratio wing
    geometry              = SUAVE.Components.Wings.Wing()
    geometry.aspect_ratio = np.inf
    geometry.section      = Data()
    
    # Create dummy settings and state
    settings = Data()
    state    = Data()
    state.conditions = Data()
    state.conditions.aerodynamics = Data()
    state.conditions.aerodynamics.pre_stall_coefficients = Data()
    state.conditions.aerodynamics.post_stall_coefficients = Data()

    # read airfoil polars 
    airfoil_polar_data_raw = import_airfoil_polars(a_polar)
    airfoil_polar_data = smooth_raw_polar_data(airfoil_polar_data_raw)
    
    # initialize new data
    airfoil_data.angle_of_attacks             = []
    airfoil_data.lift_coefficient_surrogates  = []
    airfoil_data.drag_coefficient_surrogates  = []   
    airfoil_data.re_from_polar                = []
    airfoil_data.aoa_from_polar               = []    
    aNames = []
    
    # AERODAS 
    for i in range(num_airfoils):
        aNames.append(os.path.basename(a_geo[i])[:-4])
        # Modify the "wing" slightly:
        geometry.thickness_to_chord = airfoil_data.thickness_to_chord[i]
        
        for j in range(len(a_polar[i])):
            # Extract raw data from polars
            airfoil_cl         = np.array(airfoil_polar_data.lift_coefficients[i][j]) 
            airfoil_cd         = np.array(airfoil_polar_data.drag_coefficients[i][j]) 
            airfoil_aoa        = np.array(airfoil_polar_data.angle_of_attacks[i][j]) 
            
            airfoil_cl, airfoil_cd, airfoil_aoa = remove_post_stall(airfoil_cl, airfoil_cd, airfoil_aoa)
            
            # computing approximate zero lift aoa
            airfoil_cl_plus = airfoil_cl[airfoil_cl>0]
            idx_zero_lift = np.where(airfoil_cl == min(airfoil_cl_plus))[0][0]
            airfoil_cl_crossing = airfoil_cl[idx_zero_lift-1:idx_zero_lift+1]
            airfoil_aoa_crossing = airfoil_aoa[idx_zero_lift-1:idx_zero_lift+1]
            try:
                A0  = np.interp(0,airfoil_cl_crossing, airfoil_aoa_crossing)* Units.deg 
            except:
                A0 = airfoil_aoa[idx_zero_lift] * Units.deg
            
            # max lift coefficent and associated aoa
            CL1max = np.max(airfoil_cl)
            idx_aoa_max_prestall_cl = np.where(airfoil_cl == CL1max)[0][0]
            ACL1 = airfoil_aoa[idx_aoa_max_prestall_cl] * Units.degrees

            # computing approximate lift curve slope
            linear_idxs = [np.argmin(abs(airfoil_aoa)),np.argmin(abs(airfoil_aoa - 4))]
            cl_range = airfoil_cl[linear_idxs]
            aoa_range = airfoil_aoa[linear_idxs] * Units.degrees
            S1 = (cl_range[1]-cl_range[0])/(aoa_range[1]-aoa_range[0])

            # max drag coefficent and associated aoa
            CD1max  = np.max(airfoil_cd) 
            idx_aoa_max_prestall_cd = np.where(airfoil_cd == CD1max)[0][0]
            ACD1   = airfoil_aoa[idx_aoa_max_prestall_cd] * Units.degrees     
            
            # Find the point of lowest drag and the CD
            idx_CD_min = np.where(airfoil_cd==min(airfoil_cd))[0][0]
            ACDmin     = airfoil_aoa[idx_CD_min] * Units.degrees
            CDmin      = airfoil_cd[idx_CD_min]    
            
            # Setup data structures for this run
            ones = np.ones_like(AoA_sweep_radians)
            settings.section_zero_lift_angle_of_attack                = A0
            state.conditions.aerodynamics.angle_of_attack             = AoA_sweep_radians * ones 
            geometry.section.angle_attack_max_prestall_lift           = ACL1 * ones 
            geometry.pre_stall_maximum_drag_coefficient_angle         = ACD1 * ones 
            geometry.pre_stall_maximum_lift_coefficient               = CL1max * ones 
            geometry.pre_stall_maximum_lift_drag_coefficient          = CD1max * ones 
            geometry.section.minimum_drag_coefficient                 = CDmin * ones 
            geometry.section.minimum_drag_coefficient_angle_of_attack = ACDmin
            geometry.pre_stall_lift_curve_slope                       = S1
            
            # Get prestall coefficients
            CL1, CD1 = pre_stall_coefficients(state,settings,geometry)
            
            # Get poststall coefficents
            CL2, CD2 = post_stall_coefficients(state,settings,geometry)
            
            # Take the maxes
            CL_ij = np.fmax(CL1,CL2)
            CL_ij[AoA_sweep_radians<=A0] = np.fmin(CL1[AoA_sweep_radians<=A0],CL2[AoA_sweep_radians<=A0])
            
            CD_ij = np.fmax(CD1,CD2)
            
            # Pack this loop
            CL[i,j,:] = CL_ij
            CD[i,j,:] = CD_ij
            aoa0[i,j] = A0
            cl0[i,j]  = np.interp(0,airfoil_aoa,airfoil_cl)
            
            if use_pre_stall_data == True:
                CL[i,j,:], CD[i,j,:] = apply_pre_stall_data(AoA_sweep_deg, airfoil_aoa, airfoil_cl, airfoil_cd, CL[i,j,:], CD[i,j,:])
            

        # remove placeholder values (for airfoils that have different number of polars)
        n_p      = len(a_polar[i])
        RE_data  = airfoil_polar_data.reynolds_number[i][0:n_p]
        aoa_data = AoA_sweep_radians
        
        
        CL_sur = RegularGridInterpolator((RE_data, aoa_data), CL[i,0:n_p,:],bounds_error=False,fill_value=None)  
        CD_sur = RegularGridInterpolator((RE_data, aoa_data), CD[i,0:n_p,:],bounds_error=False,fill_value=None)           
        
        airfoil_data.angle_of_attacks.append(AoA_sweep_radians)
        airfoil_data.lift_coefficient_surrogates.append(CL_sur)
        airfoil_data.drag_coefficient_surrogates.append(CD_sur) 
        airfoil_data.re_from_polar.append(RE_data)
        airfoil_data.aoa_from_polar.append(aoa_data)
        
    airfoil_data.airfoil_names                 = aNames
    
    return airfoil_data

def remove_post_stall(airfoil_cl, airfoil_cd, airfoil_aoa):
    cl_grad = np.gradient(airfoil_cl)
    a0_idx = np.argmin(abs(airfoil_aoa))
    if np.any(cl_grad[:a0_idx] < 0):
        negativeInflectionPoint = a0_idx - np.where( np.flip(cl_grad[:a0_idx]) < 0 )[0][0] # 
    else:
        negativeInflectionPoint = 0
    if np.any(cl_grad[a0_idx:] < 0):
        positiveInflectionPoint = a0_idx + np.where( cl_grad[a0_idx:] < 0)[0][0]
    else:
        positiveInflectionPoint = len(airfoil_aoa)
    
    airfoil_cl = airfoil_cl[negativeInflectionPoint:positiveInflectionPoint]
    airfoil_cd = airfoil_cd[negativeInflectionPoint:positiveInflectionPoint]
    airfoil_aoa = airfoil_aoa[negativeInflectionPoint:positiveInflectionPoint]
    return airfoil_cl, airfoil_cd, airfoil_aoa

def apply_pre_stall_data(AoA_sweep_deg, airfoil_aoa, airfoil_cl, airfoil_cd, CL, CD):
    # Coefficients in pre-stall regime taken from experimental data:
    aoa_locs = (AoA_sweep_deg>=airfoil_aoa[0]) * (AoA_sweep_deg<=airfoil_aoa[-1])
    aoa_in_data = AoA_sweep_deg[aoa_locs]
    
    # if the data is within experimental use it, if not keep the surrogate values
    CL[aoa_locs] = airfoil_cl[abs(aoa_in_data[:,None] - airfoil_aoa[None,:]).argmin(axis=-1)]
    CD[aoa_locs] = airfoil_cd[abs(aoa_in_data[:,None] - airfoil_aoa[None,:]).argmin(axis=-1)]
    
    # remove kinks/overlap between pre- and post-stall  
    data_lb = [i for i, v in enumerate(abs(CD-airfoil_cd[0])) if v == min(abs(CD-airfoil_cd[0]))][0]#np.where(CD == airfoil_cd[0])[0][0]
    data_ub = [i for i, v in enumerate(abs(CD-airfoil_cd[-1])) if v == min(abs(CD-airfoil_cd[-1]))][-1]#np.where(CD == airfoil_cd[-1])[0][-1]
    CD[0:data_lb] = np.maximum(CD[0:data_lb], CD[data_lb]*np.ones_like(CD[0:data_lb]))
    CD[data_ub:]  = np.maximum(CD[data_ub:],  CD[data_ub]*np.ones_like(CD[data_ub:])) 
    
    return CL, CD


def smooth_raw_polar_data(polar_data_raw):
    # initialize new data structure
    polar_data_new = Data()
    polar_data_new.reynolds_number = polar_data_raw.reynolds_number
    polar_data_new.mach_number = polar_data_raw.mach_number
    polar_data_new.lift_coefficients = []
    polar_data_new.drag_coefficients = []
    polar_data_new.angle_of_attacks = []
    
    # get information
    num_airfoils = len(polar_data_raw.lift_coefficients)
    for a in range(num_airfoils): 
        num_polars = len(polar_data_raw.lift_coefficients[a])
        cl_a = []
        cd_a = []
        aoa_a = []
        for i in range(num_polars):
            # extract raw data
            c_l_raw = np.array(polar_data_raw.lift_coefficients[a][i])
            c_d_raw = np.array(polar_data_raw.drag_coefficients[a][i])
            aoa_raw = np.array(polar_data_raw.angle_of_attacks[a][i])
            
            # smooth the data with rolling average
            c_l_filtered = rollingAverage(aoa_raw, c_l_raw, window_size=9, order=1)
            c_d_filtered = rollingAverage(aoa_raw, c_d_raw, window_size=9, order=1)          
            
            ## remove post-stall regions
            #c_l_grad = np.gradient(c_l_filtered)
            #c_l_grad_target = c_l_grad[np.argmin(abs(aoa_raw))]
            #pre_stalled_ids = np.where(c_l_grad > 0.4 * c_l_grad_target)
            
            cl_a.append(list(c_l_filtered)) #[pre_stalled_ids]))
            cd_a.append(list(c_d_filtered)) #[pre_stalled_ids]))
            aoa_a.append(list(aoa_raw)) #[pre_stalled_ids]))
        polar_data_new.lift_coefficients.append(cl_a)
        polar_data_new.drag_coefficients.append(cd_a)
        polar_data_new.angle_of_attacks.append(aoa_a)
    
    return polar_data_new

def rollingAverage(x, y, window_size=13, order=1):
    N = len(x)
    y_averaged = np.zeros_like(y)
    for i in range(N):
        # fit a polynomial of specified order to this window of data

        # At the start and end use half window size
        if i < window_size // 2:
            w_start = 0
            w_end = window_size // 2
        elif i > N - window_size // 2:
            w_start = N - window_size // 2
            w_end = N
        else:
            w_start = i - window_size // 2
            w_end = i + window_size // 2

        fitData = np.polyfit(x[w_start:w_end], y[w_start:w_end], deg=order)
        y_averaged[i] = np.polyval(fitData, x[i])

    return y_averaged