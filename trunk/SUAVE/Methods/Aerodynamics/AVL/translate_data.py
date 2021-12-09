## @ingroup Methods-Aerodynamics-AVL
#translate_data.py
# 
# Created:  Mar 2015, T. Momose
# Modified: Jan 2016, E. Botero
#           Apr 2017, M. Clarke
#           Dec 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

import SUAVE
from SUAVE.Core import Data, Units
from .Data.Cases import Run_Case

## @ingroup Methods-Aerodynamics-AVL
def translate_conditions_to_cases(avl ,conditions):
    """ Takes SUAVE Conditions() data structure and translates to a Container of
    avl Run_Case()s.

    Assumptions:
        None
        
    Source:
        Drela, M. and Youngren, H., AVL, http://web.mit.edu/drela/Public/web/avl

    Inputs:
        conditions.aerodynamics.angle_of_attack  [radians] 
        conditions.freestream.mach_number        [-]
        conditions.freestream.density            [kilograms per meters**3]
        conditions.freestream.gravity            [meters per second**2]

    Outputs:
        cases                                    [data structur]

    Properties Used:
        N/A
    """    
    # set up aerodynamic Conditions object
    aircraft = avl.geometry
    cases    = Run_Case.Container()
    for i in range(len(conditions.aerodynamics.angle_of_attack)):      
        case                                                  = Run_Case()
        case.tag                                              = avl.settings.filenames.case_template.format(avl.current_status.batch_index,i+1)
        case.mass                                             = conditions.weights.total_mass
        case.conditions.freestream.mach                       = conditions.freestream.mach_number
        case.conditions.freestream.density                    = conditions.freestream.density
        case.conditions.freestream.gravitational_acceleration = conditions.freestream.gravity      
        case.conditions.aerodynamics.angle_of_attack          = conditions.aerodynamics.angle_of_attack[i]/Units.deg
        case.conditions.aerodynamics.side_slip_angle          = conditions.aerodynamics.side_slip_angle  
        case.conditions.aerodynamics.lift_coefficient         = conditions.aerodynamics.lift_coefficient
        case.conditions.aerodynamics.roll_rate_coefficient    = conditions.aerodynamics.roll_rate_coefficient
        case.conditions.aerodynamics.pitch_rate_coefficient   = conditions.aerodynamics.pitch_rate_coefficient
        
        # determine the number of wings 
        n_wings = 0 
        for wing in aircraft.wings:
            n_wings += 1
            if wing.symmetric == True:
                n_wings += 1                
        case.num_wings                                        = n_wings
        case.n_sw                                             = avl.settings.number_spanwise_vortices  
                
        cases.append_case(case)
    
    return cases

def translate_results_to_conditions(cases,results):
    """ Takes avl results structure containing the results of each run case stored
        each in its own Data() object. Translates into the Conditions() data structure.

    Assumptions:
        None
        
    Source:
        Drela, M. and Youngren, H., AVL, http://web.mit.edu/drela/Public/web/avl

    Inputs:
        case_res = results 
            
     Outputs:
        cases                        [data_structure]
   
    Properties Used:
        N/A
    """   
   
    num_wings = cases[0].num_wings 
    n_sw      = cases[0].n_sw
    dim       = len(cases)
        
    # set up aerodynamic Conditions object
    res                             = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics() 
    res.stability.static            = Data()
    res.stability.dynamic           = Data()
    
    # add missing entries 
    res.S_ref                                                = np.zeros((dim,1))
    res.c_ref                                                = np.zeros_like(res.S_ref)
    res.b_ref                                                = np.zeros_like(res.S_ref)
    res.X_ref                                                = np.zeros_like(res.S_ref)
    res.Y_ref                                                = np.zeros_like(res.S_ref)
    res.Z_ref                                                = np.zeros_like(res.S_ref)   
    res.aerodynamics.AoA                                     = np.zeros_like(res.S_ref)
    res.aerodynamics.CX                                      = np.zeros_like(res.S_ref)
    res.aerodynamics.CY                                      = np.zeros_like(res.S_ref)
    res.aerodynamics.CZ                                      = np.zeros_like(res.S_ref)                                                   
    res.aerodynamics.Cltot                                   = np.zeros_like(res.S_ref)
    res.aerodynamics.Cmtot                                   = np.zeros_like(res.S_ref) 
    res.aerodynamics.Cntot                                   = np.zeros_like(res.S_ref)    
    res.aerodynamics.roll_moment_coefficient                 = np.zeros_like(res.S_ref)
    res.aerodynamics.pitch_moment_coefficient                = np.zeros_like(res.S_ref)
    res.aerodynamics.yaw_moment_coefficient                  = np.zeros_like(res.S_ref)  
    res.aerodynamics.lift_coefficient                        = np.zeros_like(res.S_ref)
    res.aerodynamics.drag_breakdown.induced                  = SUAVE.Analyses.Mission.Segments.Conditions.Conditions()
    res.aerodynamics.drag_breakdown.induced.total            = np.zeros_like(res.S_ref)
    res.aerodynamics.drag_breakdown.induced.efficiency_factor= np.zeros_like(res.S_ref)
    res.aerodynamics.oswald_efficiency                       = np.zeros_like(res.S_ref)
    
    # stability axis
    res.stability.static.CL_alpha                            = np.zeros_like(res.S_ref)
    res.stability.static.CY_alpha                            = np.zeros_like(res.S_ref)
    res.stability.static.Cl_alpha                            = np.zeros_like(res.S_ref)
    res.stability.static.Cm_alpha                            = np.zeros_like(res.S_ref)
    res.stability.static.Cn_alpha                            = np.zeros_like(res.S_ref)
    res.stability.static.CL_beta                             = np.zeros_like(res.S_ref)
    res.stability.static.CY_beta                             = np.zeros_like(res.S_ref)
    res.stability.static.Cl_beta                             = np.zeros_like(res.S_ref)
    res.stability.static.Cm_beta                             = np.zeros_like(res.S_ref)
    res.stability.static.Cn_beta                             = np.zeros_like(res.S_ref)    
    res.stability.static.CL_p                                = np.zeros_like(res.S_ref)
    res.stability.static.CL_q                                = np.zeros_like(res.S_ref)
    res.stability.static.CL_r                                = np.zeros_like(res.S_ref)
    res.stability.static.CY_p                                = np.zeros_like(res.S_ref)
    res.stability.static.CY_q                                = np.zeros_like(res.S_ref)
    res.stability.static.CY_r                                = np.zeros_like(res.S_ref)
    res.stability.static.Cl_p                                = np.zeros_like(res.S_ref)
    res.stability.static.Cl_q                                = np.zeros_like(res.S_ref)
    res.stability.static.Cl_r                                = np.zeros_like(res.S_ref)
    res.stability.static.Cm_p                                = np.zeros_like(res.S_ref)
    res.stability.static.Cm_q                                = np.zeros_like(res.S_ref) 
    res.stability.static.Cm_r                                = np.zeros_like(res.S_ref)
    res.stability.static.Cn_p                                = np.zeros_like(res.S_ref)
    res.stability.static.Cn_q                                = np.zeros_like(res.S_ref)
    res.stability.static.Cn_r                                = np.zeros_like(res.S_ref)  
    
    # body axis derivatives 
    res.stability.static.CX_u                                = np.zeros_like(res.S_ref)
    res.stability.static.CX_v                                = np.zeros_like(res.S_ref)
    res.stability.static.CX_w                                = np.zeros_like(res.S_ref)
    res.stability.static.CY_u                                = np.zeros_like(res.S_ref)
    res.stability.static.CY_v                                = np.zeros_like(res.S_ref)
    res.stability.static.CY_w                                = np.zeros_like(res.S_ref)
    res.stability.static.CZ_u                                = np.zeros_like(res.S_ref)
    res.stability.static.CZ_v                                = np.zeros_like(res.S_ref)
    res.stability.static.CZ_w                                = np.zeros_like(res.S_ref)  
    res.stability.static.Cl_u                                = np.zeros_like(res.S_ref)
    res.stability.static.Cl_v                                = np.zeros_like(res.S_ref)
    res.stability.static.Cl_w                                = np.zeros_like(res.S_ref)
    res.stability.static.Cm_u                                = np.zeros_like(res.S_ref)
    res.stability.static.Cm_v                                = np.zeros_like(res.S_ref)
    res.stability.static.Cm_w                                = np.zeros_like(res.S_ref)
    res.stability.static.Cn_u                                = np.zeros_like(res.S_ref)
    res.stability.static.Cn_v                                = np.zeros_like(res.S_ref)
    res.stability.static.Cn_w                                = np.zeros_like(res.S_ref)      
    res.stability.static.CX_p                                = np.zeros_like(res.S_ref)
    res.stability.static.CX_q                                = np.zeros_like(res.S_ref)
    res.stability.static.CX_r                                = np.zeros_like(res.S_ref)
    res.stability.static.CY_p                                = np.zeros_like(res.S_ref)
    res.stability.static.CY_q                                = np.zeros_like(res.S_ref)
    res.stability.static.CY_r                                = np.zeros_like(res.S_ref)
    res.stability.static.CZ_p                                = np.zeros_like(res.S_ref)
    res.stability.static.CZ_q                                = np.zeros_like(res.S_ref)
    res.stability.static.CZ_r                                = np.zeros_like(res.S_ref)  
    res.stability.static.Cl_p                                = np.zeros_like(res.S_ref)
    res.stability.static.Cl_q                                = np.zeros_like(res.S_ref)
    res.stability.static.Cl_r                                = np.zeros_like(res.S_ref)
    res.stability.static.Cm_p                                = np.zeros_like(res.S_ref)
    res.stability.static.Cm_q                                = np.zeros_like(res.S_ref)
    res.stability.static.Cm_r                                = np.zeros_like(res.S_ref)
    res.stability.static.Cn_p                                = np.zeros_like(res.S_ref)
    res.stability.static.Cn_q                                = np.zeros_like(res.S_ref)
    res.stability.static.Cn_r                                = np.zeros_like(res.S_ref)      
 
    res.stability.static.neutral_point                       = np.zeros_like(res.S_ref)    
    res.stability.static.spiral_criteria                     = np.zeros_like(res.S_ref)    
 
    # aero results 1: total surface forces and coefficeints 
    res.aerodynamics.wing_areas                    = np.zeros((dim,num_wings)) 
    res.aerodynamics.wing_CLs                      = np.zeros_like(res.aerodynamics.wing_areas) 
    res.aerodynamics.wing_CDs                      = np.zeros_like(res.aerodynamics.wing_areas) 

    # aero results 2 : sectional forces and coefficients 
    res.aerodynamics.wing_local_spans              = np.zeros((dim,num_wings,n_sw))
    res.aerodynamics.wing_section_chords           = np.zeros_like(res.aerodynamics.wing_local_spans)
    res.aerodynamics.wing_section_cls              = np.zeros_like(res.aerodynamics.wing_local_spans)
    res.aerodynamics.wing_section_induced_angle    = np.zeros_like(res.aerodynamics.wing_local_spans)
    res.aerodynamics.wing_section_cds              = np.zeros_like(res.aerodynamics.wing_local_spans)
    
    res.stability.static.control_surfaces_cases   = {}
    
    mach_case = list(results.keys())[0][5:9]   
    for i in range(len(results.keys())):
        aoa_case = '{:04d}'.format(i+1)
        tag = 'case_' + mach_case + '_' + aoa_case
        case_res = results[tag]       
        
        # stability file 
        res.S_ref[i][0]                                     = case_res.S_ref 
        res.c_ref[i][0]                                     = case_res.c_ref 
        res.b_ref[i][0]                                     = case_res.b_ref
        res.X_ref[i][0]                                     = case_res.X_ref 
        res.Y_ref[i][0]                                     = case_res.Y_ref 
        res.Z_ref[i][0]                                     = case_res.Z_ref       
        res.aerodynamics.AoA[i][0]                          = case_res.aerodynamics.AoA
        res.aerodynamics.CX[i][0]                           = case_res.aerodynamics.CX 
        res.aerodynamics.CY[i][0]                           = case_res.aerodynamics.CY  
        res.aerodynamics.CZ[i][0]                           = case_res.aerodynamics.CZ    
        res.aerodynamics.Cltot[i][0]                        = case_res.aerodynamics.Cltot 
        res.aerodynamics.Cmtot[i][0]                        = case_res.aerodynamics.Cmtot 
        res.aerodynamics.Cntot[i][0]                        = case_res.aerodynamics.Cntot      
        res.aerodynamics.roll_moment_coefficient[i][0]      = case_res.aerodynamics.roll_moment_coefficient
        res.aerodynamics.pitch_moment_coefficient[i][0]     = case_res.aerodynamics.pitch_moment_coefficient
        res.aerodynamics.yaw_moment_coefficient[i][0]       = case_res.aerodynamics.yaw_moment_coefficient
        res.aerodynamics.lift_coefficient[i][0]             = case_res.aerodynamics.total_lift_coefficient
        res.aerodynamics.drag_breakdown.induced.total[i][0] = case_res.aerodynamics.induced_drag_coefficient
        res.aerodynamics.drag_breakdown.induced.efficiency_factor[i][0]  = case_res.aerodynamics.oswald_efficiency 
        res.aerodynamics.oswald_efficiency[i][0]            = case_res.aerodynamics.oswald_efficiency 
        res.stability.static.CL_alpha[i][0]                 = case_res.stability.alpha_derivatives.lift_curve_slope
        res.stability.static.CY_alpha[i][0]                 = case_res.stability.alpha_derivatives.side_force_derivative
        res.stability.static.Cl_alpha[i][0]                 = case_res.stability.alpha_derivatives.roll_moment_derivative
        res.stability.static.Cm_alpha[i][0]                 = case_res.stability.alpha_derivatives.pitch_moment_derivative
        res.stability.static.Cn_alpha[i][0]                 = case_res.stability.alpha_derivatives.yaw_moment_derivative
        res.stability.static.CL_beta[i][0]                  = case_res.stability.beta_derivatives.lift_coefficient_derivative
        res.stability.static.CY_beta[i][0]                  = case_res.stability.beta_derivatives.side_force_derivative
        res.stability.static.Cl_beta[i][0]                  = case_res.stability.beta_derivatives.roll_moment_derivative
        res.stability.static.Cm_beta[i][0]                  = case_res.stability.beta_derivatives.pitch_moment_derivative
        res.stability.static.Cn_beta[i][0]                  = case_res.stability.beta_derivatives.yaw_moment_derivative        
        res.stability.static.CL_p[i][0]                     = case_res.stability.CL_p 
        res.stability.static.CL_q[i][0]                     = case_res.stability.CL_q
        res.stability.static.CL_r[i][0]                     = case_res.stability.CL_r 
        res.stability.static.CY_p[i][0]                     = case_res.stability.CY_p 
        res.stability.static.CY_q[i][0]                     = case_res.stability.CY_q 
        res.stability.static.CY_r[i][0]                     = case_res.stability.CY_r
        res.stability.static.Cl_p[i][0]                     = case_res.stability.Cl_p 
        res.stability.static.Cl_q[i][0]                     = case_res.stability.Cl_q 
        res.stability.static.Cl_r[i][0]                     = case_res.stability.Cl_r 
        res.stability.static.Cm_p[i][0]                     = case_res.stability.Cm_p 
        res.stability.static.Cm_q[i][0]                     = case_res.stability.Cm_q 
        res.stability.static.Cm_r[i][0]                     = case_res.stability.Cm_r 
        res.stability.static.Cn_p[i][0]                     = case_res.stability.Cn_p 
        res.stability.static.Cn_q[i][0]                     = case_res.stability.Cn_q 
        res.stability.static.Cn_r[i][0]                     = case_res.stability.Cn_r
        res.stability.static.CX_u[i][0]                     = case_res.stability.CX_u
        res.stability.static.CX_v[i][0]                     = case_res.stability.CX_v
        res.stability.static.CX_w[i][0]                     = case_res.stability.CX_w
        res.stability.static.CY_u[i][0]                     = case_res.stability.CY_u
        res.stability.static.CY_v[i][0]                     = case_res.stability.CY_v
        res.stability.static.CY_w[i][0]                     = case_res.stability.CY_w
        res.stability.static.CZ_u[i][0]                     = case_res.stability.CZ_u
        res.stability.static.CZ_v[i][0]                     = case_res.stability.CZ_v
        res.stability.static.CZ_w[i][0]                     = case_res.stability.CZ_w
        res.stability.static.Cl_u[i][0]                     = case_res.stability.Cl_u
        res.stability.static.Cl_v[i][0]                     = case_res.stability.Cl_v
        res.stability.static.Cl_w[i][0]                     = case_res.stability.Cl_w
        res.stability.static.Cm_u[i][0]                     = case_res.stability.Cm_u
        res.stability.static.Cm_v[i][0]                     = case_res.stability.Cm_v
        res.stability.static.Cm_w[i][0]                     = case_res.stability.Cm_w
        res.stability.static.Cn_u[i][0]                     = case_res.stability.Cn_u
        res.stability.static.Cn_v[i][0]                     = case_res.stability.Cn_v
        res.stability.static.Cn_w[i][0]                     = case_res.stability.Cn_w
        res.stability.static.CX_p[i][0]                     = case_res.stability.CX_p
        res.stability.static.CX_q[i][0]                     = case_res.stability.CX_q
        res.stability.static.CX_r[i][0]                     = case_res.stability.CX_r
        res.stability.static.CY_p[i][0]                     = case_res.stability.CY_p
        res.stability.static.CY_q[i][0]                     = case_res.stability.CY_q
        res.stability.static.CY_r[i][0]                     = case_res.stability.CY_r
        res.stability.static.CZ_p[i][0]                     = case_res.stability.CZ_p
        res.stability.static.CZ_q[i][0]                     = case_res.stability.CZ_q
        res.stability.static.CZ_r[i][0]                     = case_res.stability.CZ_r
        res.stability.static.Cl_p[i][0]                     = case_res.stability.Cl_p
        res.stability.static.Cl_q[i][0]                     = case_res.stability.Cl_q
        res.stability.static.Cl_r[i][0]                     = case_res.stability.Cl_r
        res.stability.static.Cm_p[i][0]                     = case_res.stability.Cm_p
        res.stability.static.Cm_q[i][0]                     = case_res.stability.Cm_q
        res.stability.static.Cm_r[i][0]                     = case_res.stability.Cm_r
        res.stability.static.Cn_p[i][0]                     = case_res.stability.Cn_p
        res.stability.static.Cn_q[i][0]                     = case_res.stability.Cn_q
        res.stability.static.Cn_r[i][0]                     = case_res.stability.Cn_r
        res.stability.static.neutral_point[i][0]            = case_res.stability.neutral_point
        res.stability.static.spiral_criteria[i][0]          = case_res.stability.spiral_criteria
        
        # aero surface forces file 
        res.aerodynamics.wing_areas[i][:]                   = case_res.aerodynamics.wing_areas   
        res.aerodynamics.wing_CLs[i][:]                     = case_res.aerodynamics.wing_CLs    
        res.aerodynamics.wing_CDs[i][:]                     = case_res.aerodynamics.wing_CDs    
        
        # aero sectional forces file
        res.aerodynamics.wing_local_spans[i][:]             = case_res.aerodynamics.wing_local_spans
        res.aerodynamics.wing_section_chords[i][:]          = case_res.aerodynamics.wing_section_chords  
        res.aerodynamics.wing_section_cls[i][:]             = case_res.aerodynamics.wing_section_cls    
        res.aerodynamics.wing_section_induced_angle[i][:]   = case_res.aerodynamics.wing_section_aoa_i
        res.aerodynamics.wing_section_cds[i][:]             = case_res.aerodynamics.wing_section_cds   
        
        res.stability.static.control_surfaces_cases[tag]    = case_res.stability.control_surfaces
        
    return res
