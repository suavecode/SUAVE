## @ingroup Methods-Aerodynamics-AVL
#translate_data.py
# 
# Created:  Mar 2015, T. Momose
# Modified: Jan 2016, E. Botero
#           Apr 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

import SUAVE
from SUAVE.Core import Data, Units
from .Data.Cases import Run_Case

## @ingroup Methods-Aerodynamics-AVL
def translate_conditions_to_cases(avl,conditions):
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
    cases = Run_Case.Container()
    for i in range(len(conditions.aerodynamics.angle_of_attack)):      
        case = Run_Case()
        case.tag  = avl.settings.filenames.case_template.format(avl.current_status.batch_index,i+1)
        case.mass = conditions.weights.total_mass
        case.conditions.freestream.mach     = conditions.freestream.mach_number
        case.conditions.freestream.density  = conditions.freestream.density
        case.conditions.freestream.gravitational_acceleration = conditions.freestream.gravity
        case.conditions.aerodynamics.angle_of_attack = conditions.aerodynamics.angle_of_attack[i]/Units.deg
        case.conditions.aerodynamics.side_slip_angle = 0 
        case.stability_and_control.control_deflections = np.array([[]]) 
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
         
            case_res.aerodynamics.Sref - reference area                                        [meters**2]
            case_res.aerodynamics.Cref - reference chord                                       [meters]
            case_res.aerodynamics.Bref - reference span                                        [meters]
            case_res.aerodynamics.Xref - x cooridate of reference(used for evaluating moments) [meters] 
            case_res.aerodynamics.Yref - y cooridate of reference(used for evaluating moments) [meters] 
            case_res.aerodynamics.Zref - z cooridate of reference(used for evaluating moments) [meters] 
            case_res.aerodynamics.CX  
            case_res.aerodynamics.CY   
            case_res.aerodynamics.CZ  
            
            case_res.aerodynamics.Cltot - total roll coefficient                [dimensionless]                         
            case_res.aerodynamics.Cmtot - total pitch coefficient               [dimensionless]              
            case_res.aerodynamics.Cntot - total yaw coefficent                  [dimensionless]        
            
            case_res.aerodynamics.roll_moment_coefficient                       [dimensionless] 
            case_res.aerodynamics.pitch_moment_coefficient                      [dimensionless] 
            case_res.aerodynamics.yaw_moment_coefficient                        [dimensionless] 
            case_res.aerodynamics.total_lift_coefficient                        [dimensionless] 
            case_res.aerodynamics.induced_drag_coefficient                      [dimensionless] 
            case_res.aerodynamics.span_efficiency_factor                        [dimensionless] 
                    
            case_res.stability.alpha_derivatives.lift_curve_slope               [dimensionless] 
            case_res.stability.alpha_derivatives.side_force_derivative          [dimensionless] 
            case_res.stability.alpha_derivatives.roll_moment_derivative         [dimensionless] 
            case_res.stability.alpha_derivatives.pitch_moment_derivative        [dimensionless] 
            case_res.stability.alpha_derivatives.yaw_moment_derivative          [dimensionless] 
            case_res.stability.beta_derivatives.lift_coefficient_derivative     [dimensionless] 
            case_res.stability.beta_derivatives.side_force_derivative           [dimensionless] 
            case_res.stability.beta_derivatives.roll_moment_derivative          [dimensionless] 
            case_res.stability.beta_derivatives.pitch_moment_derivative         [dimensionless] 
            case_res.stability.beta_derivatives.yaw_moment_derivative           [dimensionless] 
            
            case_res.stability.CL_p                        [dimensionless] 
            case_res.stability.CL_q                        [dimensionless]   
            case_res.stability.CL_r                        [dimensionless]   
            case_res.stability.CY_p                        [dimensionless]  
            case_res.stability.CY_q                        [dimensionless]    
            case_res.stability.CY_r                        [dimensionless]    
            case_res.stability.Cl_p                        [dimensionless]    
            case_res.stability.Cl_q                        [dimensionless]    
            case_res.stability.Cl_r                        [dimensionless]    
            case_res.stability.Cm_p                        [dimensionless]     
            case_res.stability.Cm_q                        [dimensionless]    
            case_res.stability.Cm_r                        [dimensionless]     
            case_res.stability.Cn_p                        [dimensionless]     
            case_res.stability.Cn_q                        [dimensionless] 
            case_res.stability.Cn_r                        [dimensionless] 
            
     Outputs:
        cases                        [data_structure]
   
    Properties Used:
        N/A
    """   
    # set up aerodynamic Conditions object
    res = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    ones_1col = res.ones_row(1)
    # add missing entries
       
    res.aerodynamics.Sref  = ones_1col * 0
    res.aerodynamics.Cref  = ones_1col * 0
    res.aerodynamics.Bref  = ones_1col * 0
    res.aerodynamics.Xref  = ones_1col * 0
    res.aerodynamics.Yref  = ones_1col * 0
    res.aerodynamics.Zref  = ones_1col * 0   
    res.aerodynamics.CX    = ones_1col * 0
    res.aerodynamics.CY    = ones_1col * 0
    res.aerodynamics.CZ    = ones_1col * 0
    
    res.aerodynamics.Cltot = ones_1col * 0
    res.aerodynamics.Cmtot = ones_1col * 0 
    res.aerodynamics.Cntot = ones_1col * 0     
    
    res.aerodynamics.roll_moment_coefficient  = ones_1col * 0
    res.aerodynamics.pitch_moment_coefficient = ones_1col * 0
    res.aerodynamics.yaw_moment_coefficient   = ones_1col * 0
    res.aerodynamics.drag_breakdown.induced   = SUAVE.Analyses.Mission.Segments.Conditions.Conditions()
    res.aerodynamics.drag_breakdown.induced.total = ones_1col * 0
    res.aerodynamics.drag_breakdown.induced.efficiency_factor = ones_1col * 0
    
    res.aerodynamics.cz_alpha                 = ones_1col * 0
    res.aerodynamics.cy_alpha                 = ones_1col * 0
    res.aerodynamics.cl_alpha                 = ones_1col * 0
    res.aerodynamics.cm_alpha                 = ones_1col * 0
    res.aerodynamics.cn_alpha                 = ones_1col * 0
    res.aerodynamics.cz_beta                  = ones_1col * 0
    res.aerodynamics.cy_beta                  = ones_1col * 0
    res.aerodynamics.cl_beta                  = ones_1col * 0
    res.aerodynamics.cm_beta                  = ones_1col * 0
    res.aerodynamics.cn_beta                  = ones_1col * 0
    
    res.aerodynamics.CL_p                     = ones_1col * 0
    res.aerodynamics.CL_q                     = ones_1col * 0
    res.aerodynamics.CL_r                     = ones_1col * 0
    res.aerodynamics.CY_p                     = ones_1col * 0
    res.aerodynamics.CY_q                     = ones_1col * 0
    res.aerodynamics.CY_r                     = ones_1col * 0
    res.aerodynamics.Cl_p                     = ones_1col * 0
    res.aerodynamics.Cl_q                     = ones_1col * 0
    res.aerodynamics.Cl_r                     = ones_1col * 0
    res.aerodynamics.Cm_p                     = ones_1col * 0
    res.aerodynamics.Cm_q                     = ones_1col * 0 
    res.aerodynamics.Cm_r                     = ones_1col * 0
    res.aerodynamics.Cn_p                     = ones_1col * 0
    res.aerodynamics.Cn_q                     = ones_1col * 0
    res.aerodynamics.Cn_r                     = ones_1col * 0
   
    res.aerodynamics.neutral_point            = ones_1col * 0
    
    res.aerodynamics.roll_mode                = ones_1col * 0  
    res.aerodynamics.dutch_roll_mode_1_real   = ones_1col * 0
    res.aerodynamics.dutch_roll_mode_1_imag   = ones_1col * 0
    res.aerodynamics.dutch_roll_mode_2_real   = ones_1col * 0 
    res.aerodynamics.dutch_roll_mode_2_imag   = ones_1col * 0 
    res.aerodynamics.short_period_mode_1_real = ones_1col * 0
    res.aerodynamics.short_period_mode_1_imag = ones_1col * 0
    res.aerodynamics.short_period_mode_2_real = ones_1col * 0
    res.aerodynamics.short_period_mode_2_imag = ones_1col * 0
    res.aerodynamics.spiral_mode              = ones_1col * 0       
    res.aerodynamics.phugoid_mode_mode_1_real = ones_1col * 0
    res.aerodynamics.phugoid_mode_mode_1_imag = ones_1col * 0
    res.aerodynamics.phugoid_mode_mode_2_real = ones_1col * 0
    res.aerodynamics.phugoid_mode_mode_2_imag = ones_1col * 0       

    res.expand_rows(len(cases))

    mach_case = list(results.keys())[0][5:8]   
    for i in range(len(results.keys())):
        aoa_case = '{:02d}'.format(i+1)
        tag = 'case_' + mach_case + '_' + aoa_case
        case_res = results[tag]
       

        res.aerodynamics.Sref[i][0] = case_res.aerodynamics.Sref 
        res.aerodynamics.Cref[i][0] = case_res.aerodynamics.Cref 
        res.aerodynamics.Bref[i][0] = case_res.aerodynamics.Bref
        res.aerodynamics.Xref[i][0] = case_res.aerodynamics.Xref 
        res.aerodynamics.Yref[i][0] = case_res.aerodynamics.Yref 
        res.aerodynamics.Zref[i][0] = case_res.aerodynamics.Zref       
        res.aerodynamics.CX[i][0]   = case_res.aerodynamics.CX 
        res.aerodynamics.CY[i][0]   = case_res.aerodynamics.CY  
        res.aerodynamics.CZ[i][0]   = case_res.aerodynamics.CZ 
        
        
        res.aerodynamics.Cltot[i][0] = case_res.aerodynamics.Cltot 
        res.aerodynamics.Cmtot[i][0] = case_res.aerodynamics.Cmtot 
        res.aerodynamics.Cntot[i][0] = case_res.aerodynamics.Cntot        
    
        res.aerodynamics.roll_moment_coefficient[i][0]      = case_res.aerodynamics.roll_moment_coefficient
        res.aerodynamics.pitch_moment_coefficient[i][0]     = case_res.aerodynamics.pitch_moment_coefficient
        res.aerodynamics.yaw_moment_coefficient[i][0]       = case_res.aerodynamics.yaw_moment_coefficient
        res.aerodynamics.lift_coefficient[i][0]             = case_res.aerodynamics.total_lift_coefficient
        res.aerodynamics.drag_breakdown.induced.total[i][0] = case_res.aerodynamics.induced_drag_coefficient
        res.aerodynamics.drag_breakdown.induced.efficiency_factor[i][0]  = case_res.aerodynamics.span_efficiency_factor
        res.aerodynamics.cz_alpha[i][0] = -case_res.stability.alpha_derivatives.lift_curve_slope
        res.aerodynamics.cy_alpha[i][0] = case_res.stability.alpha_derivatives.side_force_derivative
        res.aerodynamics.cl_alpha[i][0] = case_res.stability.alpha_derivatives.roll_moment_derivative
        res.aerodynamics.cm_alpha[i][0] = case_res.stability.alpha_derivatives.pitch_moment_derivative
        res.aerodynamics.cn_alpha[i][0] = case_res.stability.alpha_derivatives.yaw_moment_derivative
        res.aerodynamics.cz_beta[i][0]  = -case_res.stability.beta_derivatives.lift_coefficient_derivative
        res.aerodynamics.cy_beta[i][0]  = case_res.stability.beta_derivatives.side_force_derivative
        res.aerodynamics.cl_beta[i][0]  = case_res.stability.beta_derivatives.roll_moment_derivative
        res.aerodynamics.cm_beta[i][0]  = case_res.stability.beta_derivatives.pitch_moment_derivative
        res.aerodynamics.cn_beta[i][0]  = case_res.stability.beta_derivatives.yaw_moment_derivative
        
        res.aerodynamics.CL_p[i][0] = case_res.stability.CL_p 
        res.aerodynamics.CL_q[i][0] = case_res.stability.CL_q
        res.aerodynamics.CL_r[i][0] = case_res.stability.CL_r 
        res.aerodynamics.CY_p[i][0] = case_res.stability.CY_p 
        res.aerodynamics.CY_q[i][0] = case_res.stability.CY_q 
        res.aerodynamics.CY_r[i][0] = case_res.stability.CY_r
        res.aerodynamics.Cl_p[i][0] = case_res.stability.Cl_p 
        res.aerodynamics.Cl_q[i][0] = case_res.stability.Cl_q 
        res.aerodynamics.Cl_r[i][0] = case_res.stability.Cl_r 
        res.aerodynamics.Cm_p[i][0] = case_res.stability.Cm_p 
        res.aerodynamics.Cm_q[i][0] = case_res.stability.Cm_q 
        res.aerodynamics.Cm_r[i][0] = case_res.stability.Cm_r 
        res.aerodynamics.Cn_p[i][0] = case_res.stability.Cn_p 
        res.aerodynamics.Cn_q[i][0] = case_res.stability.Cn_q 
        res.aerodynamics.Cn_r[i][0] = case_res.stability.Cn_r 
        
        res.aerodynamics.neutral_point[i][0] = case_res.stability.neutral_point
        
        #-----------------------------------------------------------------------------------------------
        #                         SUAVE-AVL dynamic stability analysis under development
        #  
        #res.aerodynamics.roll_mode[i][0]                = case_res.stability.roll_mode_real        
        #res.aerodynamics.dutch_roll_mode_1_real[i][0]   = case_res.stability.dutch_roll_mode_1_real
        #res.aerodynamics.dutch_roll_mode_1_imag[i][0]   = case_res.stability.dutch_roll_mode_1_imag 
        #res.aerodynamics.dutch_roll_mode_2_real[i][0]   = case_res.stability.dutch_roll_mode_2_real 
        #res.aerodynamics.dutch_roll_mode_2_imag[i][0]   = case_res.stability.dutch_roll_mode_2_imag 
        #res.aerodynamics.short_period_mode_1_real[i][0] = case_res.stability.short_period_mode_1_real 
        #res.aerodynamics.short_period_mode_1_imag[i][0] = case_res.stability.short_period_mode_1_imag
        #res.aerodynamics.short_period_mode_2_real[i][0] = case_res.stability.short_period_mode_2_real 
        #res.aerodynamics.short_period_mode_2_imag[i][0] = case_res.stability.short_period_mode_2_imag 
        #res.aerodynamics.spiral_mode[i][0]              = case_res.stability.spiral_mode_real       
        #res.aerodynamics.phugoid_mode_mode_1_real[i][0] = case_res.stability.phugoid_mode_1_real
        #res.aerodynamics.phugoid_mode_mode_1_imag[i][0] = case_res.stability.phugoid_mode_1_imag
        #res.aerodynamics.phugoid_mode_mode_2_real[i][0] = case_res.stability.phugoid_mode_2_real 
        #res.aerodynamics.phugoid_mode_mode_2_imag[i][0] = case_res.stability.phugoid_mode_2_imag      
        #
        #-----------------------------------------------------------------------------------------------

    return res