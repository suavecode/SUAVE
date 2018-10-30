## @ingroup Methods-Aerodynamics-AVL
#read_results.py
# 
# Created:  Mar 2015, T. Momose
# Modified: Jan 2016, E. Botero
#           Dec 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data

## @ingroup Methods-Aerodynamics-AVL
def read_results(avl_object):
    """ This functions reads the results from the results text file created 
    at the end of an AVL function call

    Assumptions:
        None
        
    Source:
        Drela, M. and Youngren, H., AVL, http://web.mit.edu/drela/Public/web/avl

    Inputs:
        None

    Outputs:
        results     

    Properties Used:
        N/A
    """    
    
    results = Data()
    #i = 0 Used in the dynamic stability module (under development)
    for case_name in avl_object.current_status.cases:
        case = avl_object.current_status.cases[case_name]
        num_ctrl =  case.stability_and_control.number_control_surfaces
        with open(case.result_filename,'r') as res_file:
            
            case_res = Data()  
            case_res.aerodynamics = Data()
            case_res.stability    = Data()
            case_res.stability.alpha_derivatives = Data()
            case_res.stability.beta_derivatives  = Data()   
            
            case_res.tag = case.tag
 
            lines   = res_file.readlines()
        
            case_res.aerodynamics.Sref = float(lines[8][10:16].strip())
            case_res.aerodynamics.Cref = float(lines[8][31:37].strip())
            case_res.aerodynamics.Bref = float(lines[8][52:58].strip())
            case_res.aerodynamics.Xref = float(lines[9][10:16].strip())
            case_res.aerodynamics.Yref = float(lines[9][31:37].strip())
            case_res.aerodynamics.Zref = float(lines[9][52:58].strip())  
        
            case_res.aerodynamics.CX = float(lines[19][11:19].strip())
            case_res.aerodynamics.CY = float(lines[20][11:19].strip()) 
            case_res.aerodynamics.CZ = float(lines[21][11:19].strip())
        
            case_res.aerodynamics.Cltot = float(lines[19][33:41].strip())
            case_res.aerodynamics.Cmtot = float(lines[20][33:41].strip()) 
            case_res.aerodynamics.Cntot = float(lines[21][33:41].strip())
        
            case_res.aerodynamics.roll_moment_coefficient  = float(lines[19][32:42].strip())
            case_res.aerodynamics.pitch_moment_coefficient = float(lines[20][32:42].strip())
            case_res.aerodynamics.yaw_moment_coefficient   = float(lines[21][32:42].strip())
            case_res.aerodynamics.total_lift_coefficient   = float(lines[23][10:20].strip())
            case_res.aerodynamics.total_drag_coefficient   = float(lines[24][10:20].strip())
            case_res.aerodynamics.induced_drag_coefficient = float(lines[25][32:42].strip())
            case_res.aerodynamics.span_efficiency_factor   = float(lines[27][32:42].strip())
        
            case_res.stability.alpha_derivatives.lift_curve_slope           = float(lines[36+num_ctrl][24:34].strip()) # CL_a
            case_res.stability.alpha_derivatives.side_force_derivative      = float(lines[37+num_ctrl][24:34].strip()) # CY_a
            case_res.stability.alpha_derivatives.roll_moment_derivative     = float(lines[38+num_ctrl][24:34].strip()) # Cl_a
            case_res.stability.alpha_derivatives.pitch_moment_derivative    = float(lines[39+num_ctrl][24:34].strip()) # Cm_a
            case_res.stability.alpha_derivatives.yaw_moment_derivative      = float(lines[40+num_ctrl][24:34].strip()) # Cn_a
            case_res.stability.beta_derivatives.lift_coefficient_derivative = float(lines[36+num_ctrl][43:54].strip()) # CL_b
            case_res.stability.beta_derivatives.side_force_derivative       = float(lines[37+num_ctrl][43:54].strip()) # CY_b
            case_res.stability.beta_derivatives.roll_moment_derivative      = float(lines[38+num_ctrl][43:54].strip()) # Cl_b
            case_res.stability.beta_derivatives.pitch_moment_derivative     = float(lines[39+num_ctrl][43:54].strip()) # Cm_b
            case_res.stability.beta_derivatives.yaw_moment_derivative       = float(lines[40+num_ctrl][43:54].strip()) # Cn_b
        
            case_res.stability.CL_p = float(lines[44+num_ctrl][24:34].strip())
            case_res.stability.CL_q = float(lines[44+num_ctrl][43:54].strip())
            case_res.stability.CL_r = float(lines[44+num_ctrl][65:74].strip())
            case_res.stability.CY_p = float(lines[45+num_ctrl][24:34].strip())
            case_res.stability.CY_q = float(lines[45+num_ctrl][43:54].strip())
            case_res.stability.CY_r = float(lines[45+num_ctrl][65:74].strip())
            case_res.stability.Cl_p = float(lines[46+num_ctrl][24:34].strip())
            case_res.stability.Cl_q = float(lines[46+num_ctrl][43:54].strip())
            case_res.stability.Cl_r = float(lines[44+num_ctrl][65:74].strip())
            case_res.stability.Cm_p = float(lines[47+num_ctrl][24:34].strip())
            case_res.stability.Cm_q = float(lines[47+num_ctrl][43:54].strip())
            case_res.stability.Cm_r = float(lines[44+num_ctrl][65:74].strip())
            case_res.stability.Cn_p = float(lines[48+num_ctrl][24:34].strip())
            case_res.stability.Cn_q = float(lines[48+num_ctrl][43:54].strip())
            case_res.stability.Cn_r = float(lines[48+num_ctrl][65:74].strip())
        
            case_res.stability.neutral_point  = float(lines[50+12*(num_ctrl>0)+num_ctrl][22:33].strip())
        
            results.append(case_res)        
       
        #------------------------------------------------------------------------------------------
        #          SUAVE-AVL dynamic stability analysis under development
        #          
        #with open(case.eigen_result_filename,'r') as eigen_res_file:
            #lines   = eigen_res_file.readlines()
            #index = i*8
            #case_res.stability.roll_mode_real             = float(lines[3+index][11:26].strip())
            #case_res.stability.dutch_roll_mode_1_real     = float(lines[4+index][11:26].strip())
            #case_res.stability.dutch_roll_mode_1_imag     = float(lines[4+index][29:40].strip())
            #case_res.stability.dutch_roll_mode_2_real     = float(lines[5+index][11:26].strip())
            #case_res.stability.dutch_roll_mode_2_imag     = float(lines[5+index][11:26].strip())
            #case_res.stability.short_period_mode_1_real   = float(lines[6+index][29:40].strip())
            #case_res.stability.short_period_mode_1_imag   = float(lines[6+index][11:26].strip())
            #case_res.stability.short_period_mode_2_real   = float(lines[7+index][29:40].strip())
            #case_res.stability.short_period_mode_2_imag   = float(lines[7+index][11:26].strip())
            #case_res.stability.spiral_mode_real           = float(lines[8+index][29:40].strip())
            #case_res.stability.phugoid_mode_1_real        = float(lines[9+index][11:26].strip())
            #case_res.stability.phugoid_mode_1_imag        = float(lines[9+index][29:40].strip())
            #case_res.stability.phugoid_mode_2_real        = float(lines[10+index][11:26].strip())                        
            #case_res.stability.phugoid_mode_2_imag        = float(lines[10+index][29:40].strip())
            
        #i += 1
        #results.append(case_res)
        #
        #------------------------------------------------------------------------------------------
    return results
