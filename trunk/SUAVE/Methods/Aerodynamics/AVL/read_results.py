# read_results.py
# 
# Created:  Mar 2015, T. Momose
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from .Data.Results import Results

def read_results(avl_object):

    results = Data()

    for case in avl_object.current_status.cases:
        num_ctrl = case.stability_and_control.control_deflections.size
        with open(case.result_filename,'r') as res_file:
            case_res = Results()
            case_res.tag = case.tag
            lines   = res_file.readlines()
            case_res.aerodynamics.roll_moment_coefficient  = float(lines[19][32:42].strip())
            case_res.aerodynamics.pitch_moment_coefficient = float(lines[20][32:42].strip())
            case_res.aerodynamics.yaw_moment_coefficient   = float(lines[21][32:42].strip())
            case_res.aerodynamics.total_lift_coefficient   = float(lines[23][10:20].strip())
            case_res.aerodynamics.total_drag_coefficient   = float(lines[24][10:20].strip())
            case_res.aerodynamics.induced_drag_coefficient = float(lines[25][32:42].strip())
            case_res.aerodynamics.span_efficiency_factor   = float(lines[27][32:42].strip())

            case_res.stability.alpha_derivatives.lift_curve_slope           = float(lines[36+num_ctrl][25:35].strip())
            case_res.stability.alpha_derivatives.side_force_derivative      = float(lines[37+num_ctrl][25:35].strip())
            case_res.stability.alpha_derivatives.roll_moment_derivative     = float(lines[38+num_ctrl][25:35].strip())
            case_res.stability.alpha_derivatives.pitch_moment_derivative    = float(lines[39+num_ctrl][25:35].strip())
            case_res.stability.alpha_derivatives.yaw_moment_derivative      = float(lines[40+num_ctrl][25:35].strip())
            case_res.stability.beta_derivatives.lift_coefficient_derivative = float(lines[36+num_ctrl][44:55].strip())
            case_res.stability.beta_derivatives.side_force_derivative       = float(lines[37+num_ctrl][44:55].strip())
            case_res.stability.beta_derivatives.roll_moment_derivative      = float(lines[38+num_ctrl][44:55].strip())
            case_res.stability.beta_derivatives.pitch_moment_derivative     = float(lines[39+num_ctrl][44:55].strip())
            case_res.stability.beta_derivatives.yaw_moment_derivative       = float(lines[40+num_ctrl][44:55].strip())
            case_res.stability.neutral_point                                = float(lines[50+13*(num_ctrl>0)][22:33].strip())

            results.append(case_res)


    return results
