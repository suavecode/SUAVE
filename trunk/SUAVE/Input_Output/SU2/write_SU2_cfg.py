## @ingroup Input_Output-SU2
# write_SU2_cfg.py
# 
# Created:  Oct 2016, T. MacDonald
# Modified: Jan 2017, T. MacDonald
#           Mar 2018, T. MacDonald
#           Mar 2020, E. Botero

## @ingroup Input_Output-SU2
def write_SU2_cfg(tag,SU2_settings):
    """Creates an SU2 .cfg file that can be used for an SU2 run.

    Assumptions:
    Almost all settings are current hard coded.

    Source:
    N/A

    Inputs:
    tag                          <string>  This determines the name of the .cfg
    SU2_settings.
      reference_area            [m^2]
      mach_number               [-]
      angle_of_attack           [degrees]
      maximum_iterations        [-]

    Outputs:
    <tag>.cfg

    Properties Used:
    N/A
    """      
    
    ref_area = SU2_settings.reference_area
    mach     = SU2_settings.mach_number
    AOA      = SU2_settings.angle_of_attack
    iters    = SU2_settings.maximum_iterations
    
    
    filename = tag + '.cfg'
    f = open(filename,mode='w')

    # Problem definition
    f.write('SOLVER = EULER\n\n')
    f.write('KIND_TURB_MODEL = NONE\n\n')
    f.write('KIND_VERIFICATION_SOLUTION= NO_VERIFICATION_SOLUTION\n\n')
    f.write('MATH_PROBLEM = DIRECT\n\n')
    f.write('AXISYMMETRIC= NO\n\n')
    f.write('RESTART_SOL = NO\n\n')
    f.write('DISCARD_INFILES= NO\n\n')
    f.write('SYSTEM_MEASUREMENTS= SI\n\n')
    
    # Freestream definition
    f.write('MACH_NUMBER = ' + str(float(mach)) + '\n\n')
    f.write('AOA = ' + str(float(AOA)) + '\n\n')
    f.write('SIDESLIP_ANGLE = 0.0\n\n')
    f.write('FREESTREAM_PRESSURE = 101325.0\n\n')
    f.write('FREESTREAM_TEMPERATURE = 288.15\n\n')
    
    # Reference definition
    f.write('REF_ORIGIN_MOMENT_X = 0.25\n\n')
    f.write('REF_ORIGIN_MOMENT_Y = 0.00\n\n')
    f.write('REF_ORIGIN_MOMENT_Z = 0.00\n\n')
    f.write('REF_LENGTH = 1.0\n\n')
    f.write('REF_AREA = ' + str(float(ref_area)) + '\n\n')
    f.write('REF_DIMENSIONALIZATION = FREESTREAM_VEL_EQ_ONE\n\n')
    
    # Boundary conditions
    ## need a way to handle if there is a symmetry plane or not
    f.write('MARKER_EULER = ( VEHICLE )\n\n')
    f.write('MARKER_FAR = ( FARFIELD )\n\n')
    f.write('MARKER_SYM = ( SYMPLANE )\n\n')
    
    # Surface IDs
    f.write('MARKER_PLOTTING = ( VEHICLE )\n\n')
    f.write('MARKER_MONITORING = ( VEHICLE )\n\n')
    f.write('MARKER_DESIGNING = ( VEHICLE )\n\n')
    
    # Common numerical method parameters
    f.write('NUM_METHOD_GRAD = WEIGHTED_LEAST_SQUARES\n\n')
    f.write('OBJECTIVE_FUNCTION = DRAG\n\n')
    f.write('CFL_NUMBER = 5.0\n\n')
    f.write('CFL_ADAPT = YES\n\n')
    f.write('CFL_ADAPT_PARAM = ( 1.5, 0.5, 1.0, 100.0 )\n\n')
    f.write('RK_ALPHA_COEFF = ( 0.66667, 0.66667, 1.000000 )\n\n')
    f.write('INNER_ITER ='+str(int(iters)) +'\n\n')
    f.write('LINEAR_SOLVER = FGMRES\n\n')
    f.write('LINEAR_SOLVER_ERROR = 1E-6\n\n')
    f.write('LINEAR_SOLVER_ITER = 2\n\n')
    
    # Slope limiter
    f.write('VENKAT_LIMITER_COEFF = 0.3\n\n')
    f.write('ADJ_SHARP_LIMITER_COEFF = 3.0\n\n')
    f.write('REF_SHARP_EDGES = 3.0\n\n')
    f.write('SENS_REMOVE_SHARP = YES\n\n')
    
    # Multigrid parameters
    f.write('MGLEVEL = 3\n\n')
    f.write('MGCYCLE = W_CYCLE\n\n')
    f.write('MG_PRE_SMOOTH = ( 1, 2, 3, 3 )\n\n')
    f.write('MG_POST_SMOOTH = ( 0, 0, 0, 0 )\n\n')
    f.write('MG_CORRECTION_SMOOTH = ( 0, 0, 0, 0 )\n\n')
    f.write('MG_DAMP_RESTRICTION = 0.9\n\n')
    f.write('MG_DAMP_PROLONGATION = 0.9\n\n')
    
    # Flow numerical method
    f.write('CONV_NUM_METHOD_FLOW = JST\n\n')
    f.write('MUSCL_FLOW = YES\n\n')
    f.write('SLOPE_LIMITER_FLOW = VENKATAKRISHNAN\n\n')
    f.write('JST_SENSOR_COEFF = ( 0.5, 0.02 )\n\n')
    f.write('TIME_DISCRE_FLOW = EULER_IMPLICIT\n\n')
    
    # Adjoint-flow numerical method
    f.write('CONV_NUM_METHOD_ADJFLOW = JST\n\n')
    f.write('MUSCL_ADJFLOW = YES\n\n')
    f.write('SLOPE_LIMITER_ADJFLOW = VENKATAKRISHNAN\n\n')
    f.write('ADJ_JST_SENSOR_COEFF = ( 0.0, 0.02 )\n\n')
    f.write('CFL_REDUCTION_ADJFLOW = 0.5\n\n')
    f.write('TIME_DISCRE_ADJFLOW = EULER_IMPLICIT\n\n')
    
    # Convergence parameters
    f.write('CONV_CRITERIA = CAUCHY\n\n')
    f.write('CONV_RESIDUAL_MINVAL = -12\n\n')
    f.write('CONV_STARTITER = 25\n\n')
    f.write('CONV_CAUCHY_ELEMS = 100\n\n')
    f.write('CONV_CAUCHY_EPS = 1E-6\n\n')
    
    # Input/Output
    f.write('SCREEN_OUTPUT= ( LIFT, DRAG, INNER_ITER, WALL_TIME, RMS_DENSITY, RMS_ENERGY)\n\n')
    f.write('HISTORY_OUTPUT= ( ITER, AERO_COEFF, RMS_RES )\n\n')
    f.write('MESH_FILENAME = ' + tag + '.su2\n\n')
    f.write('MESH_OUT_FILENAME = mesh_out.su2\n\n')
    f.write('SOLUTION_FILENAME = solution_flow.dat\n\n')
    f.write('SOLUTION_ADJ_FILENAME = solution_adj.dat\n\n')
    f.write('MESH_FORMAT = SU2\n\n')
    f.write('TABULAR_FORMAT = TECPLOT\n\n')
    f.write('CONV_FILENAME = ' + tag + '_history\n\n')
    f.write('BREAKDOWN_FILENAME = ' + tag + '_forces_breakdown.dat\n\n')
    f.write('RESTART_FILENAME = ' + tag + '_restart_flow.dat\n\n')
    f.write('RESTART_ADJ_FILENAME = restart_adj.dat\n\n')
    f.write('VOLUME_FILENAME = ' + tag + '_flow\n\n')
    f.write('VOLUME_ADJ_FILENAME = adjoint\n\n')
    f.write('GRAD_OBJFUNC_FILENAME = of_grad.dat\n\n')
    f.write('SURFACE_FILENAME = ' + tag + '_surface_flow\n\n')
    f.write('SURFACE_ADJ_FILENAME = surface_adjoint\n\n')
    f.write('WRT_SOL_FREQ = 1000\n\n')
    f.write('WRT_CON_FREQ = 1\n\n')
    
    f.close()
