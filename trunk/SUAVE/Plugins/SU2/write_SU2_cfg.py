
def write_SU2_cfg(tag,SU2_settings):
    
    ref_area = SU2_settings.reference_area
    mach     = SU2_settings.mach_number
    AoA      = SU2_settings.angle_of_attack
    
    
    filename = tag + '.cfg'
    f = open(filename,mode='w')

    # Problem definition
    f.write('PHYSICAL_PROBLEM = EULER\n\n')
    f.write('MATH_PROBLEM = DIRECT\n\n')
    f.write('RESTART_SOL = NO\n\n')
    
    # Freestream definition
    f.write('MACH_NUMBER = ' + str(float(mach)) + '\n\n')
    f.write('AoA = ' + str(float(AoA)) + '\n\n')
    f.write('SIDESLIP_ANGLE = 0.0\n\n')
    f.write('FREESTREAM_PRESSURE = 101325.0\n\n')
    f.write('FREESTREAM_TEMPERATURE = 288.15\n\n')
    
    # Reference definition
    f.write('REF_ORIGIN_MOMENT_X = 0.25\n\n')
    f.write('REF_ORIGIN_MOMENT_Y = 0.00\n\n')
    f.write('REF_ORIGIN_MOMENT_Z = 0.00\n\n')
    f.write('REF_LENGTH_MOMENT = 1.0\n\n')
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
    f.write('CFL_ADAPT = NO\n\n')
    f.write('CFL_ADAPT_PARAM = ( 1.5, 0.5, 1.0, 100.0 )\n\n')
    f.write('RK_ALPHA_COEFF = ( 0.66667, 0.66667, 1.000000 )\n\n')
    f.write('EXT_ITER = 99999\n\n')
    f.write('LINEAR_SOLVER = FGMRES\n\n')
    f.write('LINEAR_SOLVER_ERROR = 1E-6\n\n')
    f.write('LINEAR_SOLVER_ITER = 2\n\n')
    
    # Slope limiter
    f.write('REF_ELEM_LENGTH = 0.1\n\n')
    f.write('LIMITER_COEFF = 0.3\n\n')
    f.write('SHARP_EDGES_COEFF = 3.0\n\n')
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
    f.write('SPATIAL_ORDER_FLOW = 2ND_ORDER\n\n')
    f.write('SLOPE_LIMITER_FLOW = VENKATAKRISHNAN\n\n')
    f.write('AD_COEFF_FLOW = ( 0.15, 0.5, 0.02 )\n\n')
    f.write('TIME_DISCRE_FLOW = EULER_IMPLICIT\n\n')
    
    # Adjoint-flow numerical method
    f.write('CONV_NUM_METHOD_ADJFLOW = JST\n\n')
    f.write('SPATIAL_ORDER_ADJFLOW = 2ND_ORDER\n\n')
    f.write('SLOPE_LIMITER_ADJFLOW = VENKATAKRISHNAN\n\n')
    f.write('AD_COEFF_ADJFLOW = ( 0.15, 0.0, 0.02 )\n\n')
    f.write('CFL_REDUCTION_ADJFLOW = 0.5\n\n')
    f.write('TIME_DISCRE_ADJFLOW = EULER_IMPLICIT\n\n')
    
    # Convergence parameters
    f.write('CONV_CRITERIA = RESIDUAL\n\n')
    f.write('RESIDUAL_REDUCTION = 8\n\n')
    f.write('RESIDUAL_MINVAL = -12\n\n')
    f.write('STARTCONV_ITER = 25\n\n')
    f.write('CAUCHY_ELEMS = 100\n\n')
    f.write('CAUCHY_EPS = 1E-10\n\n')
    f.write('CAUCHY_FUNC_FLOW = DRAG\n\n')
    
    # Input/Output
    f.write('MESH_FILENAME = ' + tag + '.su2\n\n')
    f.write('MESH_OUT_FILENAME = mesh_out.su2\n\n')
    f.write('SOLUTION_FLOW_FILENAME = solution_flow.dat\n\n')
    f.write('SOLUTION_ADJ_FILENAME = solution_adj.dat\n\n')
    f.write('MESH_FORMAT = SU2\n\n')
    f.write('OUTPUT_FORMAT = TECPLOT\n\n')
    f.write('CONV_FILENAME = history\n\n')
    f.write('BREAKDOWN_FILENAME = ' + tag + '_forces_breakdown.dat\n\n')
    f.write('RESTART_FLOW_FILENAME = ' + tag + '_restart_flow.dat\n\n')
    f.write('RESTART_ADJ_FILENAME = restart_adj.dat\n\n')
    f.write('VOLUME_FLOW_FILENAME = ' + tag + '_flow\n\n')
    f.write('VOLUME_ADJ_FILENAME = adjoint\n\n')
    f.write('GRAD_OBJFUNC_FILENAME = of_grad.dat\n\n')
    f.write('SURFACE_FLOW_FILENAME = ' + tag + '_surface_flow\n\n')
    f.write('SURFACE_ADJ_FILENAME = surface_adjoint\n\n')
    f.write('WRT_SOL_FREQ = 100\n\n')
    f.write('WRT_CON_FREQ = 1\n\n')
    
    f.close()