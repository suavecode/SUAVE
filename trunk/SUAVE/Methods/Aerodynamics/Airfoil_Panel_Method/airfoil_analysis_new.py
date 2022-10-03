## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# airfoil_analysis.py
# 
# Created:  Aug 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

import SUAVE 
from SUAVE.Core import Data  
import numpy as np    
from SUAVE.Methods.Aerodynamics.Airfoil_Panel_Method.airfoil_paneling     import * 
from SUAVE.Methods.Aerodynamics.Airfoil_Panel_Method.post_processing      import * 
from SUAVE.Methods.Aerodynamics.Airfoil_Panel_Method.supporting_functions import * 
from SUAVE.Methods.Aerodynamics.Airfoil_Panel_Method.inviscid_functions   import * 
from SUAVE.Methods.Aerodynamics.Airfoil_Panel_Method.viscous_functions    import *

def airfoil_analysis_new(airfoil_geometry,AoA,Re,Ma,airfoil_stations = [0],viscous_flag = False, moment_reference_x = 0.25 , moment_reference_z = 0.25):
    ''' This computes the aerodynamic polars as well as the boundary layer properties of 
    an airfoil at a defined set of reynolds numbers and angle of attacks

    Assumptions: 

    Source:
    N/A

    Inputs: 
    airfoil_geometry   - airfoil geometry points                                                             [unitless]
    AoA                - angle of attacks                                                                    [radians]
    Re                 - Reynolds numbers                                                                    [unitless]
    Ma                 - Mach numbers                                                                        [unitless]
    rho                - Density                                                                             [kg/m^3]
    Vinf               - Freestream velocity                                                                 [m/s]
    npanel             - number of airfoil panels                                                            [unitless] 
    airfoil_stations   - airfoil station                                                                     [unitless]
    viscous_flag       - viscous flag                                                                        [unitless]
    cl_target_flag     - lift coefficent flag                                                                [unitless]
    moment_reference_x - moment reference point x coordinate                                                 [m]
    moment_reference_z - moment reference point y coordinate                                                 [m] 
    
    Outputs: 
    airfoil_properties. 
        cp        -  pressure coefficient distribution                                 [unitless] 
        cpi        - inviscid pressure coefficient distribution                        [unitless]
        cl         - lift coefficients                                                 [unitless]     
        cl_ue      - linearization of cl w.r.t. ue                                     [unitless]
        normals    - surface normals                                                   [unitless]
        cl_alpha   - linearization of cl w.r.t. alpha                                  [unitless]
        cm         - moment coefficients                                               [unitless]
        normals    - surface normals of airfoil                                        [unitless]
        x          - x coordinate points on airfoil                                    [unitless]
        y          - y coordinate points on airfoil                                    [unitless] 
        cdpi       - near-field pressure drag coefficient                              [unitless]
        cd         - total drag coefficient                                            [unitless]
        cdf        - skin friction drag coefficient                                    [unitless]
        cdp        - pressure drag coefficient                                         [unitless]
        theta      - momentum thickness                                                [m]
        delta_star - displacement thickness                                            [m]
        delta      - boundary layer thickness                                          [m] 
        sa         - amplification factor/shear lag coeff distribution                 [unitless]
        ue         - edge velocity (compressible) distribution                         [m/s]
        uei        - inviscid edge velocity (compressible) distribution                [m/s]
        cf         - local skin friction coefficient                                   [unitless] 
        Re_theta   - Re_theta distribution                                             [unitless] 
        H          - kinematic shape parameter distribution                            [unitless]  
        dVe        - derivative of boundary layer velocity                             [m/s-m] 

    Properties Used:
    N/A  

    '''  
     
    # number of cases 
    num_cases = len(airfoil_stations)
    npanel    = len(airfoil_geometry.x_coordinates[0]) 
    
    # airfoil results data structure 
    Airfoil_results   = initialize_results(num_cases,npanel,viscous_flag)

    for case in range(num_cases): 
        Foil = initialize_case(airfoil_geometry,AoA[case,0],Re[case,0],Ma[case,0],npanel,airfoil_stations[case],viscous_flag, moment_reference_x, moment_reference_z)
    
        if (Foil.oper.viscous):
            solve_viscous(Foil) 
        else:
            solve_inviscid(Foil)  
        set_results(Foil,Airfoil_results,case,viscous_flag)
        
    return Airfoil_results


def initialize_case(airfoil_geometry,AoA,Re,Ma,npanel,airfoil_stations,viscous_flag, moment_reference_x, moment_reference_z): 
    """ Initializes airfoil analysis and sets default data structures 
 
    Assumptions: 
      None 


    Source:
      N/A

    Inputs: 
       airfoil_geometry   - airfoil geometry points                 [unitless]
       AoA                - angle of attacks                        [radians]
       Re                 - Reynolds numbers                        [unitless]
       Ma                 - Mach numbers                            [unitless]
       rho                - Density                                 [kg/m^3]
       Vinf               - Freestream velocity                     [m/s]
       npanel             - number of airfoil panels                [unitless] 
       airfoil_stations   - airfoil station                         [unitless]
       viscous_flag       - viscous flag                            [unitless]
       cl_target_flag     - lift coefficent flag                    [unitless]
       moment_reference_x - moment reference point x coordinate     [m]
       moment_reference_z - moment reference point y coordinate     [m] 
    
    Outputs: 
       Airfoils           - data structure storing results from analyses 

    Properties Used:
       N/A   
    
    """ 

    x_pts                      = np.take(airfoil_geometry.x_coordinates,airfoil_stations,axis=0).T 
    z_pts                      = np.take(airfoil_geometry.y_coordinates,airfoil_stations,axis=0).T 
          
    Airfoils                   = Data() 
    Airfoils.geom              = Data()    # geometry
    Airfoils.geom.chord        = 1         # chord length
    Airfoils.geom.wakelen      = 1.0       # wake extent length, in chords
    Airfoils.geom.npoint       = 0         # number of geometry representation points   
    Airfoils.geom.xpoint       = np.concatenate((x_pts[None,:],z_pts[None,:]),axis = 0)   
    Airfoils.geom.xref         = np.array([moment_reference_x,moment_reference_z]) # moment reference point    

    # airfoil panels 
    Airfoils.foil              = Data()
    Airfoils.foil.N            = 0            # number of nodes
    Airfoils.foil.x            = []           # node coordinates, [2 x N]
    Airfoils.foil.s            = []           # arclength values at nodes
    Airfoils.foil.t            = []           # dx/ds, dy/ds tangents at nodes 
        
    Airfoils.wake              = Data()          # wake panels  
    Airfoils.wake.N            = 0               # number of nodes
    Airfoils.wake.x            = np.empty([1,0]) # node coordinates, [2 x N]
    Airfoils.wake.s            = np.empty([1,0]) # arclength values at nodes
    Airfoils.wake.t            = np.empty([1,0]) # dx/ds, dy/ds tangents at nodes
    Airfoils.oper              = Data()
    Airfoils.oper.Vinf         = 1.                # velocity magnitude   
    Airfoils.oper.alpha        = np.array([AoA])  
    Airfoils.oper.rho          = 1.                 # density 
    Airfoils.oper.initbl       = True               # true to initialize the boundary layer
    Airfoils.oper.viscous      = viscous_flag       # true to do viscous 
    Airfoils.oper.Re           = np.array([Re])     # viscous Reynolds number
    Airfoils.oper.Ma           = Ma                 # Mach number  
    Airfoils.oper.redowake     = False 

    # inviscid solution variables
    Airfoils.isol               = Data()
    Airfoils.isol.AIC           = []                          # aero influence coeff matrix
    Airfoils.isol.gamref        = []                          # 0,90-deg alpha vortex strengths at airfoil nodes
    Airfoils.isol.gam           = []                          # vortex strengths at airfoil nodes (for current alpha)
    Airfoils.sstag              = 0.                          # s location of stagnation point
    Airfoils.sstag_g            = np.array([0,0])             # lin of sstag w.r.t. adjacent gammas
    Airfoils.sstag_ue           = np.array([0,0])             # lin of sstag w.r.t. adjacent ue values
    Airfoils.Istag              = np.array([0,0])             # node indices before/after stagnation 
    Airfoils.isol.sgnue         = []                          # +/- 1 on upper/lower surface nodes
    Airfoils.isol.xi            = []                          # distance from the stagnation at all points
    Airfoils.isol.uewi          = []                          # inviscid edge velocity in wake
    Airfoils.isol.uewiref       = []                          # 0,90-deg alpha inviscid ue solutions on wake     
    
    Airfoils.vsol               = Data()
    Airfoils.vsol.theta         = []                           # theta = momentum thickness [Nsys]
    Airfoils.vsol.delta_star    = []                           # delta star = displacement thickness [Nsys]
    Airfoils.vsol.Is            = {}                           # 3 cell arrays of surface indices
    Airfoils.vsol.wgap          = []                           # wake gap over wake points
    Airfoils.vsol.ue_m          = []                           # linearization of ue w.r.t. mass (all nodes)
    Airfoils.vsol.sigma_m       = []                           # d(source)/d(mass) matrix
    Airfoils.vsol.ue_sigma      = []                           # d(ue)/d(source) matrix
    Airfoils.vsol.turb          = []                           # flag over nodes indicating if turbulent (1) or lam (0) 
    Airfoils.vsol.xt            = 0.                           # transition location (xi) on current surface under consideration
    Airfoils.vsol.Xt            = np.array([[0,0],[0,0]])      # transition xi/x for lower and upper surfaces 


    # global system variables
    Airfoils.glob              = Data()
    Airfoils.glob.Nsys         = 0                     # number of equations and states
    Airfoils.glob.U            = np.empty(shape=[1,0]) # primary states (th,ds,sa,ue) [4 x Nsys]
    Airfoils.glob.dU           = []                    # primary state update
    Airfoils.glob.dalpha       = 0                     # angle of attack update
    Airfoils.glob.conv         = True                  # converged flag
    Airfoils.glob.R            = []                    # residuals [3*Nsys x 1]
    Airfoils.glob.R_U          = []                    # residual Jacobian w.r.t. primary states
    Airfoils.glob.R_x          = []                    # residual Jacobian w.r.t. xi (s-values) [3*Nsys x Nsys]

    # post-processing quantities
    Airfoils.post             = Data()
    Airfoils.post.cp          = []       # cp distribution
    Airfoils.post.cpi         = []       # inviscid cp distribution
    Airfoils.post.cl          = 0        # lift coefficient
    Airfoils.post.cl_ue       = []       # linearization of cl w.r.t. ue [N, airfoil only]
    Airfoils.post.cl_alpha    = 0        # linearization of cl w.r.t. alpha
    Airfoils.post.cm          = 0        # moment coefficient
    Airfoils.post.cdpi        = 0        # near-field pressure drag coefficient
    Airfoils.post.cd          = 0        # total drag coefficient
    Airfoils.post.cdf         = 0        # skin friction drag coefficient
    Airfoils.post.cdp         = 0        # pressure drag coefficient

    # distributions
    Airfoils.post.theta       = []       # theta = momentum thickness distribution
    Airfoils.post.delta       = []       # delta  = boundary layer thickness 
    Airfoils.post.delta_star  = []       # delta* = displacement thickness distribution
    Airfoils.post.sa          = []       # amplification factor/shear lag coeff distribution
    Airfoils.post.ue          = []       # edge velocity (compressible) distribution
    Airfoils.post.uei         = []       # inviscid edge velocity (compressible) distribution
    Airfoils.post.cf          = []       # skin friction distribution
    Airfoils.post.Re_theta    = []       # Re_theta distribution
    Airfoils.post.Hk          = []       # kinematic shape parameter distribution

    Airfoils.param            = Data()
    Airfoils.param.verb       = 1     # printing verbosity level (higher -> more verbose)
    Airfoils.param.rtol       = 1e-10 # residual tolerance for Newton                                  
    Airfoils.param.niglob     = 50    # maximum number of global iterations
    Airfoils.param.doplot     = True  # true to plot results after each solution
    Airfoils.param.axplot     = []    # plotting axes (for more control of where plots go)

    # viscous parameters
    Airfoils.param.ncrit      = 9.0   # critical amplification factor    
    Airfoils.param.Cuq        = 1.0   # scales the uq term in the shear lag equation
    Airfoils.param.Dlr        = 0.9   # wall/wake dissipation length ratio
    Airfoils.param.SlagK      = 5.6   # shear lag constant

    # initial Ctau after transition
    Airfoils.param.CtauC      = 1.8   # Ctau constant
    Airfoils.param.CtauE      = 3.3   # Ctau exponent

    # G-beta locus: G = GA*sqrt(1+GB*beta) + GC/(H*Rt*sqrt(cf/2))
    Airfoils.param.GA         = 6.7   # G-beta A constant
    Airfoils.param.GB         = 0.75  # G-beta B constant
    Airfoils.param.GC         = 18.0  # G-beta C constant

    # operating conditions and thermodynamics
    Airfoils.param.Minf       = 0         # freestream Mach number
    Airfoils.param.Vinf       = 0         # freestream speed
    Airfoils.param.muinf      = 0         # freestream dynamic viscosity
    Airfoils.param.mu0        = 0         # stagnation dynamic viscosity
    Airfoils.param.rho0       = 1         # stagnation density
    Airfoils.param.H0         = 0         # stagnation enthalpy
    Airfoils.param.Tsrat      = 0.35      # Sutherland Ts/Tref
    Airfoils.param.gam        = 1.4       # ratio of specific heats
    Airfoils.param.KTb        = 1         # Karman-Tsien beta = sqrt(1-Minf^2)
    Airfoils.param.KTl        = 0         # Karman-Tsien lambda = Minf^2/(1+KTb)^2
    Airfoils.param.cps        = 0         # sonic cp

    # station information
    Airfoils.param.simi       = False # true at a similarity station
    Airfoils.param.turb       = False # true at a turbulent station
    Airfoils.param.wake       = False # true at a wake station 

    make_panels(Airfoils, npanel)   
    return Airfoils 

def initialize_results(cases,npanel,viscous_flag): 
     
    if viscous_flag:
        nwake  = int(np.ceil(npanel/10 + 10)) 
    else:
        nwake = 0
    npanel_and_wake   = npanel + nwake   # number of points on wake 

    Airfoil_results                = Data()
    Airfoil_results.AoA            = np.zeros((cases,1))  
    Airfoil_results.Re             = np.zeros((cases,1)) 
    Airfoil_results.num_foil_pts   = 0 
    Airfoil_results.num_wake_pts   = 0  
    Airfoil_results.Ma             = np.zeros((cases,1))   
    Airfoil_results.rho            = np.zeros((cases,1))   
    Airfoil_results.Vinf           = np.zeros((cases,1))   
    Airfoil_results.foil_pts       = np.zeros((cases,npanel,2)) 
    Airfoil_results.wake_pts       = np.zeros((cases,nwake,2))  
    Airfoil_results.cpi            = np.zeros((cases,npanel_and_wake)) 
    Airfoil_results.cp             = np.zeros((cases,npanel_and_wake)) 
    Airfoil_results.cpi            = np.zeros((cases,npanel_and_wake)) 
    Airfoil_results.dcp_dx         = np.zeros((cases,npanel)) 
    Airfoil_results.cl             = np.zeros((cases,1))
    Airfoil_results.cl_ue          = np.zeros((cases,npanel)) 
    Airfoil_results.normals        = np.zeros((cases,npanel_and_wake,2)) 
    Airfoil_results.cl_alpha       = np.zeros((cases,1))
    Airfoil_results.cm             = np.zeros((cases,1))
    Airfoil_results.cdpi           = np.zeros((cases,1))
    Airfoil_results.cd             = np.zeros((cases,1))
    Airfoil_results.cdf            = np.zeros((cases,1))
    Airfoil_results.cdp            = np.zeros((cases,1)) 
    Airfoil_results.converged_soln = np.zeros((cases,1)) 
    Airfoil_results.theta          = np.zeros((cases,npanel_and_wake))
    Airfoil_results.delta          = np.zeros((cases,npanel_and_wake))
    Airfoil_results.delta_star     = np.zeros((cases,npanel_and_wake))
    Airfoil_results.sa             = np.zeros((cases,npanel_and_wake))
    Airfoil_results.ue             = np.zeros((cases,npanel_and_wake)) 
    Airfoil_results.ue_inv         = np.zeros((cases,npanel_and_wake)) 
    Airfoil_results.cf             = np.zeros((cases,npanel_and_wake)) 
    Airfoil_results.Re_theta       = np.zeros((cases,npanel_and_wake)) 
    Airfoil_results.H              = np.zeros((cases,npanel_and_wake))
    Airfoil_results.vsol_Is        = {}
    
    return Airfoil_results


def set_results(Airfoils,Airfoil_results,case,viscous_flag):  

    Airfoil_results.foil_pts[case][:,0] = Airfoils.foil.x[0,:] 
    Airfoil_results.foil_pts[case][:,1] = Airfoils.foil.x[1,:] 
    Airfoil_results.AoA[case]           = Airfoils.oper.alpha[0]
    Airfoil_results.Re[case]            = Airfoils.oper.Re  
    Airfoil_results.num_foil_pts        = Airfoils.foil.N
    Airfoil_results.Ma[case]            = Airfoils.oper.Re[0]  
    Airfoil_results.rho[case]           = Airfoils.oper.rho
    Airfoil_results.Vinf[case]          = Airfoils.oper.Vinf
    Airfoil_results.cp[case]            = Airfoils.post.cp[:,0]      
    Airfoil_results.cpi[case]           = Airfoils.post.cpi[:,0] 
    Airfoil_results.cl[case]            = Airfoils.post.cl[0]      
    Airfoil_results.cl_ue[case]         = Airfoils.post.cl_ue[:,0]   
    Airfoil_results.cl_alpha[case]      = Airfoils.post.cl_alpha[0]
    Airfoil_results.cm[case]            = Airfoils.post.cm[0]      
    Airfoil_results.cdpi[case]          = Airfoils.post.cdpi[0]    
    Airfoil_results.cd[case]            = Airfoils.post.cd[0]      
    Airfoil_results.cdf[case]           = Airfoils.post.cdf[0]     
    Airfoil_results.cdp[case]           = Airfoils.post.cdp[0] 
    Airfoil_results.num_wake_pts        = Airfoils.wake.N
    Airfoil_results.converged_soln[case]= Airfoils.glob.conv
    Airfoil_results.viscous_flag        = viscous_flag
    Airfoil_results.vsol_Is[case]       = Airfoils.vsol.Is
    
    if viscous_flag:
        Airfoils.wake_pts                   = Airfoils.wake.N
        Airfoil_results.wake_pts[case][:,0] = Airfoils.wake.x[0,:]  
        Airfoil_results.wake_pts[case][:,1] = Airfoils.wake.x[1,:]   
        Airfoil_results.dcp_dx[case]        = np.gradient(Airfoils.post.cp[:-Airfoils.wake.N,0], Airfoils.foil.x[0,:])
        Airfoil_results.theta[case]         = Airfoils.post.theta   
        Airfoil_results.delta[case]         = Airfoils.post.delta[:,0]
        Airfoil_results.delta_star[case]    = Airfoils.post.delta_star   
        Airfoil_results.sa[case]            = Airfoils.post.sa      
        Airfoil_results.ue[case]            = Airfoils.post.ue      
        Airfoil_results.ue_inv[case]        = Airfoils.post.uei[:,0]     
        Airfoil_results.cf[case]            = Airfoils.post.cf[:,0]      
        Airfoil_results.Re_theta[case]      = Airfoils.post.Ret[:,0]    
        Airfoil_results.H[case]             = Airfoils.post.Hk[:,0] 
        Airfoil_results.normals[case]       = Airfoils.post.normals.T
    else: 
        Airfoil_results.dcp_dx[case]        = np.gradient(Airfoils.post.cp[:,0], Airfoils.foil.x[0,:])
    
    return Airfoils
