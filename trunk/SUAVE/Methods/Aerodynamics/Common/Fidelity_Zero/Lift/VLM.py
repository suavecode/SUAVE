## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# VLM.py
# 
# Created:  May 2019, M. Clarke
#           Jul 2020, E. Botero
#           Sep 2020, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports 
import numpy as np 
from SUAVE.Core import Data
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wing_induced_velocity      import compute_wing_induced_velocity
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wing_induced_velocity_sup  import compute_wing_induced_velocity_sup
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.generate_wing_vortex_distribution  import generate_wing_vortex_distribution, compute_unit_normal
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_RHS_matrix                 import compute_RHS_matrix 

# ----------------------------------------------------------------------
#  Vortex Lattice
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift 
def VLM(conditions,settings,geometry,initial_timestep_offset = 0 ,wake_development_time = 0.05 ):
    """Uses the vortex lattice method to compute the lift, induced drag and moment coefficients  

    Assumptions:
    None

    Source:
    1. Aerodynamics for Engineers, Sixth Edition by John Bertin & Russel Cummings 
    Pgs. 379-397(Literature)
    
    2. Low-Speed Aerodynamics, Second Edition by Joseph katz, Allen Plotkin
    Pgs. 331-338(Literature), 579-586 (Fortran Code implementation)
    
    3. Yahyaoui, M. "Generalized Vortex Lattice Method for Predicting Characteristics of Wings
    with Flap and Aileron Deflection" , World Academy of Science, Engineering and Technology 
    International Journal of Mechanical, Aerospace, Industrial and Mechatronics Engineering 
    Vol:8 No:10, 2014
    
    4. Miranda, Luis R., Robert D. Elliot, and William M. Baker. "A generalized vortex 
    lattice method for subsonic and supersonic flow applications." (1977). (NASA CR)

    Inputs:
    geometry.
       wing.
         spans.projected                       [m]
         chords.root                           [m]
         chords.tip                            [m]
         sweeps.quarter_chord                  [radians]
         taper                                 [Unitless]
         twists.root                           [radians]
         twists.tip                            [radians]
         symmetric                             [Boolean]
         aspect_ratio                          [Unitless]
         areas.reference                       [m^2]
         vertical                              [Boolean]
         origin                                [m]
       fuselage.
        origin                                 [m]
        width                                  [m]
        heights.maximum                        [m]      
        lengths.nose                           [m]    
        lengths.tail                           [m]     
        lengths.total                          [m]     
        lengths.cabin                          [m]     
        fineness.nose                          [Unitless]
        fineness.tail                          [Unitless]
        
       settings.number_spanwise_vortices         [Unitless]
       settings.number_chordwise_vortices        [Unitless]
       settings.use_surrogate                  [Unitless]
       settings.propeller_wake_model           [Unitless]
       conditions.aerodynamics.angle_of_attack [radians]
       conditions.freestream.mach_number       [Unitless]
       
    Outputs:                                   
    CL                                         [Unitless]
    Cl                                         [Unitless]
    CDi                                        [Unitless]
    Cdi                                        [Unitless]

    Properties Used:
    N/A
    """ 
   
    # unpack settings
    n_sw       = settings.number_spanwise_vortices    
    n_cw       = settings.number_chordwise_vortices 
    pwm        = settings.propeller_wake_model
    Sref       = geometry.reference_area 
    

    # define point about which moment coefficient is computed
    if 'main_wing' in geometry.wings:
        c_bar      = geometry.wings['main_wing'].chords.mean_aerodynamic
        x_mac      = geometry.wings['main_wing'].aerodynamic_center[0] + geometry.wings['main_wing'].origin[0][0]
    else:
        c_bar = 0.
        x_mac = 0.
        for wing in geometry.wings:
            if wing.vertical == False:
                if c_bar <= wing.chords.mean_aerodynamic:
                    c_bar = wing.chords.mean_aerodynamic
                    x_mac = wing.aerodynamic_center[0] + wing.origin[0][0]
            
    x_cg       = geometry.mass_properties.center_of_gravity[0][0]
    if x_cg == None:
        x_m = x_mac 
    else:
        x_m = x_cg
    
    aoa  = conditions.aerodynamics.angle_of_attack   # angle of attack  
    mach = conditions.freestream.mach_number         # mach number
    ones = np.atleast_2d(np.ones_like(mach)) 
   
    # generate vortex distribution 
    VD   = generate_wing_vortex_distribution(geometry,settings)   
    
    # pack vortex distribution 
    geometry.vortex_distribution = VD
    
    # Build induced velocity matrix, C_mn
    C_mn, DW_mn, s, CHORD = compute_wing_induced_velocity_sup(VD,n_sw,n_cw,aoa,mach) 
     
    # Compute flow tangency conditions   
    inv_root_beta           = np.zeros_like(mach)
    inv_root_beta[mach<1]   = 1/np.sqrt(1-mach[mach<1]**2)     
    inv_root_beta[mach>1]   = 1.#1/np.sqrt(mach[mach>1]**2-1)   
    inv_root_beta           = np.atleast_2d(inv_root_beta)
    
    phi   = np.arctan((VD.ZBC - VD.ZAC)/(VD.YBC - VD.YAC))*ones          # dihedral angle 
    delta = np.arctan((VD.ZC - VD.ZCH)/((VD.XC - VD.XCH)*inv_root_beta)) # mean camber surface angle 
   
    # Build Aerodynamic Influence Coefficient Matrix
    A =   np.multiply(C_mn[:,:,:,0],np.atleast_3d(np.sin(delta)*np.cos(phi))) \
        + np.multiply(C_mn[:,:,:,1],np.atleast_3d(np.cos(delta)*np.sin(phi))) \
        - np.multiply(C_mn[:,:,:,2],np.atleast_3d(np.cos(phi)*np.cos(delta)))   # validated from book eqn 7.42  
   
    # Build the vector
    RHS  ,Vx_ind_total , Vz_ind_total , V_distribution , dt = compute_RHS_matrix(n_sw,n_cw,delta,phi,conditions,geometry,\
                                                                                 pwm,initial_timestep_offset,wake_development_time ) 
    # Compute vortex strength  
    n_cp     = VD.n_cp  
    gamma    = np.linalg.solve(A,RHS)
    gamma_3d = np.repeat(np.atleast_3d(gamma), n_cp ,axis = 2 )
    u        = np.sum(C_mn[:,:,:,0]*gamma_3d, axis = 2)  
    w        = np.sum(DW_mn[:,:,:,2]*gamma_3d, axis = 2) 
    
    ## ---------------------------------------------------------------------------------------
    ## STEP 10: Compute Pressure Coefficient
    ## ------------------ --------------------------------------------------------------------
    
    ## Calculate the vacuum and stagnation limits
    #CPSTAG = np.ones_like(u)
    #XM1    = np.ones_like(u)
    #XM2    = np.zeros_like(u)
    #XM3    = np.ones_like(u)
    #XM4    = np.ones_like(u)
    #XM5    = np.zeros_like(u)
    #CPVAC  = -142.86 * np.ones_like(u)
    B2     = np.tile((1 - mach**2),n_cp)
    
    ## Supersonic
    #CPSTAG[B2<-0.92] = ((1.2 + .2 *B2[B2<-0.92] ) **3.5 - 1.) /(.7*(1.+B2[B2<-0.92] ))
    #CPVAC[B2<-0.92]  = - 1.0 /(1.0 + B2[B2<-0.92])
    #XM1[B2<-0.92]    = 1.4286 /(1.0 + B2[B2<-0.92])
    #XM2[B2<-0.92]    = 1.0
    #XM3[B2<-0.92]    = 0.2 *(1.0 + B2[B2<-0.92])
    #XM4[B2<-0.92]    = 3.5
    #XM5[B2<-0.92]    = 1.0   
    
    # COMPUTE FREE-STREAM AND ONSET FLOW PARAMETERS. If yaw is ever added these equations would change
    COSALF = np.tile(np.cos(aoa),n_cp)
    FORAXL = COSALF
    
    #FORLAT = 0.
    #LAX    = 1.0
    FLAX   = 1.0 # Assume cosine spaced panels (LAX)
    
    #KC     =  # The chordwise panel number. All are positve numbers
    #KS     =  # The spanwise panel number
    
    RNMAX   = n_cw*1.
    #PION    = np.pi/RNMAX
    FACTOR  = FORAXL
    CHORD   = CHORD[:,:,0]
    
    # COMPUTE LOAD COEFFICIENT
    GNET = gamma*FACTOR*RNMAX/CHORD
    DCP  = 2*GNET
    CP   = DCP
    
    ## ---------------------------------------------------------------------------------------
    ## STEP 11: Compute aerodynamic coefficients 
    ## ------------------ -------------------------------------------------------------------- 
    
    # PSI is 0 for zero yaw, we also are doing zero roll and no pitch rate
    SINALFA = np.tile(np.sin(aoa),n_cp)
    COSALF  = np.tile(np.cos(aoa),n_cp)
    SINPSI  = 0.
    COPSI   = 1.
    COSIN   = 0.
    COSINP  = 0.
    COSCOS  = COSALF *COPSI
    PITCH   = 0.
    ROLL    = 0.
    YAW     = 0.
    
    # Work panel by panel
    SURF = np.array(VD.wing_areas)
    SREF = Sref

    
    # Leading edge sweep and trailing edge sweep. VORLAX does it panel by panel. This will be spanwise.

    YBH = VD.YBH*ones
    XA1 = VD.XA1*ones
    XB1 = VD.XB1*ones
    YA1 = VD.YA1*ones
    YB1 = VD.YB1*ones
    
    
    # This is only valid for calculating LE sweeps
    boolean = YBH<0. 
    XA1[boolean], XB1[boolean] = XB1[boolean], XA1[boolean]
        
    TLE = (YB1-YA1)/(XB1-XA1)
    T2  = TLE**2
    STB = np.zeros_like(u)
    STB[B2<T2] = np.sqrt(T2[B2<T2]-B2[B2<T2])
    
    # Panel Dihedral Angle, using AH and BH location
    YAH = VD.YAH*ones
    ZAH = VD.ZAH*ones
    ZBH = VD.ZBH*ones
    
    YAH[boolean], YBH[boolean] = YBH[boolean], YAH[boolean]
    
    D   = np.sqrt((YAH-YBH)**2+(ZAH-ZBH)**2)
    
    SID = (ZBH-ZAH)/D
    COD = (YBH-YAH)/D
    
    
    # Now on to each strip
    PION = (np.pi *(1.0 - FLAX) + 2.0 *FLAX) /RNMAX
    ADC  = 0.5*PION
    JTS  = 0.
    
    # XLE = LOCATION OF FIRST VORTEX MIDPOINT IN FRACTION OF CHORD.
    XLE =.5 *(1.0 -np.cos (.5 *PION)) *(1.0 - FLAX) + 0.125 *PION *FLAX
    
    RJTS = JTS
    GAF  = 0.5 + 0.5 *RJTS **2
    
    # SINF REFERENCES THE LOAD CONTRIBUTION OF IRT-VORTEX TO THE
    # STRIP NOMINAL AREA, I.E., AREA OF STRIP ASSUMING CONSTANT
    # (CHORDWISE) HORSESHOE SPAN.    
    
    VST = np.abs((VD.YBH - VD.YAH)*ones)
    VSS = np.abs((VD.YBH - VD.YAH)*ones)
    
    SINF = ADC * DCP *VST /VSS
    
    # Split into chordwise strengths and sum into strips
    CNC = np.sum(np.array(np.array_split(SINF,n_cw,axis=1)),axis=2).T
    
    # COMPUTE SLOPE (TX) WITH RESPECT TO X-AXIS AT LOAD POINTS BY INTER
    # POLATING BETWEEN CONTROL POINTS AND TAKING INTO ACCOUNT THE LOCAL
    # INCIDENCE.    
    n_w  = VD.n_w    
    RK   = np.tile(np.linspace(1,n_cw,n_cw),n_sw*n_w)*ones
    IRT  = np.tile(np.linspace(1,n_sw,n_sw),n_cw*n_w)*ones
    XX   = .5 *(1. - np.cos ((RK - .5) *PION))
    XX   = XX *(1.0 - FLAX) + (RK - .75) *PION /2.0 *FLAX
    K    = 1*RK
    KX   = 1*K
    IRTX = 1*IRT
    KX[K>1]   = K[K>1]-1
    IRTX[K>1] = IRT[[K>1]] - 1
    
    # The exact results of IRTX will not match VORLAX because of differencing indexing to python
    
    RKX = KX
    X1  = .5 *(1. - np.cos (RKX *PION))
    X1  = X1 *(1.0 - FLAX) + (RKX - .25) *PION /2.0 *FLAX
    X2  = .5 *(1. - np.cos((RKX + 1.) *PION))
    X2  = X2 *(1.0 - FLAX) + (RKX + .75) *PION /2.0 *FLAX
    
    # CHECK ME LATER WITH CAMBER!!!
    XA2 = VD.XA2*ones
    XB2 = VD.XB2*ones
    ZA1 = VD.ZA1*ones
    ZB1 = VD.ZB1*ones       
    ZA2 = VD.ZA2*ones
    ZB2 = VD.ZB2*ones
    
    X1c  = (XA1+XB1)/2
    X2c  = (XA2+XB2)/2
    Z1c  = (ZA1+ZB1)/2
    Z2c  = (ZA2+ZB2)/2
    
    SLOPE = (Z2c-Z1c)/(X2c-X1c)
    
    ####################
    #### NOT FINISHED
    ####################
    
    F1 = SLOPE*1.
    #F1[K>1] = SLOPE
    F2 = SLOPE*1
    
    ZETA = 0.
    
    ####################
    #### NOT FINISHED 
    ####################
    
    TANX = (XX-X2)/(X1-X2)*F1 +(XX-X1)/(X2-X1)*F2
    TX   = TANX - ZETA
    CAXL = -SINF*TX/(1.0+TX**2) # These are the axial forces on each panel
    
    # Leading edge suction is skipped for linear chord spacing
    CLE = np.zeros_like(u)
    
    XX = XLE
    
    
    ### SPC NEEDS TO BE SET!!
    
    CLE  = CLE + 0.5*DCP *np.sqrt(XX)*FLAX
    CSUC =  0.5*np.pi*np.abs(SPC)*(CLE**2)*STB

    
    # Do a wing by wing summation
    
    
    
    

    
     
    
    
    
    
    
     
    ## ---------------------------------------------------------------------------------------
    ## STEP 10: Compute aerodynamic coefficients 
    ## ------------------ -------------------------------------------------------------------- 
    #n_w        = VD.n_w
    #CS         = VD.CS*ones
    #CS_w       = np.array(np.array_split(CS,n_w,axis=1)) 
    #wing_areas = np.array(VD.wing_areas)
    #X_M        = np.ones(n_cp)*x_m  *ones
    #CL_wing    = np.zeros(n_w)
    #CDi_wing   = np.zeros(n_w) 
    #Del_Y      = np.abs(VD.YB1 - VD.YA1)*ones  
    
    ## Use split to divide u, w, gamma, and Del_y into more arrays
    #u_n_w        = np.array(np.array_split(u,n_w,axis=1))  
    #w_ind_n_w    = np.array(np.array_split(w_ind,n_w,axis=1)) 
    #w_ind_n_w_sw = np.array(np.array_split(w_ind,n_w*n_sw,axis=1)) 
    #gamma_n_w    = np.array(np.array_split(gamma,n_w,axis=1))
    #gamma_n_w_sw = np.array(np.array_split(gamma,n_w*n_sw,axis=1))
    #Del_Y_n_w    = np.array(np.array_split(Del_Y,n_w,axis=1))
    #Del_Y_n_w_sw = np.array(np.array_split(Del_Y,n_w*n_sw,axis=1)) 
    
    ## --------------------------------------------------------------------------------------------------------
    ## LIFT                                                                          
    ## --------------------------------------------------------------------------------------------------------    
    ## lift coefficients on each wing   
    #L_wing            = np.sum(np.multiply(u_n_w+1,(gamma_n_w*Del_Y_n_w)),axis=2).T
    #D_wing            = np.sum(np.multiply(w_ind_n_w,(gamma_n_w*Del_Y_n_w)),axis=2).T
    #CL_wing           = L_wing/(0.5*wing_areas)
    
    ## Calculate spanwise lift 
    #spanwise_Del_y    = Del_Y_n_w_sw[:,:,0]
    #spanwise_Del_y_w  = np.array(np.array_split(Del_Y_n_w_sw[:,:,0].T,n_w,axis = 1))
    
    #cl_y              = (2*(np.sum(gamma_n_w_sw,axis=2)*spanwise_Del_y).T)/CS
    #cl_y_w            = np.array(np.array_split(cl_y ,n_w,axis=1)) 
    
    ## total lift and lift coefficient
    #L                 = np.atleast_2d(np.sum(np.multiply((1+u),gamma*Del_Y),axis=1)).T 
    #D                 = np.atleast_2d(np.sum(np.multiply(w_ind,Del_Y),axis=1)).T 
    #CL                = L/(0.5*Sref)   # validated form page 402-404, aerodynamics for engineers
    
    ### --------------------------------------------------------------------------------------------------------
    ### DRAG                                                                          
    ### --------------------------------------------------------------------------------------------------------         
    ### drag coefficients on each wing   
    ##w_ind_sw_w        = np.array(np.array_split(np.sum(w_ind_n_w_sw,axis = 2).T ,n_w,axis = 1))
    ##Di_wing           = np.sum(w_ind_sw_w*spanwise_Del_y_w*cl_y_w*CS_w,axis = 2) 
    ##CDi_wing          = Di_wing.T/(wing_areas)  
    
    ### total drag and drag coefficient 
    ##spanwise_w_ind    = np.sum(w_ind_n_w_sw,axis=2).T    
    ##D                 = np.sum(spanwise_w_ind*spanwise_Del_y.T*cl_y*CS,axis = 1) 
    ##cdi_y             = spanwise_w_ind*spanwise_Del_y.T*cl_y*CS
    ##CDi               = np.atleast_2d(D/(Sref)).T
    
    #CDC   = 0.
    #CDC   = CHORD*CDC
    #ES    = 2*s
    #DRAG  = CDC*ES
    #AX    = 1/Sref
    #CDTOT = np.sum(DRAG)*AX 
    
    ## --------------------------------------------------------------------------------------------------------
    ## PRESSURE                                                                      
    ## --------------------------------------------------------------------------------------------------------          
    #L_ij              = np.multiply((1+u),gamma*Del_Y) 
    #CP                = 2*L_ij/VD.panel_areas  
    
    ## Check the CL values
    #normal_vec = compute_unit_normal(VD)
    #Z_vec      = normal_vec[:,2]
    
    #CL_check = 2*np.sum(L_ij*Z_vec,axis=1)/Sref
    
    
    ## --------------------------------------------------------------------------------------------------------
    ## MOMENT                                                                        
    ## --------------------------------------------------------------------------------------------------------             
    #CM                = np.atleast_2d(np.sum(np.multiply((X_M - VD.XCH*ones),Del_Y*gamma),axis=1)/(Sref*c_bar)).T     
    
    #Velocity_Profile = Data()
    #Velocity_Profile.Vx_ind   = Vx_ind_total
    #Velocity_Profile.Vz_ind   = Vz_ind_total
    #Velocity_Profile.V        = V_distribution 
    #Velocity_Profile.dt       = dt 
    
    return CL, CDi, CM, CL_wing, CDi_wing, cl_y , cdi_y , CP ,Velocity_Profile




##  VX, VY, VZ ARE THE FLOW ONSET VELOCITY COMPONENTS AT THE LEADING
##  EDGE (STRIP MIDPOINT). VX, VY, VZ AND THE ROTATION RATES ARE
##  REFERENCED TO THE FREE STREAM VELOCITY. 

## Rotation terms are removed
#VX = COSCOS
#VY = COSINP
#VZ = SINALF 


## CCNTL AND SCNTL ARE DIRECTION COSINE PARAMETERS OF TANGENT TO
## CAMBERLINE AT LEADING EDGE.    
#LE_slopes = SLOPE[:,0:n_cw]
#SLE   =  np.tile(LE_slopes,n_cw)# LE edge slope
#CCNTL = 1. /np.sqrt (1.0 + SLE **2)
#SCNTL = SLE *CCNTL    


## EFFINC = COMPONENT OF ONSET FLOW ALONG NORMAL TO CAMBERLINE AT
##          LEADING EDGE.    

#EFFINC = VX *SCNTL + VY *CCNTL *SID - VZ *CCNTL *COD
#CLE = CLE-EFFINC
