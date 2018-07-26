## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# weissinger_vortex_lattice.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Apr 2017, T. MacDonald
#           Oct 2017, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np 

# ----------------------------------------------------------------------
#  Weissinger Vortex Lattice
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def weissinger_vortex_lattice(conditions,settings,wing, propulsors):
    """Uses the vortex lattice method to compute the lift coefficient and induced drag component

    Assumptions:
    None

    Source:
    An Introduction to Theoretical and Computational Aerodynamics by Jack Moran

    Inputs:
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
    configuration.number_panels_spanwise    [Unitless]
    configuration.number_panels_chordwise   [Unitless]
    conditions.aerodynamics.angle_of_attack [radians]

    Outputs:
    Cl                                      [Unitless]
    Cd                                      [Unitless]

    Properties Used:
    N/A
    """ 
 

        
    orientation = wing.vertical    
    
    span        = wing.spans.projected
    root_chord  = wing.chords.root
    tip_chord   = wing.chords.tip
    sweep       = wing.sweeps.quarter_chord
    taper       = wing.taper
    twist_rc    = wing.twists.root
    twist_tc    = wing.twists.tip
    sym_para    = wing.symmetric
    Sref        = wing.areas.reference
    orientation = wing.vertical
    
    ##-----------------------------------
    ## test parameters  
    #V_inf = 65
    #V_j = 70    
    #rho = 1.2 
        
    #prop_Cl= 0.8 # propeller.sectional_lift_coefficent
    #C_T    =  1 #propeller.thrust_coeffient
    #T = 1 # thrust     
    
    #aoa  = 1 * np.pi/180  
    ##-----------------------------------

    n           = 50            # number_panels_spanwise
    # conditions
    aoa = conditions.aerodynamics.angle_of_attack
         
    # chord difference
    dchord = (root_chord-tip_chord)
    if sym_para is True :
        span = span/2
        
    deltax = span/n
    
    sin_aoa = np.sin(aoa)
    cos_aoa = np.cos(aoa)

    if orientation == False :

        # Determine if wing segments are defined  
        segment_keys = wing.Segments.keys()
        n_segments   = len(segment_keys)
        segment_vortex_index = np.zeros(n_segments)
        # If spanwise stations are setup
        if n_segments>0:
            # discretizing the wing sections into panels
            i              = np.arange(0,n)
            j              = np.arange(0,n+1)
            y_coordinates = (j)*deltax             
            segment_chord = np.zeros(n_segments)
            segment_twist = np.zeros(n_segments)
            segment_sweep = np.zeros(n_segments)
            segment_span = np.zeros(n_segments)
            segment_chord_x_offset = np.zeros(n_segments)
            section_stations = np.zeros(n_segments)
            
            # obtain chord and twist at the beginning/end of each segment
            for i_seg in xrange(n_segments):                
                segment_chord[i_seg] = wing.Segments[segment_keys[i_seg]].root_chord_percent*root_chord
                segment_twist[i_seg] = wing.Segments[segment_keys[i_seg]].twist
                segment_sweep[i_seg] = wing.Segments[segment_keys[i_seg]].sweeps.quarter_chord
                section_stations[i_seg] = wing.Segments[segment_keys[i_seg]].percent_span_location*span
                
                if i_seg == 0:
                    segment_span[i_seg] = 0.0
                    segment_chord_x_offset[i_seg] = 0.25*root_chord # weissinger uses quarter chord as reference
                else:
                    segment_span[i_seg]    = wing.Segments[segment_keys[i_seg]].percent_span_location*span - wing.Segments[segment_keys[i_seg-1]].percent_span_location*span
                    segment_chord_x_offset[i_seg]  = segment_chord_x_offset[i_seg-1] + segment_span[i_seg]*np.tan(segment_sweep[i_seg-1])
            
            # shift spanwise vortices onto section breaks 
            for i_seg in xrange(n_segments):
                idx =  (np.abs(y_coordinates-section_stations[i_seg])).argmin()
                y_coordinates[idx] = section_stations[i_seg]
            
            # define y coordinates of horseshoe vortices      
            ya = np.atleast_2d(y_coordinates[i])           
            yb = np.atleast_2d(y_coordinates[i+1])          
            deltax = y_coordinates[i+1] - y_coordinates[i]
            xa =  np.zeros(n)
            x  = np.zeros(n)
            y  =  np.zeros(n)
            twist_distribution =  np.zeros(n)
            section_length =  np.zeros(n)
            
            # define coordinates of horseshoe vortices and control points
            i_seg = 0
            for idx in xrange(n):
                twist_distribution[idx]   =  segment_twist[i_seg] + ((yb[0][idx] - deltax[idx]/2 - section_stations[i_seg]) * (segment_twist[i_seg+1] - segment_twist[i_seg])/segment_span[i_seg+1])     
                section_length[idx] =  segment_chord[i_seg] + ((yb[0][idx] - deltax[idx]/2 - section_stations[i_seg]) * (segment_chord[i_seg+1] - segment_chord[i_seg])/segment_span[i_seg+1])
                xa[idx]= segment_chord_x_offset[i_seg] + (yb[0][idx] - deltax[idx]/2 - section_stations[i_seg])*np.tan(segment_sweep[i_seg])                                                    # computer quarter chord points for each horseshoe vortex
                x[idx] = segment_chord_x_offset[i_seg] + (yb[0][idx] - deltax[idx]/2 - section_stations[i_seg])*np.tan(segment_sweep[i_seg])  + 0.5*section_length[idx]                         # computer three-quarter chord control points for each horseshoe vortex
                y[idx] = (yb[0][idx] -  deltax[idx]/2)                
                
                if y_coordinates[idx] == wing.Segments[segment_keys[i_seg+1]].percent_span_location*span: 
                    i_seg += 1                    
                if y_coordinates[idx+1] == span:
                    continue
                                  
            ya = np.atleast_2d(ya)  # y coordinate of start of horseshoe vortex on panel
            yb = np.atleast_2d(yb)  # y coordinate of end horseshoe vortex on panel
            xa = np.atleast_2d(xa)  # x coordinate of horseshoe vortex on panel
            x  = np.atleast_2d(x)   # x coordinate of control points on panel
            y  = np.atleast_2d(y)   # y coordinate of control points on panel
            
        else:   # no segments defined on wing 
            # discretizing the wing sections into panels 
            i              = np.arange(0,n)
            section_length = dchord/span*(span-(i+1)*deltax+deltax/2) + tip_chord
            twist_distribution   = twist_rc + i/float(n)*(twist_tc-twist_rc)
            
            ya = np.atleast_2d((i)*deltax)                                                  # y coordinate of start of horseshoe vortex on panel
            yb = np.atleast_2d((i+1)*deltax)                                                # y coordinate of end horseshoe vortex on panel
            xa = np.atleast_2d(((i+1)*deltax-deltax/2)*np.tan(sweep) + 0.25*section_length) # x coordinate of horseshoe vortex on panel
            x  = np.atleast_2d(((i+1)*deltax-deltax/2)*np.tan(sweep) + 0.75*section_length) # x coordinate of control points on panel
            y  = np.atleast_2d(((i+1)*deltax-deltax/2))                                     # y coordinate of control points on panel 
                                                       
        
        
        chord_distribution  = section_length 
        q_distribution      = conditions.freestream.dynamic_pressure
        
        # Check to see if there are any propellers  
        if propulsors.has_key('network'):
            propeller   =  propulsors['network'].propeller            
            propeller_status = True
        else: 
            propeller_status = False        
 
        if propeller_status : # If propellers present, find propeller location and re-vectorize wing with embedded propeller 
            if propeller.origin[0][0] < wing.origin[0] and propeller.origin[0][1] < span :
                num_prop = len(propeller.origin)          # number of propeller 
                c_b_old  = propeller.chord_distribution # chord distribution of propeller from propulsion analysis
                beta_old = propeller.twist_distribution  
                Wt_old   = conditions.propulsion.acoustic_outputs.Wt
                Wa_old   = conditions.propulsion.acoustic_outputs.Wa
                aoa_prop_old = conditions.propulsion.acoustic_outputs.aoa
                n_old    = len(propeller.chord_distribution)  # number of spanwise divisions 
                #D_p      = propeller.tip_radius*2
                R_p      = propeller.tip_radius
                          
                #y_j      = np.zeros((num_prop,3))         # [y:y+:y-] -> coordinates of propeller center; y+ (orgin + propeller radius) : y- (orgin - propeller radius)
                prop_coordinates = np.zeros((num_prop,2))  
                #aoa1_distribution = np.zeros(num_prop) # angle of attack distribution in free stream (no prop)
             
                #for i in xrange(num_prop):              
                    #y_j[:][i] = [propeller.origin[i][1] ,propeller.origin[i][1]  + R_p  ,propeller.origin[i][1]  - R_p ]
                    
                for i in xrange(num_prop):
                    prop_vec_plus  = y - (propeller.origin[i][1]  + R_p)
                    prop_vec_minus = y - (propeller.origin[i][1]  - R_p)
                    RHS_vec        = np.extract(prop_vec_plus >0 ,prop_vec_plus)         
                    LHS_vec        = np.extract(prop_vec_minus <=0 ,prop_vec_minus) 
                    prop_coordinates[i][0] = np.where(prop_vec_minus == max(LHS_vec))[0] 
                    prop_coordinates[i][1] = np.where(prop_vec_plus == min(RHS_vec))[0]
                    n  =  ( np.where(prop_vec_plus == min(RHS_vec))[0] - np.where(prop_vec_minus == max(LHS_vec))[0])/2 
                         
              
                c_b      = np.interp(np.linspace(0,R_p,n ), c_b_old, np.linspace(0,R_p,n_old))
                beta     = np.interp(np.linspace(0,R_p,n ), beta_old, np.linspace(0,R_p,n_old))            
                Wt       = np.interp(np.linspace(0,R_p,n ), Wt_old, np.linspace(0,R_p,n_old)) 
                Wa       = np.interp(np.linspace(0,R_p,n ), Wa_old, np.linspace(0,R_p,n_old))
                aoa_prop = np.interp(np.linspace(0,R_p,n ), aoa_prop_old , np.linspace(0,R_p,n_old))             
    
                # modify aoa with prop induced aoa
                
                
                # upblowing and downblowing side
                
                
                # modifiy q_distribution
                
                
            CL , CD , LT , DT = compute_forces(x,y,xa,ya,yb,twist_distribution,aoa,q_distribution,chord_distribution)            
        else:
            CL , CD , LT , DT = compute_forces(x,y,xa,ya,yb,twist_distribution,aoa,q_distribution,chord_distribution)
        
        
                  
        #----------------------------------------------------------------------------------------------------------------------
        # METHOD 1: 0th ORDER APPROXIMATION JET/PROPELLER INTERACTION
        #----------------------------------------------------------------------------------------------------------------------        
        # pre-defined terms  
        B  = propeller.tip_radius*2         # propeller span
        b  = y_coordinates                  # spanwise y discretization
        mu = V_inf/V_j                      # velocity ratio = (V/V_j)
        num_prop = len(propeller.origin)    # number of propeller              
        y_j      = np.zeros((num_prop,3))   # [y:y+:y-] -> coordinates of propeller center; y+ (orgin + propeller radius) : y- (orgin - propeller radius)
        c_j      = np.zeros((num_prop,2))   # wing chord sections at start and 
        S_wj     = np.zeros(num_prop)       # reference area of jet stream (wing chord* jet slipstream span)
        L_a1     = np.zeros(num_prop)       # Lift of full wing in free stream (no prop)
        aoa1_distribution = np.zeros(num_prop) # angle of attack distribution in free stream (no prop)
        
        for i in xrange(num_prop):              
            y_j[:][i] = [propeller.origin[i][1] ,propeller.origin[i][1]  + B/2  ,propeller.origin[i][1]  - B/2 ]
            
        # Find chord sections for jet - if spanwise stations are setup
        segment_keys = wing.Segments.keys()
        n_segments   = len(segment_keys)        
        if n_segments>0:               
            for i_seg in xrange(n_segments-1):
                c_j[i][0] =  segment_chord[i_seg] + ((y_j[i][1] - section_stations[i_seg]) * (segment_chord[i_seg+1] - segment_chord[i_seg])/segment_span[i_seg+1])
                c_j[i][1] =  segment_chord[i_seg] + ((y_j[i][2] - section_stations[i_seg]) * (segment_chord[i_seg+1] - segment_chord[i_seg])/segment_span[i_seg+1])
        # Find chord sections for jet - if spanwise stations are not setup
        else:
            c_j[i][0] =  y_j[i][1]  * (tip_chord - root_chord)/b
            c_j[i][1] =  y_j[i][2] * (tip_chord- root_chord)/b      
        
        # reference area of jets
        S_wj[:] = B*(c_j[:][0] + c_j[:][1])/2 
        
        # angle of attack distribution
        alpha_distribution = twist_distribution+aoa 
        for i in xrange(num_prop):
            prop_vec_plus  = b - y_j[i][1]
            prop_vec_minus = b - y_j[i][2]
            RHS_vec        = np.extract(prop_vec_plus >0 ,prop_vec_plus)         
            LHS_vec        = np.extract(prop_vec_minus <=0 ,prop_vec_minus) 
            L_jet_start    =   np.where(prop_vec_minus == max(LHS_vec)) 
            L_jet_end      =   np.where(prop_vec_plus == min(RHS_vec))
            L_a1 [i]           = sum(L[0][L_jet_start[0]:L_jet_end[0]]) 
            aoa1_distribution[i] = np.average(alpha_distribution[L_jet_start[0]:L_jet_end[0]])   # taking average of AoA distribution in prop jet stream
         
        CL_a1   = CL                         # lift slope of part of wing section in free stream
        CL_ajmu = mu*L_a1/(0.5*S_wj)         # lift slope of part of the wing in the jet at a velocity ration mu 
        aoa_wj1   = aoa1_distribution            # angle of attach distribution of part of the wing in the jet downwash in free stream
        aoa_wjmu  = aoa1_distribution            # angle of attach distribution of part of the wing in the jet at a velocity ration mu  (same as aoa_wj1 if propeller not inclinded)
        delta_L = sum(S_wj*(V_j**2*CL_ajmu*aoa_wjmu - V_inf**2*CL_a1*aoa_wj1)) # additional lift due to propellers 
        delta_D = sum(delta_L/np.tan(aoa_wjmu ))                             # additional drag due to propellers
        
        
        LT_prop = np.zeros(2)
        DT_prop = np.zeros(2)
        Cl_prop = np.zeros(2)
        Cd_prop = np.zeros(2)
        
        # Total Lift & Drag with Propeller
        LT_prop[0] = LT + delta_L 
        DT_prop[0] = DT + delta_D
    
        # CL and CD wtih Propeller
        Cl_prop[0] = 2*LT_prop[0] /(0.5*Sref)
        Cd_prop[0] = 2*DT_prop[0]/(0.5*Sref)        


        #----------------------------------------------------------------------------------------------------------------------
        # METHOD 2: SQUARE WAVE VELOCITY PROFILE JET/PROPELLER INTERACTION
        #----------------------------------------------------------------------------------------------------------------------   
        V_inf_distribution     = np.ones(len(y_coordinates)-1)*V_inf             # free stream distribution
        q                      = np.ones(len(y_coordinates)-1)*0.5*1.2*V_inf**2  # dynamic pressure distribution
        q_prop                 = np.ones(len(y_coordinates)-1)*0.5*1.2*V_inf**2  # dynamic pressure distribution with propeller
        for i in xrange(num_prop):              
                prop_origin    = propeller.origin[i][1]      # coordinate of horseshoe vortex
                prop_vec_plus  = b - (prop_origin - B/2)
                prop_vec_minus = b - (prop_origin + B/2) 
                RHS_vec        = np.extract(prop_vec_plus >0 ,prop_vec_plus)         
                LHS_vec        = np.extract(prop_vec_minus <=0 ,prop_vec_minus) 
                L_jet_start    =    np.where(prop_vec_plus == min(RHS_vec))
                L_jet_end      =    np.where(prop_vec_minus == max(LHS_vec)) 
                V_inf_distribution[L_jet_start[0]:L_jet_end[0]] = V_j
                q_prop[L_jet_start[0]:L_jet_end[0]]             =  0.5*1.2*V_j**2
        
        
        Lift_distribution      = q*Lft*chord_distribution       # lift distribution in free stream with no proppeller
        Drag_distribution      = q*Dg*chord_distribution        # lift distribution in free stream with no proppeller
        
        Lift_distribution_prop = q_prop*Lft*chord_distribution  # lift distribution  with no proppeller
        Drag_distribution_prop = q_prop*Dg*chord_distribution   # drag distribution  with no proppeller
   
        # Total Lift & Drag with Propeller
        LT_prop[1] = sum(Lift_distribution_prop[0]) 
        DT_prop[1] = sum(Drag_distribution_prop[0]) 
        
        # CL and CD wtih Propeller         
        Cl_prop[1] = 2*LT_prop[1] /(0.5*Sref)
        Cd_prop[1] = 2*DT_prop[1] /(0.5*Sref)
                   
                   
        #-----------------------------------------------------------
        # PLOT LIFT & DRAF DISTRIBUTION
        #-----------------------------------------------------------
        Lft = Lft.squeeze()[::1]
        wing_span          = np.array(np.linspace(0,span,n))
        chord_distribution = np.array(np.linspace(root_chord,tip_chord,n))
        CL_distribution    = 0.5*Lft/(chord_distribution)    
    
        fig = plt.figure('Semi Span Aerodynamics')
        fig.set_size_inches(10, 8)
    
        axes1 = fig.add_subplot(2,2,1)
        axes1.plot( wing_span , CL_distribution, 'bo-' )
        axes1.set_xlabel('Span (m)')
        axes1.set_ylabel(r'$C_{l}$')
        plt.ylim((0,0.05))
        axes1.grid(True)      
    
        axes2 = fig.add_subplot(2,2,2)
        axes2.plot( wing_span , V_inf_distribution, 'ro-' )
        axes2.set_xlabel('Span (m)')
        axes2.set_ylabel(r'Local Velocity $m/s$')
        plt.ylim((50,100))
        axes2.grid(True)        
    
    
        axes3 = fig.add_subplot(2,2,3)
        axes3.plot( wing_span , Lift_distribution[0], 'bo-',wing_span , Lift_distribution_prop[0], 'go-'  )
        axes3.set_xlabel('Span (m)')
        axes3.set_ylabel(r'$Spanwise Lift$')
        axes3.grid(True)        
        plt.ylim((20,50))
        plt.show()    
        
        ##----------------------------------------------------------------------------------------------------------------------
        ## METHOD 3: 1st ORDER APPROXIMATION JET/PROPELLER INTERACTION
        ##----------------------------------------------------------------------------------------------------------------------                
        ## c_b     - blade section chord
        ## r       - radial distance from propeller
        ## R_p     - propeller radius
        ## w       - (omega) angular velocity
        ## D_p     - propeller diameter
        ## prop_Cl - blade section lift coefficent
        ## beta_t  - geometric aoa at propeller tip
        ## ep      - (epsilon) blade section induced aoa
        ## ep_inf  - (epsilon infinity) blade section advance angle of attack
        
        #n      = len(propeller.chord_distribution)  # number of spanwise divisions
        #theta =  np.linspace(-np.pi/2,-np.pi/2,n) # angular stations    
        #c_b    = propeller.chord_distribution 
        #beta   = propeller.twist_distribution
        #D_p    = propeller.tip_radius*2
        #R_p    = propeller.tip_radius
        #r      = np.linspace(0,R_p,n) 
        #prop_Cl= 0.8 # propeller.sectional_lift_coefficent
        #b      = propeller.number_blades
        #w      = propeller.angular_velocity
        #ep_inf = c_b - np.tan(V_inf/(w*R_p))
        #beta_t = beta[-1] 
        
        ## finding blade section induced angle of attack
        #ep = np.zeros(n)
        #from scipy.optimize import fsolve
        #import sympy
        #for i in xrange(n):
            #c_b_val = c_b[i]
            #r_val   = r[i]
            #ep_inf_val = ep_inf[i] 
            #def f(x):
                #return (b*c_b_val*prop_Cl/(16*r_val)) - np.arccos(np.exp(- (b*(1-(2*r_val/D_p)))/(2*np.sin(beta_t))))* np.tan(x)*np.sin(ep_inf_val + x)
            #x = fsolve(f, 0.5)        
            #ep[i] = x 
        
        ## total induced velocity
        #V_i = w*r*np.sin(ep)/np.cos(ep_inf)
        
        #fig = plt.figure()
        #axes  = fig.add_subplot(1,1,1)
        #axes.plot( r, V_i, 'bo-' )
        #axes.grid(True)
        #plt.show()
        
        ## finding angle of attack d wing_spanistribution 
        #aoa_i
        
        #v_prime_axial      =   V_i*np.cos(aoa_i)
        #v_prime_tangential =   V_i*np.sin(aoa_i)
    
        #Vx = V_inf + v_prime_axial
        #Vy = V_y   + v_prime_tangential
    
        ## compute dynamic pressure 
  

        
  
        
        
        CL = 1
            
 
     
                         
        
        return        
        
 
    
# ----------------------------------------------------------------------
#   Helper Functions
# ----------------------------------------------------------------------
def whav(x1,y1,x2,y2):
    """ Helper function of vortex lattice method      
        Inputs:
            x1,x2 -x coordinates of bound vortex
            y1,y2 -y coordinates of bound vortex
        Outpus:
            Cl_comp - lift coefficient
            Cd_comp - drag  coefficient       
        Assumptions:
            if needed
    """    

    use_base    = 1 - np.isclose(x1,x2)*1.
    no_use_base = np.isclose(x1,x2)*1.
    
    whv = 1/(y1-y2)*(1+ (np.sqrt((x1-x2)**2+(y1-y2)**2)/(x1-x2)))*use_base + (1/(y1 -y2))*no_use_base
    
    return whv 


def compute_forces(x,y,xa,ya,yb,twist_distribution,aoa_distribution,q_distribution,chord_distribution):    
    RHS  = np.atleast_2d(np.sin(twist_distribution+aoa_distribution))   
    A = (whav(x,y,xa.T,ya.T)-whav(x,y,xa.T,yb.T)\
         -whav(x,y,xa.T,-ya.T)+whav(x,y,xa.T,-yb.T))*0.25/np.pi

    # Vortex strength computation by matrix inversion
    T = np.linalg.solve(A.T,RHS.T).T

    # Calculating the effective velocty         
    A_v = A*0.25/np.pi*T
    v   = np.sum(A_v,axis=1)

    Lfi = -T * (sin_aoa-v)
    Lfk =  T * cos_aoa 
    Lft = -Lfi * sin_aoa + Lfk * cos_aoa
    Dg  =  Lfi * cos_aoa + Lfk * sin_aoa

    L  = deltax * Lft
    D  = deltax * Dg

    # Lift & Drag distribution
    Lift_distribution      = q_distribution *Lft*chord_distribution        
    Drag_distribution      = q_distribution *Dg*chord_distribution       
    
    # Total Lift and Draf
    LT = sum(Lift_distribution) 
    DT = sum(Drag_distribution) 

    # CL and CD     
    CL  = 2*LT  /(0.5*Sref)
    CD = 2*DT  /(0.5*Sref)
    
    return CL, CD ,LT, DT 