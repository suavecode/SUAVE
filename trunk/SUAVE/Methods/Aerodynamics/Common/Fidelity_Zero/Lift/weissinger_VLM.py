## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# weissinger_vortex_lattice.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Apr 2017, T. MacDonald
#           Oct 2017, E. Botero
#           Jun 2018, M. Clarke
#
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
#
# package imports
import numpy as np 

# ----------------------------------------------------------------------
#  Weissinger Vortex Lattice
# ----------------------------------------------------------------------
## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def weissinger_VLM(conditions,settings,wing,propulsors):
    """Uses the vortex lattice method to compute the lift coefficient and induced drag component

    Assumptions:
    None

    Source:
    An Introduction to Theoretical and Computational Aerodynamics by Jack Moran
    Pgs. 124-135(Literature), 151-153 (Fortran Code implementation)

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

    # unpack vehicle
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

    # unpack conditions
    aoa         = conditions.aerodynamics.angle_of_attack
    mach        = conditions.freestream.mach_number 
    
    if settings.use_surrogate:   
        rho    = np.zeros_like(aoa)
        V_inf  = np.zeros_like(aoa)        
        
    else:
        rho    = conditions.freestream.density                
        V_inf  = conditions.freestream.velocity
        
    m          = len(aoa)
    n          = settings.number_panels_spanwise       
    #-------------------------------------------------------------------------------------------------------
    # WEISSINGER VORTEX LATTICE METHOD   
    #-------------------------------------------------------------------------------------------------------
    if orientation == False :
        if 'propulsor' in propulsors:
            prop =  propulsors['propulsor'].propeller            
            propeller_status = True
            n = 50
        else: 
            propeller_status = False  

        # chord difference
        dchord = (root_chord-tip_chord)
        if sym_para is True :
            span = span/2            
        delta_y = span/n 
        
        Cl               = np.zeros((m,n))
        Cd               = np.zeros((m,n))   
        V_distri   = np.ones((m,n))*V_inf      
        aoa_distri = np.ones((m,n))*aoa
        
        # Determine if wing segments are defined  
        n_segments           = len(wing.Segments.keys())
        segment_vortex_index = np.zeros(n_segments)
        # If spanwise stations are setup
        if n_segments>0:
            # discretizing the wing sections into panels
            i                      = np.arange(0,n)
            j                      = np.arange(0,n+1)
            y_coordinates          = (j)*delta_y             
            segment_chord          = np.zeros(n_segments)
            segment_twist          = np.zeros(n_segments)
            segment_sweep          = np.zeros(n_segments)
            segment_dihedral       = np.zeros(n_segments)
            segment_span           = np.zeros(n_segments)
            segment_chord_x_offset = np.zeros(n_segments)
            segment_chord_z_offset = np.zeros(n_segments)
            section_stations       = np.zeros(n_segments)
            
            # obtain chord and twist at the beginning/end of each segment    
            S_wing = 0.0
            for i_seg in range(n_segments):                
                segment_chord[i_seg]    = wing.Segments[i_seg].root_chord_percent*root_chord
                segment_twist[i_seg]    = wing.Segments[i_seg].twist
                segment_sweep[i_seg]    = wing.Segments[i_seg].sweeps.quarter_chord
                segment_dihedral[i_seg] = wing.Segments[i_seg].dihedral_outboard
                section_stations[i_seg] = wing.Segments[i_seg].percent_span_location*span

                if i_seg == 0:
                    segment_span[i_seg]           = 0.0
                    segment_chord_x_offset[i_seg] = 0.25*root_chord # weissinger uses quarter chord as reference
                    segment_chord_z_offset[i_seg] = wing.origin[1]
                else:
                    segment_span[i_seg]           = wing.Segments[i_seg].percent_span_location*span - wing.Segments[i_seg-1].percent_span_location*span
                    segment_chord_x_offset[i_seg] = segment_chord_x_offset[i_seg-1] + segment_span[i_seg]*np.tan(segment_sweep[i_seg-1])
                    segment_chord_z_offset[i_seg] = segment_chord_z_offset[i_seg-1] + segment_span[i_seg]*np.tan(segment_dihedral[i_seg-1])
                    Sref_seg                      = segment_span[i_seg] *( segment_chord[i_seg-1] + segment_chord[i_seg])*0.5
                    S_wing                       += Sref_seg 
                    
            # shift spanwise vortices onto section breaks 
            for i_seg in range(n_segments):
                idx                =  (np.abs(y_coordinates-section_stations[i_seg])).argmin()
                y_coordinates[idx] = section_stations[i_seg]

            # define yc coordinates of horseshoe vortices      
            yah          = np.atleast_2d(y_coordinates[i])           
            ybh          = np.atleast_2d(y_coordinates[i+1])  
            yc           = np.zeros(n)
            delta_y      = y_coordinates[i+1] - y_coordinates[i]
            xah          = np.zeros(n)
            xbh          = np.zeros(n)
            xc           = np.zeros(n)            
            za           = np.zeros(n)
            zb           = np.zeros(n)
            zc           = np.zeros(n)
            twist_distri = np.zeros(n)
            chord_distri = np.zeros(n)

            # define coordinates of horseshoe vortices and control points
            i_seg = 0
            for idx in range(n):
                twist_distri[idx] = segment_twist[i_seg] + ((ybh[0][idx] - delta_y[idx]/2 - section_stations[i_seg]) * (segment_twist[i_seg+1] - segment_twist[i_seg])/segment_span[i_seg+1])     
                chord_distri[idx] = segment_chord[i_seg] + ((ybh[0][idx] - delta_y[idx]/2 - section_stations[i_seg]) * (segment_chord[i_seg+1] - segment_chord[i_seg])/segment_span[i_seg+1])
                xah[idx]           = segment_chord_x_offset[i_seg] + (yah[0][idx] - section_stations[i_seg])*np.tan(segment_sweep[i_seg])                                   
                xbh[idx]           = segment_chord_x_offset[i_seg] + (ybh[0][idx] - section_stations[i_seg])*np.tan(segment_sweep[i_seg])  
                xc[idx]            = segment_chord_x_offset[i_seg] + (ybh[0][idx] - delta_y[idx]/2 - section_stations[i_seg])*np.tan(segment_sweep[i_seg])  + 0.5*chord_distri[idx]                 
                za[idx]           = segment_chord_z_offset[i_seg] + (yah[0][idx] - section_stations[i_seg])*np.tan(segment_dihedral[i_seg])                                   
                zb[idx]           = segment_chord_z_offset[i_seg] + (ybh[0][idx] - section_stations[i_seg])*np.tan(segment_dihedral[i_seg])  
                zc[idx]            = segment_chord_z_offset[i_seg] + (ybh[0][idx] - delta_y[idx]/2 - section_stations[i_seg])*np.tan(segment_dihedral[i_seg])                 
                yc[idx]            = (ybh[0][idx] - delta_y[idx]/2)                
                
                if y_coordinates[idx+1] == wing.Segments[i_seg+1].percent_span_location*span: 
                    i_seg += 1                    

            yah = np.atleast_2d(yah)  # yc coordinate of start of horseshoe vortex on panel
            ybh = np.atleast_2d(ybh)  # yc coordinate of end horseshoe vortex on panel
            xah = np.atleast_2d(xah)  # xc coordinate of beginning of horseshoe vortex on panel
            xbh = np.atleast_2d(xah)  # xc coordinate of end of horseshoe vortex on panel
            xc  = np.atleast_2d(xc)   # xc coordinate of control points on panel
            yc  = np.atleast_2d(yc)   # yc coordinate of control points on panel

        else:   # no segments defined on wing 
            # discretizing the wing sections into panels 
            i            = np.arange(0,n)
            chord_distri = dchord/span*(span-(i+1)*delta_y+delta_y/2) + tip_chord
            twist_distri = twist_rc + i/float(n)*(twist_tc-twist_rc)
            S_wing       = span*(root_chord + tip_chord)*0.5
            yah           = np.atleast_3d((i)*delta_y)                                                 # yc coordinate of start of horseshoe vortex on panel
            ybh           = np.atleast_3d((i+1)*delta_y)                                               # yc coordinate of end horseshoe vortex on panel
            xah           = np.atleast_3d(((i)*delta_y)*np.tan(sweep) + 0.25*chord_distri)             # xc coordinate of beginning of horseshoe vortex on panel
            xbh           = np.atleast_3d(((i+1)*delta_y)*np.tan(sweep) + 0.25*chord_distri)           # xc coordinate of end of horseshoe vortex on panel
            xc            = np.atleast_3d(((i+1)*delta_y-delta_y/2)*np.tan(sweep) + 0.75*chord_distri) # xc coordinate of control points on panel
            yc            = np.atleast_3d(((i+1)*delta_y-delta_y/2))                                   # yc coordinate of control points on panel 
               
        #-------------------------------------------------------------------------------------------------------
        # PROPELLER SLIPSTREAM MODEL
        #-------------------------------------------------------------------------------------------------------
        '''
        SOURCE: Aerodynamic Modelling of the Wing-Propeller Interaction
        Summary: This method uses the blade element momentum solution to modify the local angle of 
        attach and axial velocity incident on the wing
        The code below assumes that if the propeller is interacting with the wing, the wing is always situated at the 
        center of the propeller 
        '''
        if 'propulsor' in propulsors:
            prop =  propulsors['propulsor'].propeller            
            propeller_status = True
        else: 
            propeller_status = False
            
        # If propellers present, find propeller location and re-vectorize wing with embedded propeller               
        if propeller_status : 
            prop_slipstream_V_contribution = 0              
            num_prop      = len(prop.origin)    
            # loop through propellers on aircraft to get combined effect of slipstreams
            for i in range(num_prop): 
                # optain propeller and slipstream properties
                D_p       = prop.tip_radius*2
                R_p       = prop.tip_radius                     
                            
                r_nacelle = prop.hub_radius 
                vt_old    = np.concatenate((prop.outputs.vt , - prop.outputs.vt[::-1]), axis=1)  # induced tangential velocity at propeller disc using propeller discretization
                va_old    = np.concatenate((-prop.outputs.va, - prop.outputs.va[::-1]), axis=1) # induced axial velocity at propeller disc  using propeller discretization
                n_old     = len(prop.chord_distri)                                                                          # spanwise discretization of propeller
                r_old     = np.linspace(prop.hub_radius,R_p,n_old) 
                d_old     = np.concatenate((-r_old[::-1], r_old) ,  axis=0)   
                
                # compute slipstream development factor from propeller disc to control point on wing
                s                 =  prop.origin[i][0] - wing.origin[0]      
                Kd                = 1 + s/(np.sqrt(s**2 + R_p**2))  
                Vx                = np.tile(np.atleast_2d(prop.outputs.velocity[:,0]).T, (1, n_old))
                r_prime           = np.zeros((m,n_old))                
                r_prime[:,0]      = prop.hub_radius
                Kv                = (2*Vx[:,1:] + prop.outputs.va[:,1:] + prop.outputs.va[:,:-1]) /(2*Vx[:,1:] + Kd*(prop.outputs.va[:,1:] +  prop.outputs.va[:,:-1]))
                r_ratio_vals      =  r_old[1:]**2 - r_old[:-1]**2 
                r_ratio           = np.ones((m,n_old-1))*(r_old[1:]**2 - r_old[:-1]**2 )
                r_prime[:,1:]     =  np.sqrt(r_prime[:,:-1]**2 + (r_ratio)*Kv)   
                r_div_r_prime_val = r_old /r_prime                    
                r_div_r_prime_old =  np.concatenate((r_div_r_prime_val[:,::-1], r_div_r_prime_val), axis=1)   
  
                # determine if slipstream wake interacts with wing in the z direction
                if (prop.origin[i][2] + r_prime[0,-1]) > wing.origin[2]  and  (prop.origin[i][2] - r_prime[0,-1]) < wing.origin[2]:
                    
                    # determine yc location of propeller on wing                  
                    prop_vec_minus = yc - (prop.origin[i][1] - R_p*np.sqrt(1 - (abs(prop.origin[i][2] - wing.origin[2])/R_p)))               
                    LHS_vec        = np.extract(prop_vec_minus <=0 ,prop_vec_minus)                 
                    
                    # determine if slipstream wake interacts with wing in the yc direction
                    if (prop.origin[i][1] + R_p) < span: 
                        prop_vec_plus  = yc - (prop.origin[i][1] + R_p*np.sqrt(1 - (abs(prop.origin[i][2] - wing.origin[2])/R_p)))
                        RHS_vec        = np.extract(prop_vec_plus >0 ,prop_vec_plus)   
                        end_val        = np.where(prop_vec_plus == min(RHS_vec))[1][0] +1 
                        discre         = (np.where(prop_vec_plus == min(RHS_vec))[1] - np.where(prop_vec_minus == max(LHS_vec))[1]) + 1 
                           
                    else: 
                        end_val       = len(yc[0])
                        discre        = (end_val  - np.where(prop_vec_minus == max(LHS_vec))[1])                      
                        y_prop        = d_old + span
                        cut_off       = (y_prop - span)[( y_prop - span) <= 0]
                        cut_off       = np.argmax(cut_off)
                        vt_old        = vt_old[:,:cut_off]
                        va_old        = va_old[:,:cut_off]
                        r_old         =  r_old[:cut_off] 
                        d_old         =  d_old[:cut_off] 
                        r_div_r_prime_old = r_div_r_prime_old[:,:cut_off]
                    
                    # changes the discretization on propeller diameter to match the discretization on the wing   
                    D_old         = np.ones((m,1)) *d_old 
                    d             = np.interp(np.linspace(-R_p,max(d_old),discre) , d_old, d_old) 
                    vt            = np.zeros((m,discre[0]))           
                    va            = np.zeros((m,discre[0]))            
                    r_div_r_prime = np.zeros((m,discre[0]))  
                    for k in range(m):                        
                        vt[k,:]             = np.interp(np.linspace(-R_p,max(d_old),discre[0]) , d_old, vt_old[k,:])            # induced tangential velocity at propeller disc using wing discretization
                        va[k,:]             = np.interp(np.linspace(-R_p,max(d_old),discre[0]) , d_old, va_old[k,:])            # induced axial velocity at propeller disc using wing discretization            
                        r_div_r_prime[k,:]  = np.interp(np.linspace(-R_p,max(d_old),discre[0]) , d_old, r_div_r_prime_old[k,:])                      
                    
                    # adjust axial and tangential components if propeller is off centered 
                    va_prime       = Kd*va*np.sqrt(1 - (abs(prop.origin[i][2] - wing.origin[2])/R_p))
                    vt_prime       = 2*vt*r_div_r_prime*np.sqrt(1 - (abs(prop.origin[i][2] - wing.origin[2])/R_p))
                    
                    # adjust for clockwise/counter clockwise rotation
                    if prop.rotation != None:
                        vt_prime = vt_prime*prop.rotation[i]
                    
                    # compute new components of freestream
                    Vx             = V_inf*np.cos(aoa) - va_prime   
                    Vy             = V_inf*np.sin(aoa) - vt_prime   
                    modified_V_inf = np.sqrt(Vx**2 + Vy**2 )                    
                    modified_aoa   = np.arctan(Vy/Vx)          
                    
                    # modifiy air speed distribution being propeller 
                    start_val                       = np.where(prop_vec_minus == max(LHS_vec))[1][0]  
                    V_distri[:,start_val:end_val]   = modified_V_inf 
                    aoa_distri[:,start_val:end_val] = modified_aoa
        
        ones = np.atleast_3d(np.ones_like(aoa))
        VD = Data(
            X  = np.atleast_3D(xc *ones),
            Y  = np.atleast_3D(yc *ones),
            Z  = np.atleast_3D(zc *ones),
            XA = np.atleast_3D(xah*ones),
            YA = np.atleast_3D(yah*ones),
            ZA = np.atleast_3D(za*ones),
            XB = np.atleast_3D(xbh*ones),
            YB = np.atleast_3D(yah*ones),
            ZB = np.atleast_3D(zb*ones)
            )
        
        A_inf                        = compute_influence_coefficient_matrix(VD,mach)
        CL , CDi , cl_y, cdi_y ,cp_y = compute_forces(A_inf,delta_y,twist_distri,aoa_distri,mach,chord_distri, V_distri,Sref)            
    
    else: 
        CL   = np.zeros((m,1))
        CDi  = np.zeros((m,1))
        cl_y,= np.zeros((m,n))
        cd_y = np.zeros((m,n))
        cp_y = np.zeros((m,n))
    
    return   CL , CDi , cl_y, cdi_y, cp_y


def compute_influence_coefficient_matrix(VD,mach):

    return A_inf    


def compute_forces(A_inf,delta_y,twist_distri,aoa_distri,mach,chord_distri,V_distri, Sref):    
    sin_aoa = np.sin(aoa_distri)
    cos_aoa = np.cos(aoa_distri)

    RHS  = np.atleast_2d(np.sin(twist_distri+aoa_distri))   
    
    A = (whav(xc,yc,xah.T,yah.T,mach)-whav(xc,yc,xah.T,ybh.T,mach)\
        -whav(xc,yc,xah.T,-yah.T,mach)+whav(xc,yc,xah.T,-ybh.T,mach))*0.25/np.pi

    # Vortex strength computation by matrix inversion
    gamma = np.linalg.solve(A_inf.T,RHS.T).T

    # Calculating the effective velocty         
    A_v   = (A_inf.T*0.25/np.pi*gamma.T).T
    w     = np.sum(A_v,axis=1)

    Lfi   =  gamma * (w-sin_aoa) * 2
    Lfk   =  gamma * cos_aoa * 2
    Lft   = -Lfi * sin_aoa + Lfk * cos_aoa
    Dg    =  Lfi * cos_aoa + Lfk * sin_aoa

    L     = delta_y * Lft
    D     = delta_y * Dg
    
    cl_y  = Lft/(chord_distri) 
    cdi_y = Dg/(chord_distri) 
    
    # Total lift
    LT    = np.atleast_2d(np.sum(L,axis=1)).T
    DT    = np.atleast_2d(np.sum(D,axis=1)).T

    CL    = 2*LT/(Sref)
    CDi   = 2*np.pi*DT/(Sref) 
    
    # pressure coefficient
    U_tot = np.sqrt(1 + w*w)
    cp_y  = 1 - (U_tot)*(U_tot)    
    
    return  CL ,  CDi , cl_y , cdi_y , cp_y

# ----------------------------------------------------------------------
#   Helper Functions
# ----------------------------------------------------------------------
def whav(x1,y1,x2,y2,M):
    """ Helper function of vortex lattice method      
        Inputs:
            x1,x2 -xc coordinates of bound vortex
            y1,y2 -yc coordinates of bound vortex
        Outpus:
            Cl_comp - lift coefficient
            Cd_comp - drag  coefficient       
        Assumptions:
            if needed
    """    
    
    beta = 1/np.sqrt(1-M**2)
    
    use_base    = 1 - np.isclose(x1,x2)*1.
    no_use_base = np.isclose(x1,x2)*1.
    
    y1_y2 = y1-y2
    x1_x2 = x1-x2
    
    y1_y2[y1_y2==0] = 1E-32 
    x1_x2[x1_x2==0] = 1E-32
    
    whv = 1/(y1_y2)*(1+ (np.sqrt((x1_x2)**2+(y1_y2)**2)/(x1_x2)))*use_base + (1/(y1_y2))*no_use_base

    return whv 