## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# weissinger_vortex_lattice.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Apr 2017, T. MacDonald
#           Oct 2017, E. Botero
#           Jun 2018, M. Clarke


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np 
import pylab as plt
import matplotlib
# ----------------------------------------------------------------------
#  Weissinger Vortex Lattice
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def weissinger_vortex_lattice(conditions,settings,wing,propulsors,index):
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
    n           = 20            # number_panels_spanwise    
    
   #num_var     = len(conditions.freestream.density)    
    
    # conditions
    LT = 0.0 
    DT = 0.0 
    CL = 0.0 
    CD = 0.0 
    CL_distribution = np.zeros((1,n))
    CD_distribution= np.zeros((1,n))    

    # chord difference
    dchord = (root_chord-tip_chord)
    if sym_para is True :
        span = span/2            
    deltax = span/n
 
    #-------------------------------------------------------------------------------------------------------
    # WEISSINGER VORTEX LATTICE METHOD   
    #-------------------------------------------------------------------------------------------------------
    if orientation == False :

        # Determine if wing segments are defined  
        n_segments           = len(wing.Segments.keys())
        segment_vortex_index = np.zeros(n_segments)
        # If spanwise stations are setup
        if n_segments>0:
            # discretizing the wing sections into panels
            i             = np.arange(0,n)
            j             = np.arange(0,n+1)
            y_coordinates = (j)*deltax             
            segment_chord = np.zeros(n_segments)
            segment_twist = np.zeros(n_segments)
            segment_sweep = np.zeros(n_segments)
            segment_span  = np.zeros(n_segments)
            segment_chord_x_offset = np.zeros(n_segments)
            section_stations       = np.zeros(n_segments)
            
            # obtain chord and twist at the beginning/end of each segment    
            for i_seg in range(n_segments):                
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
            for i_seg in range(n_segments):
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
            chord_distribution =  np.zeros(n)

            # define coordinates of horseshoe vortices and control points
            i_seg = 0
            for idx in range(n):
                twist_distribution[idx]   =  segment_twist[i_seg] + ((yb[0][idx] - deltax[idx]/2 - section_stations[i_seg]) * (segment_twist[i_seg+1] - segment_twist[i_seg])/segment_span[i_seg+1])     
                chord_distribution[idx] =  segment_chord[i_seg] + ((yb[0][idx] - deltax[idx]/2 - section_stations[i_seg]) * (segment_chord[i_seg+1] - segment_chord[i_seg])/segment_span[i_seg+1])
                xa[idx]= segment_chord_x_offset[i_seg] + (yb[0][idx] - deltax[idx]/2 - section_stations[i_seg])*np.tan(segment_sweep[i_seg])                                                    # computer quarter chord points for each horseshoe vortex
                x[idx] = segment_chord_x_offset[i_seg] + (yb[0][idx] - deltax[idx]/2 - section_stations[i_seg])*np.tan(segment_sweep[i_seg])  + 0.5*chord_distribution[idx]                         # computer three-quarter chord control points for each horseshoe vortex
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
            chord_distribution = dchord/span*(span-(i+1)*deltax+deltax/2) + tip_chord
            twist_distribution   = twist_rc + i/float(n)*(twist_tc-twist_rc)

            ya = np.atleast_2d((i)*deltax)                                                      # y coordinate of start of horseshoe vortex on panel
            yb = np.atleast_2d((i+1)*deltax)                                                    # y coordinate of end horseshoe vortex on panel
            xa = np.atleast_2d(((i+1)*deltax-deltax/2)*np.tan(sweep) + 0.25*chord_distribution) # x coordinate of horseshoe vortex on panel
            x  = np.atleast_2d(((i+1)*deltax-deltax/2)*np.tan(sweep) + 0.75*chord_distribution) # x coordinate of control points on panel
            y  = np.atleast_2d(((i+1)*deltax-deltax/2))                                         # y coordinate of control points on panel 
        
        #-------------------------------------------------------------------------------------------------------
        # PROPELLER SLIPSTREAM MODEL
        #-------------------------------------------------------------------------------------------------------
        '''
        SOURCE: Aerodynamic Modelling of the Wing-Propeller Interaction
        Summary: This method uses the blade element momentum solution to modify the local angle of 
        attach and axial velocity incident on the wing
        '''
        rho              = conditions.freestream.density[index][0]                  
        q_inf            = conditions.freestream.dynamic_pressure[index][0] 
        V_inf            = conditions.freestream.velocity[index][0]  
        V_distribution   = np.ones((1,n))*V_inf      
        aoa              = conditions.aerodynamics.angle_of_attack[index][0]   
        aoa_distribution = np.ones((1,n))*aoa  
        
        if 'propulsor' in propulsors:
            prop =  propulsors['propulsor'].propeller            
            propeller_status = True
        else: 
            propeller_status = False

        if propeller_status : # If propellers present, find propeller location and re-vectorize wing with embedded propeller               
            prop_slipstream_V_contribution = 0              
            num_prop   = len(prop.origin)                                                 # number of propellers                  
            for i in range(num_prop):                                                     # loop through propellers on aircraft to get combined effect of slipstreams
                # obtain original properties of propeller 
                D_p       = prop.tip_radius*2
                R_p       = prop.tip_radius                     
                Vx        = prop.run_attributes.velocity[index]            
                r_nacelle = prop.hub_radius 
                vt_old    = np.concatenate((prop.run_attributes.vt[index], - prop.run_attributes.vt[index][::-1]), axis=0)   # induced tangential velocity at propeller disc using propeller discretization
                va_old    = np.concatenate((-prop.run_attributes.va[index], - prop.run_attributes.va[index][::-1]), axis=0)   # induced axial velocity at propeller disc  using propeller discretization
                n_old     = len(prop.chord_distribution)                      # number of spanwise divisions from propeller
                r_old     = np.linspace(prop.hub_radius,R_p,n_old) 
                d_old     = np.concatenate((-r_old[::-1], r_old) ,  axis=0)   
                
                # slipstream development from propeller disc to control point on wing
                s                 =  prop.origin[i][0] - wing.origin[0]      # assuming straight (non-tapered) wings
                Kd                = 1 + s/(np.sqrt(s**2 + R_p**2))     
                r_div_r_prime_val = np.zeros(n_old) 
                r_prime = np.zeros(n_old) 
                for j in range(n_old):
                    if j == 0:
                        r_prime[j] =  prop.hub_radius
                    else: 
                        Kv         = (2*Vx + prop.run_attributes.va[index][j] + prop.run_attributes.va[index][j-1])/(2*Vx + Kd*(prop.run_attributes.va[index][j] +  prop.run_attributes.va[index][j-1]))
                        r_prime[j] =  np.sqrt(r_prime[j-1]**2 + ( r_old[j]**2 -  r_old[j-1]**2)*Kv)   
                    r_div_r_prime_val[j] = r_old[j]/r_prime[j]                    
                r_div_r_prime_old =  np.concatenate((r_div_r_prime_val[::-1], r_div_r_prime_val), axis=0)   
  
                # determine location of propeller on wing                  
                prop_vec_minus = y - (prop.origin[i][1] - R_p)               
                LHS_vec        = np.extract(prop_vec_minus <=0 ,prop_vec_minus)                 
                if (prop.origin[i][1] + R_p) < span: 
                    prop_vec_plus  = y - (prop.origin[i][1] + R_p)
                    RHS_vec        = np.extract(prop_vec_plus >0 ,prop_vec_plus)   
                    end_val        = np.where(prop_vec_plus == min(RHS_vec))[1][0] +1 
                    n              = (np.where(prop_vec_plus == min(RHS_vec))[1] - np.where(prop_vec_minus == max(LHS_vec))[1]) + 1 
                       
                else: 
                    end_val       = len(y[0])
                    n             = (end_val  - np.where(prop_vec_minus == max(LHS_vec))[1])                      
                    y_prop        = d_old + span
                    cut_off       = (y_prop - span)[( y_prop - span) <= 0]
                    cut_off       = cut_off.argmax()
                    vt_old        = vt_old[:cut_off]
                    va_old        = va_old[:cut_off]
                    r_old         = r_old[:cut_off] 
                    d_old         = d_old[:cut_off] 
                    r_div_r_prime_old = r_div_r_prime_old[:cut_off]
                    
                # changes the discretization on propeller diameter to match the discretization on the wing            
                vt             = np.interp(np.linspace(-R_p,max(d_old),n) , d_old , vt_old)     # induced tangential velocity at propeller disc using wing discretization
                va             = np.interp(np.linspace(-R_p,max(d_old),n) , d_old , va_old)     # induced axial velocity at propeller disc using wing discretization
                d              = np.interp(np.linspace(-R_p,max(d_old),n) , d_old , d_old) 
                r_div_r_prime  = np.interp(np.linspace(-R_p,max(d_old),n) , d_old , r_div_r_prime_old)                      
        
                va_prime       = Kd*va
                vt_prime       = 2*vt*r_div_r_prime
                
                # compute new components of freestream
                Vx             = V_inf*np.cos(aoa) + va_prime
                Vy             = V_inf*np.sin(aoa) + vt_prime
                modified_V_inf = np.sqrt(Vx**2 + Vy**2 )
                modified_aoa   = np.arctan(Vx/Vy)
                numel          = len(modified_aoa)
                modified_V_inf = np.reshape(modified_V_inf, (1,numel)) # reshape vector 
                modified_aoa   = np.reshape(modified_aoa, (1,numel))   # reshape vector 
                
                # modifiy air speed distribution being propeller 
                start_val = np.where(prop_vec_minus == max(LHS_vec))[1][0]  
                V_distribution[0][start_val : end_val]   = modified_V_inf 
                aoa_distribution[0][start_val : end_val] = modified_aoa               
  
            #fig = plt.figure('Propeller Induced Speeds') 
            #axes1 = fig.add_subplot(2,1,1)
            #axes1.plot(y,aoa_distribution,'bo-')
            #axes1.set_xlabel('Span (m)')
            #axes1.set_ylabel(r'Tangential Velocity $m/s$')
            #axes1.grid(True)  
            #axes3 = fig.add_subplot(2,1,2)
            #axes3.plot(y,V_distribution,'bo-' )
            #axes3.set_xlabel('Span (m)')
            #axes3.set_ylabel(r'Velocity Distribution $m/s$')
            #axes3.grid(True)                 
            #plt.show() 
            
            q_distribution = 0.5*rho*V_distribution**2
                                 
            CL, CL_distribution, CD, CD_distribution = compute_forces(x,y,xa,ya,yb,deltax,twist_distribution,aoa_distribution,q_distribution,q_inf,Sref)            
        else:
             # aoa_distribution = 0 correct
            CL, CL_distribution, CD, CD_distribution = compute_forces(x,y,xa,ya,yb,deltax,twist_distribution,aoa_distribution,q_distribution,q_inf,Sref)

    return  CL_distribution, CL , CD_distribution,CD 
        
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

def compute_forces(x,y,xa,ya,yb,deltax,twist_distribution,aoa_distribution,q_distribution,q_inf, Sref):    
    sin_aoa = np.sin(aoa_distribution)
    cos_aoa = np.cos(aoa_distribution)

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
    CL_distribution = L[0]        
    CD_distribution = D[0]   
    
    # Lift & Drag distribution
    Lift_distribution = q_distribution*L[0]        
    Drag_distribution = q_distribution*D[0]       

    # Total Lift and Drag
    LT = sum(Lift_distribution[0]) 
    DT = sum(Drag_distribution[0])
    
    # Lift and Drag Coefficents 
    CL = 2*LT/(0.5*Sref*q_inf)
    CD = 2*DT/(0.5*Sref*q_inf)  
    
    fig = plt.figure('Lift Distribution') 
    axes1 = fig.add_subplot(2,1,1)
    axes1.plot(y,LT,'bo-')
    axes1.set_xlabel('Span (m)')
    axes1.set_ylabel(r'Tangential Velocity $m/s$')
    axes1.grid(True)  
    axes2 = fig.add_subplot(2,1,2)
    axes2.plot(y,CL,'bo-' )
    axes2.set_xlabel('Span (m)')
    axes2.set_ylabel(r'Velocity Distribution $m/s$')
    axes2.grid(True)                 
    plt.show() 
    
    return  CL , CL_distribution , CD , CD_distribution 
