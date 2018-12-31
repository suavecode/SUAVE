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
    Lift_distribution = np.zeros((1,n))
    Drag_distribution = np.zeros((1,n))    

    # chord difference
    dchord = (root_chord-tip_chord)
    if sym_para is True :
        span = span/2            
    deltax = span/n

 
    #-------------------------------------------------------------------------------------------------------
    # PROPELLER SLIPSTREAM EFFECT  
    #-------------------------------------------------------------------------------------------------------

    rho              = conditions.freestream.density[index][0]            
    aoa              = conditions.aerodynamics.angle_of_attack[index][0]         
    q_inf            = conditions.freestream.dynamic_pressure [index][0] 
    q_distribution   = np.ones(n)*q_inf
    V_distribution   = np.ones(n)*conditions.freestream.velocity[index][0]  
    aoa_distribution = np.ones(n)*aoa 

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
        if 'propulsor' in propulsors:
            prop =  propulsors['propulsor'].propeller            
            propeller_status = True
        else: 
            propeller_status = False

        if propeller_status : # If propellers present, find propeller location and re-vectorize wing with embedded propeller               
            prop_slipstream_V_contribution = 0              
            num_prop   = len(prop.origin)                                                 # number of propellers                  
            for i in range(num_prop):                                                     # loop through propellers on aircraft to get combined effect of slipstreams
                r_nacelle             = prop.hub_radius                               # naceller radius
                vt_old                = prop.run_properties.vt                            # induced tangential velocity at propeller disc using propeller discretization
                va_old                = prop.run_properties.va                            # induced axial velocity at propeller disc  using propeller discretization
                n_old                 = len(prop.chord_distribution)                      # number of spanwise divisions from propeller 
                D_p                   = prop.tip_radius*2
                R_p                   = prop.tip_radius            
                prop_vec_plus         = y - (prop.origin[i][1] + R_p)
                prop_vec_minus        = y - (prop.origin[i][1] - R_p)
                RHS_vec               = np.extract(prop_vec_plus >0 ,prop_vec_plus)         
                LHS_vec               = np.extract(prop_vec_minus <=0 ,prop_vec_minus) 
                n                     = ( np.where(prop_vec_plus == min(RHS_vec))[0] -   
                                          np.where(prop_vec_minus == max(LHS_vec))[0])/2  # changes the discretization to match the discretization on the wing            
                vt = np.interp(np.linspace(0,R_p,n), vt_old, np.linspace(0,R_p,n_old))    # induced tangential velocity at propeller disc using wing discretization
                va = np.interp(np.linspace(0,R_p,n), va_old, np.linspace(0,R_p,n_old))    # induced axial velocity at propeller disc using wing discretization
                
                # slipstream development from propeller disc to control point on wing
                s =  prop.origin[i][0] - wing.origin[0]                    # assuming straight (non-tapered) wings
                Kd = 1 + s/(sqrt(s**2 + R_p**2))
                for j in range( ):
                    if j == 1:
                        r_prim =  prop.nacelle_radius
                    else: 
                        Kv = (2*Vx + va[j] + va[j-1])/(2*Vx + Kd(va[j] +  va[j-1]))
                        r_prime =  sqrt( r_prime**2 - (r[j] - r[j-1]**2))
                
                prop_slipstream_V_contribution =+ 2                                  
                    
            # distribution of dynamic pressure      
            q_distribution =  0.5*(V_distribution+prop_slipstream_V_contribution)**2  #  ## CHECK ##
            aoa_distribution
            LT,CL,DT, CD, Lift_distribution,Drag_distribution= compute_forces(x,y,xa,ya,yb,deltax,twist_distribution,aoa_distribution ,q_inf,q_distribution,chord_distribution,Sref)            
        else:
            q_distribution = 0.5*rho*V_distribution**2          #  ## CHECK ##
            LT,CL,DT,CD,Lift_distribution,Drag_distribution = compute_forces(x,y,xa,ya,yb,deltax,twist_distribution,aoa_distribution ,q_inf,q_distribution,chord_distribution,Sref)


        ##-----------------------------------------------------------
        ## PLOT LIFT & DRAF DISTRIBUTION
        ##-----------------------------------------------------------
        #wing_span          = np.array(np.linspace(0,span,n))
        
        #fig = plt.figure('Semi Span Aerodynamics')
        #fig.set_size_inches(10, 8)

        #axes2 = fig.add_subplot(2,1,1)
        #axes2.plot( wing_span , V_distribution , 'ro-' )
        #axes2.set_xlabel('Span (m)')
        #axes2.set_ylabel(r'Local Velocity $m/s$')
        #axes2.grid(True)        
        #axes3 = fig.add_subplot(2,1,2)
        #axes3.plot( wing_span , Lift_distribution[0], 'bo-' )
        #axes3.set_xlabel('Span (m)')
        #axes3.set_ylabel(r'$Spanwise Lift$')
        #axes3.grid(True)        
        #plt.show()

    return  LT , CL , DT, CD   
        
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

def compute_forces(x,y,xa,ya,yb,deltax,twist_distribution,aoa_distribution,q_inf,q_distribution,chord_distribution,Sref):    
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
    
    # ----------------------------------------------
    # OLD CODE    
    ## Total lift
    #LT = np.sum(L)
    #DT = np.sum(D)

    # Total lift
    LT = np.sum(L)
    DT = np.sum(D)

    CL1 = 2*LT/(0.5*Sref)
    CD1 = 2*DT/(0.5*Sref) 

    # Lift & Drag distribution
    Lift_distribution      = q_distribution *L[0]        
    Drag_distribution      = q_distribution *D[0]       

    # Total Lift and Drag
    LT = sum(Lift_distribution) 
    DT = sum(Drag_distribution)
    
    CL = 2*LT/(0.5*Sref*q_inf)
    CD = 2*DT/(0.5*Sref*q_inf)  
    
    return LT , CL , DT, CD  , Lift_distribution, Drag_distribution   

