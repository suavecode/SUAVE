# weissinger_vortex_lattice.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Apr 2017, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Weissinger Vortex Lattice
# ----------------------------------------------------------------------

def weissinger_vortex_lattice(conditions,configuration,wing):
    """ SUAVE.Methods.Aerodynamics.Pass_fidelity.vlm(conditions,configuration,geometry)
        Vortex lattice method to compute the lift coefficient and induced drag component

        Inputs:
            wing - geometry dictionary with fields:
                Sref - reference area

        Outputs:

        Assumptions:
        
    """

    #unpack
    span        = wing.spans.projected
    root_chord  = wing.chords.root
    tip_chord   = wing.chords.tip
    sweep       = wing.sweeps.quarter_chord
    taper       = wing.taper
    twist_rc    = wing.twists.root
    twist_tc    = wing.twists.tip
    sym_para    = wing.symmetric
    AR          = wing.aspect_ratio
    Sref        = wing.areas.reference
    orientation = wing.vertical

    n  = configuration.number_panels_spanwise
    nn = configuration.number_panels_chordwise

    # conditions
    aoa = conditions.aerodynamics.angle_of_attack
    
    # chord difference
    dchord=(root_chord-tip_chord)
    if sym_para is True :
        span=span/2
    deltax=span/n

    if orientation == False :

        section_length = np.empty(n)
        area_section   = np.empty(n)
        sl             = np.empty(n)
        xpos           = np.empty(n)
        ya             = np.empty(n)
        yb             = np.empty(n)
        xa             = np.empty(n)
        yab            = np.empty(n)
        ybb            = np.empty(n)
        y2             = np.empty(n)   
        x              = np.empty(n)
        y              = np.empty(n)
        twist_distri   = np.empty(n)
        xloc_leading   = np.empty(n)
        xloc_trailing  = np.empty(n)  
        RHS            = np.empty([n,1])
        w              = np.empty([n,n])
        wb             = np.empty([n,n])
        A              = np.empty([n,n])
        L              = np.empty(n)
        T              = np.empty(n)
        A_v            = np.empty([n,n])
        v              = np.empty(n)
        Lfi            = np.empty(n)
        Lfk            = np.empty(n)
        Lft            = np.empty(n)
        Dg             = np.empty(n)   
        D              = np.empty(n)    
    
        # discretizing the wing sections into panels
        for i in xrange(n):
    
            section_length[i] = dchord/span*(span-(i+1)*deltax+deltax/2) + tip_chord
            area_section[i]   = section_length[i]*deltax
            sl[i]             = section_length[i]
            twist_distri[i]   = twist_rc + i/float(n)*(twist_tc-twist_rc)
            xpos[i]           = (i)*deltax
    
            ya[i] = (i)*deltax
            yb[i] = (i+1)*deltax
            xa[i] = ((i+1)*deltax-deltax/2)*np.tan(sweep)+ 0.25*sl[i]
    
            x[i] = ((i+1)*deltax-deltax/2)*np.tan(sweep) + 0.75*sl[i]
            y[i] = ((i+1)*deltax-deltax/2)
            
            xloc_leading[i]  = ((i+1)*deltax)*np.tan(sweep)
            xloc_trailing[i] = ((i+1)*deltax)*np.tan(sweep)+sl[i]
    
    
        # Influence coefficient computation
        for i in xrange(n):
    
            RHS[i,0] = np.sin(twist_distri[i]+aoa)
    
            for j in range(0,n):
    
                yad = y[i]-ya[j]
                xd  = x[i]-xa[j]
                ybd = y[i]-yb[j]
    
                yadd = y[i]-yab[j]
                ybdd = y[i]-ybb[j]
                if (i==4) and (j==3):
                    aaaa = 0
    
                A[i,j] = whav(x[i],y[i],xa[j],ya[j])-whav(x[i],y[i],xa[j],yb[j])\
                    -whav(x[i],y[i],xa[j],-ya[j])+whav(x[i],y[i],xa[j],-yb[j])
                A[i,j] = A[i,j]*0.25/np.pi
    
        # Vortex strength computation by matrix inversion
        Ac = A*1.
        T = np.linalg.solve(A,RHS)
        
        # Calculating the effective velocty 
        for i in xrange(n):
            v[i] = 0.0      
            for j in range(0,n):
    
                A_v[i,j] = whav(xa[i],y[i],xa[j],ya[j])-whav(xa[i],y[i],xa[j],yb[j])\
                    -whav(xa[i],y[i],xa[j],-ya[j])+whav(xa[i],y[i],xa[j],-yb[j])
                
                A_v[i,j] = A[i,j]*0.25/np.pi*T[j]
                v[i]     = v[i]+A_v[i,j]
            
            Lfi[i] = -T[i]*(np.sin(twist_tc)-v[i])
            Lfk[i] = T[i]*np.cos(twist_tc)        
    
            Lft[i] = (-Lfi[i]*np.sin(twist_tc)+Lfk[i]*np.cos(twist_tc))
            Dg[i]  = (Lfi[i]*np.cos(twist_tc)+Lfk[i]*np.sin(twist_tc))
    
        # Lift computation from elements
        LT    = 0.0
        DT    = 0.0
        arsec =0.0
    
        for i in xrange(n):
    
            L[i] = deltax*Lft[i] 
            D[i] = deltax*Dg[i]    
    
            LT = LT+L[i]
            DT = DT+D[i]
    
            arsec=arsec+area_section[i] 
    
        Cl = 2*LT/(0.5*Sref)
        Cd = 2*DT/(0.5*Sref)     
    
    else:
        
        Cl = 0.0
        Cd = 0.0   
    
    #if wing.tag == 'main_wing':
        #a = np.ones([1,5])*span
        #new = np.vstack((a,Ac))
        ##new = np.array([[span,A[0,0]]])
        #try:
            #base_array = np.load('vortex_test.npy')
            ##if np.any(base_array[:,0]==span):
                ##pass
            #if base_array[0,0]==span:
                #pass            
            #else:
                #new_write = np.vstack((new,base_array))
                #np.save('vortex_test.npy',new_write)
                #print span
        #except IOError:
            #np.save('vortex_test.npy',new)
            #print span

    return Cl, Cd

# ----------------------------------------------------------------------
#   Helper Functions
# ----------------------------------------------------------------------
def whav(x1,y1,x2,y2):
    """ Helper function of vortex lattice method      
        Inputs:
            x1,x2 -x coordinates of bound vortex
            y1,y2 -y coordinates of bound vortex

        Sref - reference area for non dimensionalization
        Outpus:
            Cl_comp - lift coefficient
            Cd_comp - drag  coefficient       

        Assumptions:
            if needed

    """  
    if np.isclose(x1,x2):
        whv=1/(y1-y2)
    else:  
        whv=1/(y1-y2)*(1+ (np.sqrt((x1-x2)**2+(y1-y2)**2)/(x1-x2)))

    return whv