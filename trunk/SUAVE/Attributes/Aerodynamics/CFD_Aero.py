""" CFD_Aero.py: Vortex lattice aerodynamic model (I think?) """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy 
from SUAVE.Attributes.Aerodynamics import Aerodynamics

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------
    
class CFD_Aero(Aerodynamics):

    """ CFD_Aero: Vortex lattice aerodynamic model (I think?) """
    
    def __defaults__(self):                             # can handle CD(alpha), CL(alpha); needs Mach support (spline surface)
        self.tag = 'External Data'
        self.S = 1.0                                    # reference area (m^2)
        self.M = []                                     # Mach vector
        self.alpha = []                                 # alpha vector
        self.CD = []                                    # CD vector
        self.CL = [] # CL vector
        self.AR = 0.0                                       # aspect ratio
        self.e = 1.0                                        # Oswald factor
        self.CD0 = 0.0                                      # CD at zero lift
        self.CL0 = 0.0                                      # CL at alpha = 0.0
        self.dCLdalpha = 2*numpy.pi       # dCL/dalpha
        
        self.l_fus= []
        self.d_fus= []
        self.l_nose= []
        self.l_tail= []

        
        self.mac_w= []
        self.t_c_w= []
        self.sweep_w= []
        self.S_exposed_w= []
        self.S_affected_w= []
        self.arw_w= []
        self.span_w= []
        
        
        
        self.d_engexit= []        
        
        
        
        
        #-----------individual values-------
        
        self.l_fus=0.0
        self.d_fus=0.0
        self.l_nose=0.0
        self.l_tail=0.0

        
        self.mac_w=0.0
        self.t_c_w=0.0
        self.sweep_w=0.0
        self.S_exposed_w=0.0
        self.S_affected_w=0.0
        self.arw_w=0.0
        self.span_w=0.0
        
        self.mac_h=0.0
        self.t_c_h=0.0
        self.sweep_h=0.0
        self.S_exposed_h=0.0
        self.S_affected_h=0.0
        self.arw_h=0.0
        self.span_h=0.0
        
        self.mac_v=0.0
        self.t_c_v=0.0
        self.sweep_v=0.0
        self.S_exposed_v=0.0
        self.S_affected_v=0.0
        self.arw_v=0.0
        self.span_v=0.0
        
        self.d_engexit=0.0         
        
        
    def aircraft_lift(self,vehicle,alpha):
        
        
        #compute effeciency factor for the wings and the drag
        
    #<<<<<<< .mine
        cl_w=numpy.empty(3)
        cd_w=numpy.empty(3)
        n=2
    #=======
        
    #    n=10
    #>>>>>>> .r226
        nn=1
        gm=1.4
        R=287
        
        Wing1=vehicle.Wings[0]
        #Wing1=vehicle.Wings[0]
        Wing2=vehicle.Wings[1] 
        #Wing2=vehicle.Wings[1] 
        Wing3=vehicle.Wings[2]
        #Wing3=vehicle.Wings[2]
        #Mc=state.M[curr_itr]
    
        alpha1=alpha
        alpha2=alpha
        alpha3=0
    
        #--obtained from atmos    
        #roc=state.rho[curr_itr]
        #muc =state.mew[curr_itr]
        #Tc=state.T[curr_itr]
        
        #print Tc
        
        #Vinf=Mc*numpy.sqrt(gm*R*Tc)   
        Sref=Wing1.sref
        #self.S = Sref
        
        
        #if Wing1.highlift == True: 
        #if Wing1.hl == 1: 
            #[cl_w1,cd_w1]=high_lift(Wing1,state,curr_itr)
        #else :   
        [cl_w1,cd_w1]=self.vlm(Wing1.span,Wing1.chord_root,Wing1.chord_tip,n,nn,Wing1.sweep,Wing1.taper,Wing1.alpha_rc,Wing1.alpha_tc,Sref,Wing1.symmetric,alpha1,Wing1.ar)
    
        [cl_w2,cd_w2]=self.vlm(Wing2.span,Wing2.chord_root,Wing2.chord_tip,n,nn,Wing2.sweep,Wing2.taper,Wing2.alpha_rc,Wing2.alpha_tc,Sref,Wing2.symmetric,alpha2,Wing2.ar)
        [cl_w3,cd_w3]=self.vlm(Wing3.span,Wing3.chord_root,Wing3.chord_tip,n,nn,Wing3.sweep,Wing3.taper,Wing3.alpha_rc,Wing3.alpha_tc,Sref,Wing3.symmetric,alpha3,Wing3.ar)
        #cl_wing=vlm(wing_main,State)
        #cl_horz=vlm(wing_horz,State)
    #<<<<<<< .mine
        #cl_fus=0.1
        #cl_airc=1.14*(cl_w1+cl_w2+cl_w3)   #+cl_fus
        cl_w[0]=cl_w1
        cd_w[0]=cd_w1  
        
        cl_w[1]=cl_w2
        cd_w[1]=cd_w2
        
        cl_w[2]=cl_w3
        cd_w[2]=cd_w3   
    #=======
        #cl_fus=0.1
        cl_airc=1.14*(cl_w1+cl_w2+cl_w3)   #+cl_fus
        #print cl_airc
    #>>>>>>> .r226
        #print 'Cl aircraft  ',cl_airc
        #print 'Cl aircraft  ',cl_airc
        #return cl_airc,cl_w,cd_w
        return cl_airc
    
    
    def whav(self,x1,y1,x2,y2):
    
    #<<<<<<< .mine
        if x1==x2:
            whv=1/(y1-y2)
        else:  
            whv=1/(y1-y2)*(1+ (numpy.sqrt((x1-x2)**2+(y1-y2)**2)/(x1-x2)))
        
        return whv
    #=======
    #def whav(x1,y1,x2,y2):
    #>>>>>>> .r226
    
        #if x1==x2:
            #whv=1/(y1-y2)
        #else:  
            #whv=1/(y1-y2)*(1+ (numpy.sqrt((x1-x2)**2+(y1-y2)**2)/(x1-x2)))
        
        #return whv
    
    
    
    #def vlm(Wing,State):
    def vlm(self,span,root_chord,tip_chord,n,nn,sweep,taper,alpha_rc,alpha_tc,Sref,sym_para,aoa,AR):
        
        # span=200           #half span
        # root_chord=30
        # tip_chord=10
        # n=500                 #number of sections
        # nn=1
        # sweep=20*pi/180
        # taper=0
        # alpha=3*pi/180
        # Vinf=300
        # Sref=8000
        # rho=1.2
        
        #print Vinf
        #alpha_tc=alpha_tc*numpy.pi/180
        
        #eeta=0.97
       
        #Cl_alpha=2*numpy.pi*AR/(2+(numpy.sqrt( ((AR/eeta)**2)*(1+numpy.tan(sweep)**2-Mc**2)+4)))
        #Cl=Cl_alpha*numpy.pi/180*alpha_rc
        #print Cl
        rho=1.0
        Vinf=1.0
        dchord=(root_chord-tip_chord)
        if sym_para is True :
    
            span=span/2
        
        deltax=span/n
        
        section_length= numpy.empty(n)
        area_section=numpy.empty(n)
        sl=numpy.empty(n)
        
        xpos=numpy.empty(n)
        
        ya=numpy.empty(n)
        yb=numpy.empty(n)
        xa=numpy.empty(n)
        yab=numpy.empty(n)
        ybb=numpy.empty(n)
        y2=numpy.empty(n)   
        
        x=numpy.empty(n)
        y=numpy.empty(n)
        twist_distri=numpy.empty(n)
        xloc_leading=numpy.empty(n)
        xloc_trailing=numpy.empty(n)  
        RHS=numpy.empty([n,1])
        w=numpy.empty([n,n])
        wb=numpy.empty([n,n])
        A=numpy.empty([n,n])
        L=numpy.empty(n)
        T=numpy.empty(n)
        
        A_v=numpy.empty([n,n])
        v=numpy.empty(n)
    
    
        Lfi=numpy.empty(n)
        Lfk=numpy.empty(n)
        
        Lft=numpy.empty(n)
        Dg=numpy.empty(n)   
        D=numpy.empty(n)    
        
       
        
        
        #--discretizing the wing sections into panels--------------------------------
        for i in range(0,n):
        #for i=1:n
        
            section_length[i]= dchord/span*(span-(i+1)*deltax+deltax/2) + tip_chord
            area_section[i]=section_length[i]*deltax
            sl[i]=section_length[i]
            twist_distri[i]=alpha_rc + i/float(n)*(alpha_tc-alpha_rc)
            xpos[i]=(i)*deltax
            
            ya[i]=(i)*deltax
            yb[i]=(i+1)*deltax
            xa[i]=((i+1)*deltax-deltax/2)*numpy.tan(sweep)+ 0.25*sl[i]
            
            x[i]=((i+1)*deltax-deltax/2)*numpy.tan(sweep) + 0.75*sl[i]
            y[i]=((i+1)*deltax-deltax/2)
            xloc_leading[i]=((i+1)*deltax)*numpy.tan(sweep)
            xloc_trailing[i]=((i+1)*deltax)*numpy.tan(sweep)+sl[i]
        
        #end
        
        # end
        
        #----------------------------------------------------------------------------
        
        #------------------------coordinates for the alternate panel---------------
        #yab=-ya
        #ybb=-yb
        #y2=-y
        
        #yab=-yb
        #ybb=-ya
        #y2=-y    
        
        #--------------------------------------------------------------------
        
       
        #------Influence coefficient computation-----------------------
        
        
        
        for i in range(0,n):
        #for i=1:n
            RHS[i,0]=Vinf*numpy.sin(twist_distri[i]*numpy.pi/180+aoa)  #)
            #for j=1:n
            for j in range(0,n):
              
                #if x~=xa  #(i==j) implement that separately
                #only for off diagonal terms
                
                yad=y[i]-ya[j]
                xd=x[i]-xa[j]
                ybd=y[i]-yb[j]
                
                yadd=y[i]-yab[j]
                ybdd=y[i]-ybb[j]
                
                
                #print yad
                #print yadd
                
                #w[i,j]=1/(4*numpy.pi)* ( (1/yad)*(1+ (numpy.sqrt((xd)**2+(yad)**2)/xd)) - (1/ybd)*(1+(numpy.sqrt((xd)**2+(ybd)**2)/xd)))  
                #wb[i,j]=1/(4*numpy.pi)* ( (1/yadd)*(1+ (numpy.sqrt((xd)**2+(yadd)**2)/xd)) -(1/ybdd)*(1+(numpy.sqrt((xd)**2+(ybdd)**2)/xd)))  
          
                A[i,j]=self.whav(x[i],y[i],xa[j],ya[j])-self.whav(x[i],y[i],xa[j],yb[j])-self.whav(x[i],y[i],xa[j],-ya[j])+self.whav(x[i],y[i],xa[j],-yb[j])
                A[i,j]=A[i,j]*0.25/numpy.pi
        
         
        #---------------Vortex strength computation by matrix inversion-----------
        
        #if sym_para is True :
    
            #A=w+wb
        
        #else :
            #A=w
            
    
        T=numpy.linalg.solve(A,RHS)
        
        
        #---Calculating the effective velocty--------------------------
        #W=0
        for i in range(0,n):
           v[i]=0.0      #Vinf
           for j in range(0,n):
       
               A_v[i,j]=self.whav(xa[i],y[i],xa[j],ya[j])-self.whav(xa[i],y[i],xa[j],yb[j])-self.whav(xa[i],y[i],xa[j],-ya[j])+self.whav(xa[i],y[i],xa[j],-yb[j])
               A_v[i,j]=A[i,j]*0.25/numpy.pi*T[j]
               v[i]=v[i]+A_v[i,j]
       
    
           Lfi[i]=-T[i]*(Vinf*numpy.sin(alpha_tc)-v[i])
           Lfk[i]=T[i]*Vinf*numpy.cos(alpha_tc)
    
           Lft[i]=(-Lfi[i]*numpy.sin(alpha_tc)+Lfk[i]*numpy.cos(alpha_tc))
           Dg[i]=(Lfi[i]*numpy.cos(alpha_tc)+Lfk[i]*numpy.sin(alpha_tc))
           
    
           
       
        #---------Lift computation from elements---------------------------------
        LT=0.0
        DT=0.0
        arsec=0.0
        #print y
        #print ya
        #print ybb
        
        for i in range(0,n):
    
            L[i]=deltax*rho*Lft[i]   #T(i)*v(i)*sin(alpha)
            D[i]=deltax*rho*Dg[i]    
            
            LT=LT+L[i]
            DT=DT+D[i]
            
            arsec=arsec+area_section[i]
    
        #if sym_para is True :
        #LT=2*LT
    
        
        #plotting the lift distribution
        #plt.plot(y,L,'-o',linewidth = 2.5, markersize=5, markerfacecolor='None')
        #plt.xlabel(r'$spanwise coord$', fontsize=24)
        #plt.ylabel(r'$Lift$', fontsize=24)
        #plt.savefig('wing_lift_distri.png',format='png')    
    
        
        #--Karman Tsien compressibility correction to be added to prevent singularity at M=1
        #Cl=LT/((0.5*rho*Vinf**2*Sref)*(numpy.sqrt(1-Mc**2)))
    
        
        
        
        
        
        
        
        Cl=2*LT/(0.5*rho*Vinf**2*Sref)
        #Cl_comp=Cl/(numpy.sqrt(1-Mc**2))
        #datcom_formula_used
    #<<<<<<< .mine
    
    
    
    
    
    
    
        Cd=2*DT/(0.5*rho*Vinf**2*Sref)   
        
        #
        #Cd=2*DT/(0.5*rho*Vinf**2*Sref)   
        #Cd_comp=Cd/(numpy.sqrt(1-Mc**2))
        
        #print 'AERO'
        #print Cl_comp
        #print Cd_comp
    #>>>>>>> .r226
    
    #<<<<<<< .mine
        return Cl, Cd
        
    #=======        
    
    #----------vortex lattice lift computation------------------

    
    def initialize(self,vehicle,Cl_var):
        
        #for key in vehicle.iteritems():
            #print key
            #if isinstance(val,Physical_Component.Container):
            #if key == 'Wings':
   
        self.mac_w=numpy.empty(len(vehicle.Wings))
        self.t_c_w=numpy.empty(len(vehicle.Wings))
        self.sweep_w=numpy.empty(len(vehicle.Wings))
        self.S_exposed_w=numpy.empty(len(vehicle.Wings))
        self.S_affected_w=numpy.empty(len(vehicle.Wings))
        self.arw_w=numpy.empty(len(vehicle.Wings))
        self.span_w=numpy.empty(len(vehicle.Wings))
        
        
        self.l_fus=numpy.empty(len(vehicle.Fuselages))
        self.d_fus=numpy.empty(len(vehicle.Fuselages))
        self.l_nose=numpy.empty(len(vehicle.Fuselages))
        self.l_tail=numpy.empty(len(vehicle.Fuselages))        
        
        self.d_engexit=numpy.empty(len(vehicle.Propulsors))
        
        for k in range(len(vehicle.Wings)): 
            
            self.mac_w[k]=vehicle.Wings[k].chord_mac
            self.t_c_w[k]=vehicle.Wings[k].t_c
            self.sweep_w[k]=vehicle.Wings[k].sweep
            self.S_exposed_w[k]=vehicle.Wings[k].S_exposed
            self.S_affected_w[k]=vehicle.Wings[k].S_affected
            self.arw_w[k]=vehicle.Wings[k].ar
            self.span_w[k]=vehicle.Wings[k].span                    
                    
                    
           # elif key == 'Fuselages':

        for k in range(len(vehicle.Fuselages)): 
            self.l_fus[k]=vehicle.Fuselages[k].length_cabin
            self.d_fus[k]=vehicle.Fuselages[k].width
            print self.l_fus[k]
            self.l_nose[k]=vehicle.Fuselages[k].length_nose
            self.l_tail[k]=vehicle.Fuselages[k].length_tail        
            
                    
                    
            # key == 'Propulsors':
 
        for k in range(len(vehicle.Propulsors)):
            self.d_engexit[k]=vehicle.Propulsors[k].df 

    
        Wing1=vehicle.Wings[0]
        #Wing2=vehicle.Wings[1] 
        #Wing3=vehicle.Wings[2]      
        self.Sref=Wing1.sref
        self.S = self.Sref
        #self.l_fus=vehicle.Fuselages[0].length_cabin
        #self.d_fus=vehicle.Fuselages[0].width
        #self.l_nose=vehicle.Fuselages[0].length_nose
        #self.l_tail=vehicle.Fuselages[0].length_tail        


        #self.mac_w=Wing1.chord_mac
        #self.t_c_w=Wing1.t_c
        #self.sweep_w=Wing1.sweep
        #self.S_exposed_w=Wing1.S_exposed
        #self.S_affected_w=Wing1.S_affected
        #self.arw_w=Wing1.ar
        #self.span_w=Wing1.span
        
        #self.mac_h=Wing2.chord_mac
        #self.t_c_h=Wing2.t_c
        #self.sweep_h=Wing2.sweep
        #self.S_exposed_h=Wing2.S_exposed
        #self.S_affected_h=Wing2.S_affected
        #self.arw_h=Wing2.ar
        #self.span_h=Wing2.span
        
        #self.mac_v=Wing3.chord_mac
        #self.t_c_v=Wing3.t_c
        #self.sweep_v=Wing3.sweep
        #self.S_exposed_v=Wing3.S_exposed
        #self.S_affected_v=Wing3.S_affected
        #self.arw_v=Wing3.ar
        #self.span_v=Wing3.span
        
        #self.d_engexit=vehicle.Propulsors[0].df            


        ##Cl_a=numpy.empty(6)
        ##aoa_range=numpy.empty(6)  
        ##aoa_range=[0,1,2,3,4,5]
        ###myarray = numpy.fromfile(filename,dtype=float)
        ###myarray = numpy.loadtxt(filename,delimiter=' ',dtype=float)        
        ##for i in range(0,len(aoa_range)):  
            
            ##Cl_a[i]=self.aircraft_lift(vehicle,aoa_range[i]*numpy.pi/180)
        ###print ' myarray',myarray
        
        
        
        ###nsize=myarray.size()
        ###Cl_a=numpy.empty(nsize)
        ###aoa_range=numpy.empty(nsize)  
        
        
        ####for i in range(0,len(aoa_range)):  
            
            ####Cl_a[i]=self.aircraft_lift(vehicle,aoa_range[i]*numpy.pi/180)
        
        ###aoa_range=myarray[:,1]
        ###Cl_a=my_array[:,1]
        
        self.CL0 = Cl_var[0,1] # Cl_a[0]
        self.dCLdalpha = (Cl_var[2,1]-Cl_var[1,1])/((Cl_var[2,0]-Cl_var[1,0])*numpy.pi/180)
        #(Cl_a[2]-Cl_a[1])/((aoa_range[2]-aoa_range[1])*numpy.pi/180)
        
        
    #-----------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------
    #---------------Drag    computation----------------------------------------------------------
    
    
    
    
    #--------------------------parasite drag-------------------------------------------------
    
    #CL Cd at second segemnt climb
    
    
    
    #-------get all the variables to follow the naming convention
    
    def cdp_wing(self,l_w,t_c_w,sweep_w, S_exposed_w,Sref,Mc,roc,muc,Tc):
        #----------------wing drag----------------------------
        V=Mc*numpy.sqrt(1.4*287*Tc)  #input gamma and R
        Re_w=roc*V*l_w/muc
        #print l_w
        cf_inc_w=0.455/(numpy.log10(Re_w))**2.58  #for turbulent part
        #print cf_inc_w
        #cf_inc_w=0.455/((Re_w))**2.58  #for turbulent part
        #--effect of mach number on cf for turbulent flow
        Tw=Tc*(1+0.178*Mc**2)
        Td=Tc*(1 + 0.035*Mc**2 + 0.45*(Tw/Tc -1))
        Rd_w=Re_w*(Td/Tc)**1.5 *(Td+216/Tc+216)
        cf_w=(Tc/Td)*(Re_w/Rd_w)**0.2*cf_inc_w       
    
        #---for airfoils-----------------------------------------
        C=1.1  #best
        k_w=1+ 2*C*t_c_w*(numpy.cos(sweep_w))**2/(numpy.sqrt(1- Mc**2*(numpy.cos(sweep_w))**2)) + C**2*(numpy.cos(sweep_w))**2*t_c_w**2 *(1+5*(numpy.cos(sweep_w)**2))/(2*(1-(Mc*numpy.cos(sweep_w))**2))       
    
        Swet_w=2*(1+ 0.2*t_c_w)*S_exposed_w
        cd_p_w =k_w*cf_w*Swet_w /Sref 
        
        #print 'l_w', l_w
        #print cd_p_w
        return cd_p_w
    
    
    
    def cdp_fuselage(self,l_fus, d_fus,l_nose,l_tail,Sref,Mc,roc,muc,Tc):
        #---fuselage----------------------------------------
        #V=Mc*numpy.sqrt(gamma*R*Tc)
        gamma=1.4
        R=287  
        pi=22/7
        V=Mc*(gamma*R*Tc)**0.5
        Re_fus=roc*V*l_fus/muc
        #print Re_fus
        #cf_inc_fus=0.455/(((Re_fus))**2.58)
        cf_inc_fus=0.455/((numpy.log10(Re_fus))**2.58)
        #for turbulent part
        #--effect of mach number on cf for turbulent flow
        Tw=Tc*(1+0.178*Mc**2)
        Td=Tc*(1 + 0.035*Mc**2 + 0.45*(Tw/Tc -1))
        Rd_fus=Re_fus*(Td/Tc)**1.5 *(Td+216/Tc+216)
        cf_fus=(Tc/Td)*(Re_fus/Rd_fus)**0.2*cf_inc_fus        
    
        
        #--------------for cylindrical bodies
        #d_d=float(d_fus)/float(l_fus)
        d_d=(d_fus)/(l_fus)
        D = numpy.sqrt(1-(1-Mc**2)*d_d**2)
        
        C_fus=2.3
        a=2*(1-Mc**2)*(d_d**2)/(D**3)*(numpy.arctanh(D)-D)
        #a=2*(1-Mc**2)*d_fus**2/D**3*(2*D -D)
        du_max_u = a/(2-a)/(1-Mc**2)**0.5
        k_fus=(1+C_fus*du_max_u)**2    
    
    
        S_wetted_nose=0.75*pi*d_fus*l_nose
        S_wetted_tail=0.72*pi*d_fus*l_tail
        S_fus=pi*d_fus*l_fus
        S_fusetot=S_wetted_nose+S_wetted_tail+S_fus;        
        cd_p_fus =k_fus*cf_fus*S_fusetot /Sref  
    
        #print(du_max_u)
        #print(cd_p_fus)
        
        return cd_p_fus
    
    
    
    
    
    def cdp_misc(self,sweep_w, d_engexit,Sref,Mc,roc,muc ,Tc,S_affected_w):
        pi=22/7
        
        f_gaps_w = numpy.empty(len(sweep_w))
        cd_nacelle_base = numpy.empty(len(d_engexit))
        f_gaps_t=0.0
        for i in range(0,len(sweep_w)):
        #-------------control surface gap drag-----------------------
            f_gaps_w[i]=0.0002*(numpy.cos(sweep_w[i]))**2*S_affected_w[i]
        
            f_gaps_t= f_gaps_t+  f_gaps_w[i]      
        
        #f_gapst=f_gaps_w+f_gaps_h+f_gaps_v
    
    
    
    
        #--compute this correctly         
        cd_gaps=0.0001
    
        #------------Nacelle base drag--------------------------------
        cd_nacelle_base_t=0.0
        for i in range(0,len(d_engexit)):
            cd_nacelle_base[i]=0.5/12*pi*d_engexit[i]*0.2/Sref
            cd_nacelle_base_t=cd_nacelle_base_t+cd_nacelle_base[i]
        
        #-------fuselage upsweep drag----------------------------------
        cd_upsweep = 0.006/Sref
    
        #-------------miscellaneous drag -------------
        #increment by 1.5# of parasite drag
        #---------------------induced  drag-----------------------------
    
        #cd_trim = 0.015*Cd_airplane
        #cd_trim=0.015*0.015;   #1-2% of airplane drag
        
    
        cd_misc = cd_gaps+cd_nacelle_base_t+cd_upsweep  #+cd_trim
        #print(cd_misc)
        return cd_misc
    
    
    
    
    
    #def cdp(self,fus_l, d_fus,l_nose,l_tail,mac_w,t_c_w,sweep_w, S_exposed_w, mac_h,t_c_h,sweep_h, S_exposed_h,mac_v,t_c_v,sweep_v, S_exposed_v,d_engexit,Sref,Mc,rhoc,muc ,Tc):
    def cdp(self,l_fus, d_fus,l_nose,l_tail,mac_w,t_c_w,sweep_w, S_exposed_w, d_engexit,Sref,Mc,roc,muc ,Tc,S_affected_w):
        ''' outputs = method1(inputs)
            inputs  geometry related : fuselage  -  fus_l, d_fus,l_nose,l_tail 
                                       wing      -  mac_w,t_c_w,sweep_w, S_exposed_w,S_affected_w
                                       horiz stab-  mac_h,t_c_h,sweep_h, S_exposed_h,S_affected_h
                                       vert stab -  mac_v,t_c_v,sweep_v, S_exposed_v,S_affected_v
                                       eng nac   -  d_engexit
                                       general   -  Sref
    
                    mission related  : Mc,rhoc,muc ,Tc  
                    constants        : gamma, R,    more documentation
        '''
    
    
        #cdp_w=numpy.empty(len(sweep_w))
        #cdp_fus=numpy.empty(len(l_fus))
    
        l_w=mac_w
        
        #cdp_w=self.cdp_wing(l_w,t_c_w,sweep_w, S_exposed_w,Sref,Mc,roc,muc,Tc)
    
        ##l_v=mac_v
        ##cdp_v=self.cdp_wing(l_v,t_c_v,sweep_v, S_exposed_v,Sref,Mc,roc,muc,Tc)
    
        ##l_h=mac_h
        ##cdp_h=self.cdp_wing(l_h,t_c_h,sweep_h, S_exposed_h,Sref,Mc,roc,muc,Tc)
    
        #cdp_fus=self.cdp_fuselage(l_fus, d_fus,l_nose,l_tail,Sref,Mc,roc,muc,Tc)
    
        cdp_miscel=self.cdp_misc(sweep_w, d_engexit,Sref,Mc,roc,muc ,Tc,S_affected_w)
    
        
        #print(cdp_w)
        #print(cdp_v)
        #print(cdp_h)
        #print cdp_fus
        cd_pt=0.0
        for i in range(0,len(sweep_w)):
            cdp_w=self.cdp_wing(l_w[i],t_c_w[i],sweep_w[i], S_exposed_w[i],Sref,Mc,roc,muc,Tc)
            
            cd_pt=cd_pt+cdp_w
           
        cdp_ft=0.0    
        for i in range(0,len(l_fus)):

            cdp_fus=self.cdp_fuselage(l_fus[i], d_fus[i],l_nose[i],l_tail[i],Sref,Mc,roc,muc,Tc)
            
            
            cdp_ft=cdp_ft+cdp_fus        
        
        #-------cd  total----------------------------------------------
        #cd_p = cdp_w + cdp_v + cdp_h + cdp_fus + cdp_miscel
        cd_p = cd_pt + cdp_ft+ cdp_miscel
        
        #print(cd_p)
    
        return cd_p,cdp_w,cdp_fus
    
    #-----------------------------------------------------------------------------------------
    
    
    #-------------------induced drag----------------------------------------------------------
    
    def cdi(self,Cl, AR, cdi_inv,cdp,fd_ws):
    #def cdi(self,Cl, AR, e):
        ''' outputs = method1(inputs)
            inputs    :  Cl, AR, e
            more documentation
        '''
        pi=22/7
        #span efficiency computation check
    
        #cd_i=Cl**2/(pi*AR*e)     #simple fit
        
        s=0.92         #put in fit
        
        
        
        s = -1.7861*fd_ws**2 - 0.0377*fd_ws + 1.0007
        
        cdi_inv = cdi_inv/s
        K=0.38
        cdi_viscous = K*cdp*Cl**2
        
        cd_i = cdi_inv + cdi_viscous
        
    
        #print(cd_i)
    
        #code        
        outputs = 'induced drag'
    
        return cd_i
    
    #----------------------------------------------------------------------------------------------          
    
    
    
    
    #-------------------compressiblity drag----------------------------------------------------------
    
    def cdc(self,mach,sweep_w,t_c_w,Cl):
    #def cdc(self,Minf):
        '''inputs  geometry related : 
                                   wing      -  t_c_w,sweep_w
    
                                   mission related  : mach,Cl 
                more documentation
        '''
    
    
        #cd_c=10**-4;
        
        tc=t_c_w[0]/numpy.cos(sweep_w[0])
        cl=Cl/(numpy.cos(sweep_w[0]))**2
    
        #print 'cl ',Cl
        #mcc_cos_ws=1000 *( 0.000566641866949+ 0.015760196266428*tc -0.000990173705643*cl -0.258018145323535*tc**2 -0.000010395075985*tc*cl+ 0.004243731167502*cl**2 +1.621933547856403*tc**3 +0.009642672030852*tc*2*cl -0.002384681077508*tc*cl**2  -0.008019064497717*cl**3 -3.599702380952329*tc**4  -0.007962815824228*tc**3*cl  -0.005370400795084*tc**2*cl**2  + 0.002460473744292*tc*cl**3 +  0.004900000000000*cl**4)  #original computation
        
        mcc_cos_ws=0.922321524499352 -1.153885166170620*tc -0.304541067183461*cl  + 0.332881324404729*tc**2 +  0.467317361111105*tc*cl+   0.087490431201549*cl**2;
        
        
        mcc = mcc_cos_ws/numpy.cos(sweep_w[0])
        
    
        MDiv = mcc *(1.02 +.08 *( 1 - numpy.cos(sweep_w[0])))
    
        mo_mc=mach/mcc
        
        dcdc_cos3g = 413.56*(mo_mc)**6 - 2207.8*(mo_mc)**5 + 4900.1*(mo_mc)**4 - 5786.9*(mo_mc)**3 + 3835.3*(mo_mc)**2 - 1352.5*(mo_mc) + 198.25
        
        
        #dcdc_cos3g = 44.138*(mo_mc)**5 - 194.97*(mo_mc)**4 + 343.51*(mo_mc)**3 - 301.73*(mo_mc)**2 + 132.13*(mo_mc) - 23.074   #old incorrect value
        cd_c=dcdc_cos3g*(numpy.cos(sweep_w[0]))**3
    
    
        #print(cd_c)
    
        #code        
        #outputs = 'hello aero!'
    
        return cd_c       
    
    
    
    
    #--------------------------------------------------------------------------------------------       
    
    
    
    
    
    #-----------------------------over aircraft drag---------------------------------------------       
    #def drag(l_fus, d_fus,l_nose,l_tail,mac_w,t_c_w,sweep_w, S_exposed_w, mac_h,t_c_h,sweep_h, S_exposed_h,mac_v,t_c_v,sweep_v, S_exposed_v,d_engexit,Sref,Mc,roc,muc ,Tc,Cl, AR, e,S_affected_w,S_affected_h,S_affected_v ):
    #def drag(self,fus_l, d_fus,l_nose,l_tail,mac_w,t_c_w,sweep_w, S_exposed_w, mac_h,t_c_h,sweep_h, S_exposed_h,mac_v,t_c_v,sweep_v, S_exposed_v,d_engexit,Sref,Mc,rhoc,muc ,Tc,Cl, AR, e ):
    #def drag(vehicle,Segment):
    def drag(self,cl_w,state):
        ''' outputs = method1(inputs)
               cd_p inputs  geometry related : fuselage  -  fus_l, d_fus,l_nose,l_tail 
                                wing      -  mac_w,t_c_w,sweep_w, S_exposed_w
                                horiz stab-  mac_h,t_c_h,sweep_h, S_exposed_h
                                vert stab -  mac_v,t_c_v,sweep_v, S_exposed_v
                                eng nac   -  d_engexit
                                general   -  Sref
    
             mission related  : Mc,Tinf,rhoc,muc ,Tc  
             constants        : gamma, R,    more documentation
    
    
               cd_i  inputs    :  Cl, AR, e
                more documentation
    
        '''
        
        #unpacking
        Sref=self.Sref
        
        
        
        l_fus=self.l_fus
        d_fus=self.d_fus
        l_nose=self.l_nose
        l_tail=self.l_tail       

        print 'l_fus', l_fus
        
        mac_w=self.mac_w
        t_c_w=self.t_c_w
        sweep_w=self.sweep_w
        S_exposed_w=self.S_exposed_w
        S_affected_w=self.S_affected_w
        arw_w=self.arw_w
        span_w=self.span_w
        
        #mac_h=self.mac_h
        #t_c_h=self.t_c_h
        #sweep_h=self.sweep_h
        #S_exposed_h=self.S_exposed_h
        #S_affected_h=self.S_affected_h
        #arw_h=self.arw_h
        #span_h=self.span_h
        
        #mac_v=self.mac_v
        #t_c_v=self.t_c_v
        #sweep_v=self.sweep_v
        #S_exposed_v=self.S_exposed_v
        #S_affected_v=self.S_affected_v
        #arw_v=self.arw_v
        #span_v=self.span_v
        
        d_engexit=self.d_engexit               
        
        e=0.79
        Mc=state.M
            
        #--obtained from atmos    
        roc=state.rho
        muc =state.mew
        Tc=state.T
        
       #obtained from aero itself wait for lift model 
        #Cl_w=Wing1.Cl
        #Cl_h=Wing2.Cl
        #Cl_v=Wing3.Cl    
        
        
   
        
        Cl_w=cl_w
        
        #print 'cl_w',cl_w
        
        #cdi_w=cd_w

        
        
        
        
        #----calling necessary functions for drag computation 
        
            
        [cd_pt,cd_p,cdp_fus]=self.cdp(l_fus, d_fus,l_nose,l_tail,mac_w,t_c_w,sweep_w, S_exposed_w, d_engexit,Sref,Mc,roc,muc ,Tc,S_affected_w)
        #cd_i_w=cdi(Cl_w, arw_w, cdi_w,cdp_w,d_fus/span_w)
        #cd_i_h=cdi(Cl_h, arw_h, cdi_h,cdp_v,d_fus/span_h)
        #cd_i_v=cdi(Cl_v, arw_v, cdi_v,cdp_h,d_fus/span_v)
        
        #cd_i= cd_i_w + cd_i_h + cd_i_v
  
        
            
        cdi_t=0.0
        ##for i in range(0,len(sweep_w)):
            #cd_i=Cl_w**2/(numpy.pi*arw_w[i]*e)
            #cdi_t=cdi_t+cd_i[i]
        cdi_t=Cl_w**2/(numpy.pi*arw_w[0]*e)
        
        cd_c=self.cdc(Mc,sweep_w,t_c_w,Cl_w)
        
        #cdc_t=0.0
        #for i in range(0,len(sweep_w)):
            #cdc_t=cd_c[i]+cdc_t       
            
        
        #print 'cd_c',cd_c
        cd_tot=cd_p+cdi_t#+cd_c   #without trim
        
        cd_tot_w_trim=1.02*cd_tot #with trim
        
        
    
        #code        
        #outputs = 'total aircraft drag coeff'
    
        return cd_tot           
    



    def __call__(self,alpha,segment):

        Cl_inc= self.CL0 + self.dCLdalpha*alpha  
        CL=Cl_inc/(numpy.sqrt(1-segment.M**2))
        #print 'alpha',alpha
        #print 'Cl_inc',Cl_inc
        #print 'CL',CL
        #CD = self.CD0 + (CL**2)/(np.pi*self.AR*self.e)      # parbolic drag
        CD= self.drag(CL,segment)
        #print CD
        #print CL
        return CD, CL


