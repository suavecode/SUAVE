#----3d vortex lattice methods-------------------------------

import numpy
from matplotlib import pyplot as plt
from matplotlib import mlab

import pylab
from mpl_toolkits.mplot3d import Axes3D


def panel_normal(ra,rb,rc):

    v= numpy.empty(3)

    ro = rb-ra
    r1 = rc-ra
    r2 = rc-rb
    
    r1r2=numpy.cross(r1,r2)
    r1r2_abs=numpy.linalg.norm(r1r2)
    
    v_norm = r1r2/r1r2_abs
    
    return v_norm     




def downwash_segment(ra,rb,rc):

    v= numpy.empty(3)

    ro = rb-ra
    r1 = rc-ra
    r2 = rc-rb
    
    #print ra
    #print rb
    
    
  
    r1r2=numpy.cross(r1,r2)
    r1r2_abs=numpy.linalg.norm(r1r2)
    
    r1mr2=((r1/numpy.linalg.norm(r1)) - (r2/numpy.linalg.norm(r2)))
    ror1mr2= numpy.dot(ro,r1mr2)
    
    v= r1r2/(r1r2_abs)**2 *(ror1mr2)
    
        
    v=0.25/numpy.pi*v
    
    #print r1r2_abs
    
    
    #if r1r2_abs==0:
        #v[0]=0
        #v[1]=0
        #v[2]=1/(ra[1]-rb[1])    
    
    
    
    return v    
    
    
    
    
def downwash_segment2(ra,rb,rc):

    v= numpy.empty(3)

    ro = rb-ra
    r1 = rc-ra
    r2 = rc-rb
    
    #print ra
    #print rb
    
    #if ra[0]==rc[0]:
    
        #v[0]=0
        #v[1]=0
        #v[2]=1/(ra[1]-rb[1])
        ##print v[2]
    
    #else:
  
    
  
    r1r2=numpy.cross(r1,r2)
    r1r2_abs=numpy.linalg.norm(r1r2)
    
    if r1r2_abs <= 0.00001:
        v[0]=0.0
        v[1]=0.0
        v[2]=0.0
    else:
        
        r1mr2=((r1/numpy.linalg.norm(r1)) - (r2/numpy.linalg.norm(r2)))
        ror1mr2= numpy.dot(ro,r1mr2)
        
        v= r1r2/(r1r2_abs)**2 *(ror1mr2)
    
        
    v=0.25/numpy.pi*v
    #print v

        

    return v


        


    


    
    
    
    
    
def downwash_horshoe(ra,rb,r,sl):

    #print ra
    rinf1=numpy.empty(3)
    rinf2=numpy.empty(3)

    rinf1[0]=ra[0]+100*sl
    rinf1[1]=ra[1]
    rinf1[2]=ra[2]
    
    rinf2[0]=rb[0]+100*sl
    rinf2[1]=rb[1]
    rinf2[2]=rb[2]   
    
    v1=downwash_segment(rinf1,ra,r)
    v2=downwash_segment2(ra,rb,r)
    v3=downwash_segment(rb,rinf2,r)
    
    vres= v1 + v2 + v3
    
    
    
    return vres
    



#--compute downwash for a wing ------------------------------------------

def compute_downwash(ra,rb,ra_sym,rb_sym,r,sl,num_count,panel_norm,panel_norm2):
    #alpha_tc= 0*numpy.pi/180
    vel=numpy.empty([num_count,num_count])
    vel1=numpy.empty([num_count,num_count])
    vel2=numpy.empty([num_count,num_count])
    for i in range(0,num_count):

        for j in range(0,num_count):    
            
            p_n=panel_norm[i]
            #p_n[0]=panel_norm[i,0]
            #p_n[1]=panel_norm[i,1]
            #p_n[2]=panel_norm[i,2]
            
            vl1=downwash_horshoe(ra[j],rb[j],r[i],sl[0])
            vel1=numpy.dot(vl1,p_n)
            #vel1[i,j]=-vl1[2]
            
            
            #vel1[i,j]=0.25/numpy.pi*vel1[i,j]
            
            
            vl2=downwash_horshoe(ra_sym[j],rb_sym[j],r[i],sl[0])
            vel2=numpy.dot(vl2,p_n)
            #vel2[i,j]=-vl2[2]
            
            #vel2[i,j]=0.25/numpy.pi*vel2[i,j] 
            
            vel[i,j]=vel1-vel2
            
    
    return vel




#--compute downwash for a wing ------------------------------------------

def compute_downwash2(ra,rb,ra_sym,rb_sym,r,sl,num_count,panel_norm,panel_norm2):
    #alpha_tc= 0*numpy.pi/180
    vel=numpy.empty([num_count,num_count])
    vel1=numpy.empty([num_count,num_count])
    vel2=numpy.empty([num_count,num_count])
    for i in range(0,num_count):

        for j in range(0,num_count):    
            
            p_n=panel_normal[i]
            vl1=downwash_horshoe(ra[j],rb[j],r[i],sl[0])
            #vel1=numpy.dot(vl1,p_n)
            #vel1[i,j]=-vl1[2]
            
            
            #vel1[i,j]=0.25/numpy.pi*vel1[i,j]
            
            p_n2=panel_norm2[i]
            vl2=downwash_horshoe(ra_sym[j],rb_sym[j],r[i],sl[0])
            #vel1=numpy.dot(vl2,p_n)
            #vel2[i,j]=-vl2[2]
            
            #vel2[i,j]=0.25/numpy.pi*vel2[i,j] 
            
            vel=vl1-vl2
            
    
    return vel





#------------------RHS   compute-----------------------------------

def compute_RHS(Vinf,alpha_tc,num_count,twist_distri,panel_norm,Vinf_v,panel_norm2):
    #alpha_tc= 0*numpy.pi/180
    RHS=numpy.empty([num_count,1])
    for i in range(0,num_count):

        p_n=panel_norm[i]
        RHS[i,0]=-1*numpy.dot(Vinf_v,p_n)
        #RHS[i,0]=Vinf*numpy.sin(alpha_tc+twist_distri[i])  
 
    
    return RHS



def vlm_3d(ra,rb,r,ra_sym,rb_sym,r2,sl,num_count,Vinf,alpha_tc,Sref,deltaxx,y,twist_distri,panel_norm,Vinf_v,aoa,panel_norm2):
    v= numpy.empty([num_count,3])
    Veff= numpy.empty([num_count,3])
    F= numpy.empty([num_count,3])
    Lfi=numpy.empty(num_count)
    Lfk=numpy.empty(num_count)
    
    Lft=numpy.empty(num_count)
    Dg=numpy.empty(num_count)
    L=numpy.empty(num_count)
    D=numpy.empty(num_count)  
    Gamma=numpy.empty([num_count,3]) 
    dv=numpy.empty(3) 
    
#---downwash computation-----------------------------------------    
    A=compute_downwash(ra,rb,ra_sym,rb_sym,r,sl,num_count,panel_norm,panel_norm2)
    
    
    RHS=compute_RHS(Vinf,alpha_tc,num_count,twist_distri,panel_norm,Vinf_v,panel_norm2)
   
    T=numpy.linalg.solve(A,RHS)
    
    for i in range(0,num_count):
        v[i,0]=0.0
        v[i,1]=0.0
        v[i,2]=0.0
        for j in range(0,num_count):
            dv= T[j]*(downwash_horshoe(ra[j],rb[j],r2[i],sl[0])-downwash_horshoe(ra_sym[j],rb_sym[j],r2[i],sl[0])) 
                        
            v[i]=v[i]+dv
            #v[i,0]=v[i,0]+dv[0]
            #v[i,1]=v[i,1]+dv[1]
            #v[i,2]=v[i,2]+dv[2] 
        #print v[i]   
        Veff[i]= Vinf_v +v[i]
        #print v[i]
        
        #Gamma[i] =T[i]*(rb[i]-ra[i])
        Gamma[i,0] =T[i]*(rb[i,0]-ra[i,0])
        Gamma[i,1] =T[i]*(rb[i,1]-ra[i,1])
        Gamma[i,2] =T[i]*(rb[i,2]-ra[i,2])
        
        F[i]=numpy.cross(Veff[i],Gamma[i])
        #print F[i]
        
        
    LT=0.0
    DT=0.0
    arsec=0.0 
    for i in range(0,num_count):

        #L[i]=deltaxx[i]*Lft[i]   #T(i)*v(i)*sin(alpha)
        #D[i]=deltaxx[i]*Dg[i]    
        
        #L[i]=deltaxx[i]*F[i,2]   #T(i)*v(i)*sin(alpha)
        #D[i]=deltaxx[i]*F[i,0]   
        
        L[i]=F[i,2]*numpy.cos(aoa)+F[i,0]*numpy.sin(aoa)   #T(i)*v(i)*sin(alpha)
        D[i]=F[i,0]*numpy.cos(aoa)-F[i,2]*numpy.sin(aoa)         
        
        #L[i]=F[i,2]   #T(i)*v(i)*sin(alpha)
        #D[i]=F[i,0]           
        
        
        
        LT=LT+L[i]
        DT=DT+D[i]
        
    
    Cl=2*LT/(0.5*Vinf**2*Sref)
    
    Cd=2*DT/(0.5*Vinf**2*Sref)   
    
    ##print Lft
    #plt.plot(y,Lft,'-o',linewidth = 2.5, markersize=5, markerfacecolor='None')
    #plt.xlabel(r'$spanwise coord$', fontsize=24)
    #plt.ylabel(r'$Lift$', fontsize=24)
    #plt.show()
    #plt.savefig('wing_lift_distri_3d.png',format='png')      
  
    return Cl,Cd    



  
    

    
    
    
    
    
    
    
    
         
        
        







#--------wing  split geometry -------------------------------------------------------------
def wing_geometry_3d(nn,no_of_sections,section_span,section_sweep,section_taper,root_chord,section_rt,section_tt,section_angle,wing_st):
    
    
    #----overall geometry-------------------------------
    #root_chord = 40
    #span  = 200  #overall
    n =  nn#10    #number of panels for each section
    #------
    #----inputs required---------------------------------------------------------
    #no_of_sections=2   
    #section_span=numpy.empty(no_of_sections)
    #section_sweep=numpy.empty(no_of_sections)
    #section_taper=numpy.empty(no_of_sections)
    section_starting_coord=numpy.empty(no_of_sections)
    section_starting_coord_z=numpy.empty(no_of_sections)
    section_rc=numpy.empty(no_of_sections)
    section_tc=numpy.empty(no_of_sections)
    dchord=numpy.empty(no_of_sections)
    deltax=numpy.empty(no_of_sections)

    #--------------------------------------------------------------------------------------------

    #section_span[0]=100
    #section_span[1]=100
    
    #section_sweep[0]=numpy.pi/180*20
    #section_sweep[1]=numpy.pi/180*30
    
    #section_taper[0]=0.5   
    #section_taper[1]=0.3

    

    #--------------getting the starting locations of each section----------------------------------
    num_count=n*no_of_sections
    section_starting_coord[0]=wing_st #0.0
    section_starting_coord_z[0]=0.0
    section_rc[0]=root_chord
    section_tc[0]=section_taper[0]*section_rc[0]
    dchord[0]=(section_rc[0]-section_tc[0])
    deltax[0]=section_span[0]/n
    
    for i in range(1,no_of_sections):
        section_starting_coord[i]= section_starting_coord[i-1] + section_span[i-1]*numpy.tan(section_sweep[i-1])
        section_starting_coord_z[i] = section_starting_coord_z[i-1] + section_span[i-1]*numpy.sin(section_angle[i-1])    
        
        section_rc[i]=section_tc[i-1]
        section_tc[i]=section_taper[i]*section_rc[i]
        dchord[i]=(section_rc[i]-section_tc[i])
        deltax[i]=section_span[i]/n
        
    #-----------------------------------panel each section------------------------------------------------------
    #print section_starting_coord_z
    #print section_rc
    #print section_tc
    
    #--------------------initialization for each section--------------------------------------------------------
    section_length= numpy.empty(n*no_of_sections)
    area_section=numpy.empty(n*no_of_sections)
    sl=numpy.empty(n*no_of_sections)
    xpos=numpy.empty(n*no_of_sections)
    
    ya=numpy.empty(n*no_of_sections)
    yb=numpy.empty(n*no_of_sections)
    xa=numpy.empty(n*no_of_sections)
    xb=numpy.empty(n*no_of_sections)
    xaa=numpy.empty(n*no_of_sections)
    yab=numpy.empty(n*no_of_sections)
    ybb=numpy.empty(n*no_of_sections)
    y2=numpy.empty(n*no_of_sections)
    

    za=numpy.empty(n*no_of_sections)
    zb=numpy.empty(n*no_of_sections)
    z=numpy.empty(n*no_of_sections)    
    z_lead=numpy.empty(n*no_of_sections)
    z_trail=numpy.empty(n*no_of_sections)      
    
    
    r=numpy.empty([n*no_of_sections,3]) 
    r_sym=numpy.empty([n*no_of_sections,3]) 
    r2=numpy.empty([n*no_of_sections,3]) 
    ra=numpy.empty([n*no_of_sections,3]) 
    rb=numpy.empty([n*no_of_sections,3]) 
    ra_sym=numpy.empty([n*no_of_sections,3]) 
    rb_sym=numpy.empty([n*no_of_sections,3])    
    
    
    x=numpy.empty(n*no_of_sections)
    y=numpy.empty(n*no_of_sections)
    twist_distri=numpy.empty(n*no_of_sections)
    xloc_leading=numpy.empty(n*no_of_sections)
    xloc_trailing=numpy.empty(n*no_of_sections)      
    deltaxx=numpy.empty(n*no_of_sections) 
    panel_norm=numpy.empty([n*no_of_sections,3]) 
    panel_norm2=numpy.empty([n*no_of_sections,3]) 
    #print section_rt
    #print section_tt
    #----------------------------------------------------------------------------------------------
    #--discretizing the wing sections into panels--------------------------------
    cum_span=0   #cumulative span
    for j in range(0,no_of_sections):
        
        for i in range(0,n):
        
            section_length[n*j+i]= dchord[j]/section_span[j]*(section_span[j]-(i+1)*deltax[j]+deltax[j]/2) + section_tc[j]
            
            area_section[n*j+i]=section_length[n*j+i]*deltax[j]
            sl[n*j+i]=section_length[n*j+i]
            
            twist_distri[n*j+i]=section_rt[j]+ i/float(n)*(section_tt[j]-section_rt[j])
            #xpos[n*j+i]=(i)*deltax
            
            #ya[n*j+i]=cum_span + (i)*deltax[j]
            #yb[n*j+i]=cum_span + (i+1)*deltax[j]
            
            
            ya[n*j+i]=cum_span + ((i)*deltax[j])*numpy.cos(section_angle[j])
            yb[n*j+i]=cum_span + ((i+1)*deltax[j])*numpy.cos(section_angle[j])            
            
            xloc_leading[n*j+i]=section_starting_coord[j] + ((i+1)*deltax[j]-deltax[j]/2)*numpy.tan(section_sweep[j])
            xloc_trailing[n*j+i]=section_starting_coord[j] + ((i+1)*deltax[j]-deltax[j]/2)*numpy.tan(section_sweep[j]) + sl[n*j+i]*numpy.cos(twist_distri[n*j+i])
  
            xa[n*j+i]=section_starting_coord[j] + ((i+1)*deltax[j]-deltax[j]/2)*numpy.tan(section_sweep[j])+ (0.25*sl[n*j+i])*numpy.cos(twist_distri[n*j+i])
            
            #xa[n*j+i]=section_starting_coord[j] + ((i+1)*deltax[j]-deltax[j]/2)*numpy.tan(section_sweep[j])+ (0.25*sl[n*j+i])#*numpy.cos(twist_distri[n*j+i])
            #xb[n*j+i]=section_starting_coord[j] + ((i+1)*deltax[j]-deltax[j]/2)*numpy.tan(section_sweep[j])+ (0.25*sl[n*j+i])#*numpy.cos(twist_distri[n*j+i])
            
            #xa[n*j+i]=section_starting_coord[j] + ((i)*deltax[j])*numpy.tan(section_sweep[j])+ (0.25*sl[n*j+i])#*numpy.cos(twist_distri[n*j+i])
            #xb[n*j+i]=section_starting_coord[j] + ((i+1)*deltax[j])*numpy.tan(section_sweep[j])+ (0.25*sl[n*j+i])#*numpy.cos(twist_distri[n*j+i])
            #xaa[n*j+i]=section_starting_coord[j] + ((i+1)*deltax[j]-deltax[j]/2)*numpy.tan(section_sweep[j])+ (0.25*sl[n*j+i])#*numpy.cos(twist_distri[n*j+i])
            
            
            
            
            za[n*j+i]= section_starting_coord_z[j] + (i)*deltax[j]*numpy.sin(section_angle[j])-(0.25*sl[n*j+i])*numpy.sin(twist_distri[n*j+i])
            zb[n*j+i]= section_starting_coord_z[j] + (i+1)*deltax[j]*numpy.sin(section_angle[j])-(0.25*sl[n*j+i])*numpy.sin(twist_distri[n*j+i])
            z[n*j+i]=  section_starting_coord_z[j] + (i+0.5)*deltax[j]*numpy.sin(section_angle[j])-(0.75*sl[n*j+i])*numpy.sin(twist_distri[n*j+i])
            
            #za[n*j+i]= section_starting_coord_z[j] + (i)*deltax[j]*numpy.sin(section_angle[j])#-(0.25*sl[n*j+i])*numpy.sin(twist_distri[n*j+i])
            #zb[n*j+i]= section_starting_coord_z[j] + (i+1)*deltax[j]*numpy.sin(section_angle[j])#-(0.25*sl[n*j+i])*numpy.sin(twist_distri[n*j+i])
            #z[n*j+i]=  section_starting_coord_z[j] + (i+0.5)*deltax[j]*numpy.sin(section_angle[j])-(0.5*sl[n*j+i])*numpy.sin(twist_distri[n*j+i])            
            
            z_lead[n*j+i]= section_starting_coord_z[j] + (i+0.5)*deltax[j]*numpy.sin(section_angle[j])-(0*sl[n*j+i])*numpy.sin(twist_distri[n*j+i])
            z_trail[n*j+i]= section_starting_coord_z[j] + (i+0.5)*deltax[j]*numpy.sin(section_angle[j])-(1.0*sl[n*j+i])*numpy.sin(twist_distri[n*j+i])
                       
            
            
            #print za[n*j+i]
            
            x[n*j+i]=section_starting_coord[j] + ((i+1)*deltax[j]-deltax[j]/2)*numpy.tan(section_sweep[j]) + (0.75*sl[n*j+i])*numpy.cos(twist_distri[n*j+i])
            
            #x[n*j+i]=section_starting_coord[j] + ((i+1)*deltax[j]-deltax[j]/2)*numpy.tan(section_sweep[j]) + (0.25*sl[n*j+i])+(0.5*sl[n*j+i])*numpy.cos(twist_distri[n*j+i])
            #y[n*j+i]=cum_span + ((i+1)*deltax[j]-deltax[j]/2)
            y[n*j+i]=cum_span + (((i+1)*deltax[j]-deltax[j]/2))*numpy.cos(section_angle[j])
            
            
            deltaxx[n*j+i]= deltax[j]
            #print  twist_distri[n*j+i]
            
            
           #--------------------- vectorization--------------------------------------------
            
            r[n*j+i,0]=x[n*j+i]
            r[n*j+i,1]=y[n*j+i]
            r[n*j+i,2]=z[n*j+i]
 
            ra[n*j+i,0]=xa[n*j+i]  
            ra[n*j+i,1]=ya[n*j+i]
            ra[n*j+i,2]=za[n*j+i]
            
            rb[n*j+i,0]=xa[n*j+i]  
            rb[n*j+i,1]=yb[n*j+i]
            rb[n*j+i,2]=zb[n*j+i] 

            ra_sym[n*j+i,0]=xa[n*j+i]  
            ra_sym[n*j+i,1]=-ya[n*j+i]
            ra_sym[n*j+i,2]=za[n*j+i]
            
            rb_sym[n*j+i,0]=xa[n*j+i]  
            rb_sym[n*j+i,1]=-yb[n*j+i]
            rb_sym[n*j+i,2]=zb[n*j+i]            
            
 
            r2[n*j+i,0]=xa[n*j+i]
            r2[n*j+i,1]=0.5*(ya[n*j+i]+yb[n*j+i])
            r2[n*j+i,2]=0.5*(za[n*j+i]+zb[n*j+i])            
            
            #print r2[n*j+i]
            #print za          
            
            #--------------------- vectorization---------------------------------------------
            
            pn= panel_normal(ra[n*j+i],rb[n*j+i],r[n*j+i])
            
            panel_norm[n*j+i,0]=pn[0]
            panel_norm[n*j+i,1]=pn[1]
            panel_norm[n*j+i,2]=pn[2]
            
            
            pn2= panel_normal(ra_sym[n*j+i],rb_sym[n*j+i],r[n*j+i])
            
            panel_norm2[n*j+i,0]=pn2[0]
            panel_norm2[n*j+i,1]=pn2[1]
            panel_norm2[n*j+i,2]=pn2[2]            
            
            
            #xloc_leading[n*j+i]=section_starting_coord[j] + ((i+1)*deltax[j]-deltax[j]/2)*numpy.tan(section_sweep[j])
            #xloc_trailing[n*j+i]=section_starting_coord[j] + ((i+1)*deltax[j]-deltax[j]/2)*numpy.tan(section_sweep[j]) + sl[n*j+i]*numpy.cos(twist_distri[n*j+i])
            #deltaxx[n*j+i]= deltax[j]
            ##print  twist_distri[n*j+i]
            
     
            
        cum_span=cum_span+section_span[j]*numpy.cos(section_angle[j])
   
        ###plotting the lift distribution
        
        
    #plt.plot(y,xloc_leading,'-',linewidth = 2.5, markersize=5, markerfacecolor='None')
    #plt.plot(y,xloc_trailing,'-',linewidth = 2.5, markersize=5, markerfacecolor='None')
    #plt.plot(y,xa,'o',linewidth = 2.5, markersize=5, markerfacecolor='None')
    #plt.plot(y,x,'*',linewidth = 2.5, markersize=5, markerfacecolor='None')
    #plt.xlabel(r'$spanwise coord$', fontsize=24)
    #plt.ylabel(r'$chord distribution$', fontsize=24)
    ##plt.show()
    #plt.savefig('wing_geometry_3d.png',format='png')    
    #plt.clf()

    
    
    
    
    #plt.plot(y,z,'-o',linewidth = 2.5, markersize=5, markerfacecolor='None')
    ##plt.plot(y,xloc_trailing,'-',linewidth = 2.5, markersize=5, markerfacecolor='None')
    ##plt.plot(y,xa,'o',linewidth = 2.5, markersize=5, markerfacecolor='None')
    ##plt.plot(y,x,'*',linewidth = 2.5, markersize=5, markerfacecolor='None')
    #plt.xlabel(r'$spanwise coord$', fontsize=24)
    #plt.ylabel(r'$height$', fontsize=24)
    ##plt.show()
    #plt.savefig('wing_geometry_3d_xyplane.png',format='png')       
    #plt.clf()
   
   
   

    
    
    #fig = pylab.figure()
    #ax = Axes3D(fig)
    ##ax.scatter(y, z, x)
    ##ax.scatter(y,z,xloc_leading)
    ##ax.scatter(y,z,xloc_trailing)
    
    #ax.scatter(x,y,z)
    #ax.scatter(xloc_leading,y,z)
    #ax.scatter(xloc_trailing,y,z)    
    ##Axes3D.plot_wireframe(X, Y, Z, *args, **kwargs)¶
    #plt.savefig('wing_geometry_3d_geom.png',format='png') 
    ##plt.show()   
    ##plt.clf()
    
 
    

    
  
   
   
    #print xa
    return x,y,xa,ya,yb,num_count,deltaxx,twist_distri,r,ra,rb,r2,ra_sym,rb_sym,sl,xloc_leading,xloc_trailing,z,panel_norm,za,zb,z_lead,z_trail,panel_norm2
  
















        
        

def wing_lift(a_o_a,npanels):    
#def wing_lift(a_o_a,ar):
#def wing_lift(Wing):

    gamma=1.4
    R=287.0
    n = npanels #10
    nn = 1    
    Vinf_v=numpy.empty(3)
    #span = 35.66                     #half span
    #root_chord = 7.5
    #n =10
    #nn = 1
    #sweep = 30.0*numpy.pi/180.0
    #taper = 0.15
    #tip_cord = root_chord*taper
    #alpha_rc = 0.0*numpy.pi/180.0
    #alpha_tc = a_o_a*numpy.pi/180.0
    #Mc = 0.8
    #Tinf=218.0
    #Vinf = Mc*numpy.sqrt(1.4*287*Tinf)
    #Sref = 0.5*span*(root_chord+tip_cord)*2
    #rho =  1.12
    #lex= 0 #0.16    #fraction of root chord
    #tex= 0 #0.2    #fraction of tip chord
    #ex_loc= 0.3    #fraction of span
    
    #wing_dihedral = 0 #10.0*numpy.pi/180   #angle  degrees
    #wing_dihedral2 = 0 #30.0*numpy.pi/180

    
    #root_twist=0 #5.0*numpy.pi/180
    #break_twist=0 #2.0*numpy.pi/180
    #tip_twist=0 #-1.0*numpy.pi/180
    
    
    
    #------------tornado tstcase-----------------
    #Sref=1500
    #span=numpy.sqrt(ar*Sref)
    
    
    ##span = 117.2
    #root_chord = Sref/span    #23.622    
    #span=span/2.0
    #sweep = 0.0  #26.0*numpy.pi/180.0
    #taper = 1 #0.15
    #tip_cord = root_chord*taper
    #alpha_rc = 0.0*numpy.pi/180.0
    #alpha_tc = a_o_a*numpy.pi/180.0
    #Mc = 0.01
    #Tinf=218.0
    #Vinf = Mc*numpy.sqrt(1.4*287*Tinf)
    ##Sref = 0.5*span*(root_chord+tip_cord)*2
    #rho =  1.12
    #lex= 0.0    #fraction of root chord
    #tex= 0.0    #fraction of tip chord
    #ex_loc= 0.33 #0.33    #fraction of span

    #root_twist=0.0*numpy.pi/180
    #break_twist=0.00*numpy.pi/180
    #tip_twist= 0.0*numpy.pi/180    
    
    
    ##root_twist=5.0*numpy.pi/180
    ##break_twist=2.0*numpy.pi/180
    ##tip_twist= -1.0*numpy.pi/180
    
    
    #wing_dihedral = 0.0*numpy.pi/180   #angle  degrees
    #wing_dihedral2 =0.0 *numpy.pi/180  #30.0*numpy.pi/180    
    
    
    
    
    #----B737 test case------------------------------------------------------
    
    
    #---main wing-------------------------------
    
    span = 117.2
    root_chord = 23.622    
    sweep = 0.0*numpy.pi/180.0
    taper = 0.15
    tip_cord = root_chord*taper
    alpha_rc = 0.0*numpy.pi/180.0
    alpha_tc = a_o_a*numpy.pi/180.0
    Mc = 0.01
    Tinf=218.0
    Vinf = Mc*numpy.sqrt(1.4*287*Tinf)
    Sref = 0.5*span*(root_chord+tip_cord)*2
    rho =  1.12
    lex= 0.0    #fraction of root chord
    tex= 0.0   #fraction of tip chord
    ex_loc= 0.5 #0.33    #fraction of span

    root_twist=5.0*numpy.pi/180
    break_twist=2.5*numpy.pi/180
    tip_twist= 0.0*numpy.pi/180    
    
    Vinf_v[0]=Vinf*numpy.cos(alpha_tc)
    Vinf_v[1]=0
    Vinf_v[2]=Vinf*numpy.sin(alpha_tc)
    #root_twist=3.0*numpy.pi/180
    #break_twist=2.0*numpy.pi/180
    #tip_twist= -1.0*numpy.pi/180
    
    
    wing_dihedral = 0.0*numpy.pi/180.0   #angle  degrees
    wing_dihedral2 =0.0*numpy.pi/180.0  #30.0*numpy.pi/180       
    
    
    
    
    
    rc_with_ext=root_chord*(1+lex+tex)
    no_of_sections=2
    section_span=numpy.empty(no_of_sections)
    section_sweep=numpy.empty(no_of_sections)
    section_taper=numpy.empty(no_of_sections)
    section_rt=numpy.empty(no_of_sections)
    section_tt=numpy.empty(no_of_sections)
    section_angle=numpy.empty(no_of_sections)
    
    section_span[0]=ex_loc*span
    section_span[1]=(1-ex_loc)*span
    
    #computing the 
    
    d_chord_cw = root_chord*(1-taper)
    break_cord=d_chord_cw/span*(span-ex_loc*span) + tip_cord
    taper_ext=break_cord/rc_with_ext
    
    
    section_sweep[0]=numpy.arctan((root_chord*lex +  ex_loc*span*numpy.tan(sweep))/(ex_loc*span))
    section_sweep[1]=sweep
    
   
    
    section_taper[0]=break_cord/rc_with_ext
    section_taper[1]=tip_cord/break_cord
    
    section_rt[0]=root_twist
    section_rt[1]=break_twist
    
    section_tt[0]=break_twist
    section_tt[1]=tip_twist    
    
    section_angle[0]=wing_dihedral
    section_angle[1]=wing_dihedral2   
    
    wing_starting = 0.0

    
    
    #[x,y,xa,ya,yb,num_count,deltaxx,twist_distri,r,ra,rb,r2,ra_sym,rb_sym,sl]=wing_geometry(n,no_of_sections,section_span,section_sweep,section_taper,rc_with_ext,section_rt,section_tt)
    
    [x,y,xa,ya,yb,num_count,deltaxx,twist_distri,r,ra,rb,r2,ra_sym,rb_sym,sl,xloc_leading,xloc_trailing,z,panel_norm,za,zb,z_lead,z_trail,panel_norm2]=wing_geometry_3d(n,no_of_sections,section_span,section_sweep,section_taper,rc_with_ext,section_rt,section_tt,section_angle,wing_starting)
    
    #print x
    #print panel_norm
    #print twist_distri
    ##---horz tail-------------------------------
        
    #span_h = 27.2
    #root_chord_h = 13.622    
    #sweep_h = 36.0*numpy.pi/180.0
    #taper_h = 0.15
    #tip_cord_h = root_chord_h*taper_h
    #alpha_rc_h = 0.0*numpy.pi/180.0
    #alpha_tc_h = a_o_a*numpy.pi/180.0
    #Mc = 0.01
    #Tinf=218.0
    #Vinf = Mc*numpy.sqrt(1.4*287*Tinf)
    #Sref_h = 0.5*span_h*(root_chord_h+tip_cord_h)*2
    #rho =  1.12
    #lex_h= 0.0    #fraction of root chord
    #tex_h= 0.0    #fraction of tip chord
    #ex_loc_h= 0.5 #0.33    #fraction of span

    #root_twist_h=0.0*numpy.pi/180
    #break_twist_h=0.0*numpy.pi/180
    #tip_twist_h= 0.0*numpy.pi/180    
    
    
    ##root_twist=3.0*numpy.pi/180
    ##break_twist=2.0*numpy.pi/180
    ##tip_twist= -1.0*numpy.pi/180
    
    
    #wing_dihedral_h = 49.0*numpy.pi/180   #angle  degrees
    #wing_dihedral2_h =49.0 *numpy.pi/180  #30.0*numpy.pi/180       
    
    
    
    
    
    #rc_with_ext_h=root_chord_h*(1+lex_h+tex_h)
    #no_of_sections_h=2
    #section_span_h=numpy.empty(no_of_sections_h)
    #section_sweep_h=numpy.empty(no_of_sections_h)
    #section_taper_h=numpy.empty(no_of_sections_h)
    #section_rt_h=numpy.empty(no_of_sections_h)
    #section_tt_h=numpy.empty(no_of_sections_h)
    #section_angle_h=numpy.empty(no_of_sections_h)
    
    #section_span_h[0]=ex_loc_h*span_h
    #section_span_h[1]=(1-ex_loc_h)*span_h
    
    ##computing the 
    
    #d_chord_cw_h = root_chord_h*(1-taper_h)
    #break_cord_h=d_chord_cw_h/span_h*(span_h-ex_loc_h*span_h) + tip_cord_h
    #taper_ext_h=break_cord_h/rc_with_ext_h
    
    
    #section_sweep_h[0]=numpy.arctan((root_chord_h*lex_h +  ex_loc_h*span_h*numpy.tan(sweep_h))/(ex_loc_h*span_h))
    #section_sweep_h[1]=sweep_h
    
   
    
    #section_taper_h[0]=break_cord_h/rc_with_ext_h
    #section_taper_h[1]=tip_cord_h/break_cord_h
    
    #section_rt_h[0]=root_twist_h
    #section_rt_h[1]=break_twist_h
    
    #section_tt_h[0]=break_twist_h
    #section_tt_h[1]=tip_twist_h    
    
    #section_angle_h[0]=wing_dihedral_h
    #section_angle_h[1]=wing_dihedral2_h   
    
    
    #wing_starting_h = 90.0
    
    
    ##[x,y,xa,ya,yb,num_count,deltaxx,twist_distri,r,ra,rb,r2,ra_sym,rb_sym,sl]=wing_geometry(n,no_of_sections,section_span,section_sweep,section_taper,rc_with_ext,section_rt,section_tt)
    
    #[x_h,y_h,xa_h,ya_h,yb_h,num_count_h,deltaxx_h,twist_distri_h,r_h,ra_h,rb_h,r2_h,ra_sym_h,rb_sym_h,sl_h,xloc_leading_h,xloc_trailing_h,z_h,panel_norm_h,za_h,zb_h,z_lead_h,z_trail_h]=wing_geometry_3d(n,no_of_sections_h,section_span_h,section_sweep_h,section_taper_h,rc_with_ext_h,section_rt_h,section_tt_h,section_angle_h,wing_starting_h)
        
    ##print x_h
    ##------------------------------appending all the wings----------------------------------
    #x = numpy.concatenate([x,x_h])
    #y = numpy.concatenate([y,y_h])
    #xa = numpy.concatenate([xa,xa_h])
    #ya = numpy.concatenate([ya,ya_h])
    #yb = numpy.concatenate([yb,yb_h])
    #num_count = num_count+num_count_h #numpy.concatenate([num_count,num_count_h])
    #deltaxx = numpy.concatenate([deltaxx,deltaxx_h])
    #twist_distri = numpy.concatenate([twist_distri,twist_distri_h])
    #r = numpy.concatenate([r,r_h])
    #ra= numpy.concatenate([ra,ra_h])
    #rb = numpy.concatenate([rb,rb_h])
    #r2 = numpy.concatenate([r2,r2_h])
    #ra_sym = numpy.concatenate([ra_sym,ra_sym_h])
    #rb_sym = numpy.concatenate([rb_sym,rb_sym_h])
    #sl = numpy.concatenate([sl,sl_h])
    
    #xloc_leading = numpy.concatenate([xloc_leading,xloc_leading_h])
    #xloc_trailing = numpy.concatenate([xloc_trailing,xloc_trailing_h])
    #z = numpy.concatenate([z,z_h])    
      
    #panel_norm = numpy.concatenate([panel_norm,panel_norm_h]) 
    #za = numpy.concatenate([za,za_h]) 
    #zb = numpy.concatenate([zb,zb_h]) 
    #z_lead = numpy.concatenate([z_lead,z_lead_h]) 
    #z_trail  = numpy.concatenate([z_trail,z_trail_h])   
      
      
        
   ### numpy.concatenate([a,b])
    
    
    
    

    #-------------------------------------------------------------------------
    
    
    [Cl,Cd]=vlm_3d(ra,rb,r,ra_sym,rb_sym,r2,sl,num_count,Vinf,alpha_tc,Sref,deltaxx,y,twist_distri,panel_norm,Vinf_v,alpha_tc,panel_norm2) 
    
    
    

    #plt.plot(y,xloc_leading,'-',linewidth = 2.5, markersize=5, markerfacecolor='None')
    #plt.plot(y,xloc_trailing,'-',linewidth = 2.5, markersize=5, markerfacecolor='None')
    #plt.plot(y,x,'*',linewidth = 2.5, markersize=5, markerfacecolor='None')
    #plt.plot(ya,xa,'o',linewidth = 2.5, markersize=5, markerfacecolor='None')
    ##plt.plot(yb,xa,'*',linewidth = 2.5, markersize=5, markerfacecolor='None')
    #plt.xlabel(r'$spanwise coord$', fontsize=24)
    #plt.ylabel(r'$chord distribution$', fontsize=24)
    ##plt.show()
    #plt.savefig('wing_geometry_3d.png',format='png')    
    #plt.clf()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    fig = pylab.figure()
    ax = Axes3D(fig)
    #ax.scatter(y, z, x)
    #ax.scatter(y,z,xloc_leading)
    #ax.scatter(y,z,xloc_trailing)
    
    ax.scatter(x,y,z,'-*')
    ax.scatter(xloc_leading,y,z_lead)
    ax.scatter(xa,ya,za)
    #ax.scatter(xa,yb,zb)
    ax.scatter(xloc_trailing,y,z_trail)    
    #Axes3D.plot_wireframe(X, Y, Z, *args, **kwargs)¶
    #ax.set_aspect('equal','box')

    plt.savefig('wing_geometry_3d_geom.png',format='png') 
    plt.show()   
    #plt.clf()
    
    #print x    
    
    return Cl,Cd
    #print 'Cl ',Cl
    #print 'Cd ',Cd

  
    
    
    
    
#wing_lift()
    
    
#------------testing the module---------------------------------------


#--------------cl vs alpha module----------------------------------

npanels = 12
cl=numpy.empty(6)
cd=numpy.empty(6)
a_o_a=numpy.empty(6)


for i in range(0,6):
    a_o_a[i]=i
    aoa=a_o_a[i]
    [cl[i],cd[i]]=wing_lift(aoa,npanels)
    print cl[i]
    
    

print a_o_a
print cl
print cd
#plt.plot(a_o_a,cl,'-',linewidth = 2.5, markersize=5, markerfacecolor='None')
#plt.xlabel(r'$alpha$', fontsize=24)
#plt.ylabel(r'$cl$', fontsize=24)
##plt.show()
#plt.savefig('cl_alpha.png',format='png')    
#plt.clf()


#--------------individual call----------------------------------

#aoa_1 = 1
#npanels = 2
#cl_l=wing_lift(aoa_1,npanels)
#print cl_l



#--------------cl vs alpha module tornado testcase----------------------------------


#cl=numpy.empty(2)
#cd=numpy.empty(2)
#ar=numpy.empty(15)

#dcl_dalpha=numpy.empty(15)
#dcl_dalpha_rad=numpy.empty(15)

#a_o_a=numpy.empty(2)
#a_o_a[0]=1
#a_o_a[1]=5


#for j in range(0,15):
    #ar[j]=j+1
    #for i in range(0,2):
        ##a_o_a[i]=i+1
        #aoa=a_o_a[i]
        #[cl[i],cd[i]]=wing_lift(aoa,ar[j])
        ##print cl[i]
    
    #print a_o_a[1]-a_o_a[0]    
    #dcl_dalpha[j]=(cl[1]-cl[0])/(a_o_a[1]-a_o_a[0])
    #dcl_dalpha_rad[j]=(cl[1]-cl[0])/((a_o_a[1]-a_o_a[0])*numpy.pi/180)     
    
    
    
#print ar
#print dcl_dalpha
#print dcl_dalpha_rad



        
    

#print a_o_a
#print cl
#plt.plot(a_o_a,cl,'-',linewidth = 2.5, markersize=5, markerfacecolor='None')
#plt.xlabel(r'$alpha$', fontsize=24)
#plt.ylabel(r'$cl$', fontsize=24)
##plt.show()
#plt.savefig('cl_alpha.png',format='png')  






#--------------no of panels vs alpha module----------------------------------

#npanels = 2
#cl=numpy.empty(6)
#cd=numpy.empty(6)
#a_o_a=numpy.empty(6)
#no_panels=numpy.empty(6)


#no_panels[0] = 2
#no_panels[1] = 10
#no_panels[2] = 20
#no_panels[3] = 30
#no_panels[4] = 50
#no_panels[5] = 100

#aoa=3



#for i in range(0,6):
    ##a_o_a[i]=i
    ##aoa=a_o_a[i]
    #[cl[i],cd[i]]=wing_lift(aoa,int(no_panels[i]))
    #print cl[i]
    
    

#print no_panels
#print cl
#plt.plot(no_panels,cl,'-',linewidth = 2.5, markersize=5, markerfacecolor='None')
#plt.xlabel(r'$no panels$', fontsize=24)
#plt.ylabel(r'$cl$', fontsize=24)
##plt.show()
#plt.savefig('convergence.png',format='png')    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #def cross(ra, rb):
    
        #v = numpy.empty(3)
        #v[0] = ra[1]*rb[2] - rb[1]*ra[2]
        #v[1] = ra[0]*rb[2] - rb[0]*ra[2]
        #v[2] = ra[0]*rb[1] - rb[0]*ra[1]
    
        #return v
    
    #def minus(ra,rb):
        #v = numpy.empty(3)
        #v[0]=ra[0]-rb[0]
        #v[1]=ra[1]-rb[1]
        #v[2]=ra[2]-ra[2]
        
        #return v
        
        #def downwash_inf(ra,rb,rc):
        
        
            #ro = rb-ra
            #r1 = rc-ra
            #r2 = rc-rb
            
            
            #r1r2=numpy.cross(r1,r2)
            #r1r2_abs=numpy.linalg.norm(r1r2)
            
            #r1mr2=((r1/numpy.linalg.norm(r1)) - (r2/numpy.linalg.norm(r2)))
            #ror1mr2= numpy.dot(ro,r1mr2)
            
            #v= r1r2/(r1r2_abs)**2 *(ror1mr2)        
        
        
    
