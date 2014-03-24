import numpy
from fm_id import fm_id
from Cp import Cp

#-----------------------turbofan---------------------------------------------------------

def engine_sizing_1d(Turbofan,State):    
#def engine_sizing_1d(Minf,Tinf,pinf,pid,pif,pifn,pilc,pihc,pib,pitn,Tt4,aalpha,FD):

    Minf=State.M
    Tinf=State.T
    pinf=State.p
    pid=Turbofan.diffuser_pressure_ratio
    pif=Turbofan.fan_pressure_ratio
    pifn=Turbofan.fan_nozzle_pressure_ratio
    pilc=Turbofan.lpc_pressure_ratio
    pihc=Turbofan.hpc_pressure_ratio
    pib=Turbofan.burner_pressure_ratio
    pitn=Turbofan.turbine_nozzle_pressure_ratio
    Tt4=Turbofan.Tt4
    aalpha=Turbofan.bypass_ratio
    FD=Turbofan.design_thrust

    #print Minf
    #print Tinf
    #print pinf
    #print pid
    #print pif
    #print pifn
    #print pilc
    #print pihc
    #print pib
    #print pitn
    #print Tt4
    #print aalpha
    #print FD


 #global fm
    
    Tref=288.15
    Pref=1.01325*10**5
   
   
   #input parameters
   
   
    #general properties
    R=287.87 #kj/kgK
    g=9.81#m/s2
    gamma=1.4
    Cpp=gamma*R/(gamma-1)
    
    
    #--AA283-A ch5 problem 1
    
    #freestream conitions
    To=Tinf#216#K
    po=pinf#0.2*10**5#pa
    Mo=Minf#0.85
    Ro=287.87
    mu_o=1.475*10**-5
    #R=Ro
    
    nc_den  =4480                        #nacelle density
       
    
    #flow fractions
    alpha=1
    beta=1
    gammaa=1
    alphac=0
    #pressure ratios
    #polytropic effeciencies
    
    etapold=0.98
    etapolf=0.93
    etapollc=0.91
    etapolhc=0.91
    etapolt=0.93
    etapoltn=0.95
    etapolfn=0.95
    eta_b=0.99
    #-----------------------------------------
    
    #stagnation pressures
    htf=4.3*10**7#J/Kg
    tau_f=htf/(Cpp*To)

    #other quantities
  
  
    md_lc_by_md_hc=1#is 1 if no bleed else specify
    #FD=70000 #design thrust
    M2=0.6  #fan face mach number
    M2_5=0.5 #HPC face Mach number
    HTRf=0.325 #hub to tip ratio fan
    HTRhc=0.2#hub to tip ratio high pressure compressor
    
    
    
    #for fan , turbine and compressors
    #freestream properties
    
    Cpo=Cpp
    ao=numpy.sqrt(Cpo/(Cpo-Ro)*Ro*To)
    uo=Mo*ao
    rhoo=po/(Ro*To)
    #freestream stagnation properties
    
    pto=po*(1+(gamma-1)/2*Mo**2)**(gamma/(gamma-1))
    Tto=To*(1+(gamma-1)/2*Mo**2)
    hto=Cpp*Tto#h(Tto)
    tau_r=Tto/To
    #fan and compressor quantities
    
    #inlet condition
    
    pt1_8=pto*pid
    Tt1_8=Tto*pid**((gamma-1)/(gamma*etapold))
    ht1_8=Cpp*Tt1_8    #h(Tt1_8)
    
    
    #Fan and LPC inlet conditions 
    
    #if no bli
    
    #LPC
    pt1_9=pt1_8
    Tt1_9=Tt1_8
    ht1_9=ht1_8
    
    
    #fan
    pt2=pt1_8
    Tt2=Tt1_8
    ht2=ht1_8
    
    
    #if bli
    
    
    #fan exit conditions
    
    #etapolf=Feta(pif,pif,1,1)  
    #etapolf=1  #if available for engine use that
    
    pt2_1=pt2*pif
    Tt2_1=Tt2*pif**((gamma-1)/(gamma*etapolf))
    ht2_1=Cpp*Tt2_1#h(Tt2_1)
    
    
    #fan nozzle exit conditions
    
    pt7=pt2_1*pifn
    Tt7=Tt2_1*pifn**((gamma-1)/(gamma)*etapolfn)
    ht7=Cpp*Tt7#h(Tt7)
    
    
    #LPC exit conditions
    
    #etapollc=Feta(pilc,pilc,1,1)
    #etapollc=1
    
    pt2_5=pt1_9*pilc
    Tt2_5=Tt1_9*pilc**((gamma-1)/(gamma*etapollc))
    ht2_5=Cpp*Tt2_5  #h(Tt2_5)
    
    
    #HPC exit calculations
    
    #etapolhc=Feta(pihc,pihc,1,1)
    #etapolhc=1
    
    pt3=pt2_5*pihc
    Tt3=Tt2_5*pihc**((gamma-1)/(gamma*etapolhc))
    ht3=Cpp*Tt3#h(Tt3)
    
   
    #cooling mass flow
    
    #to be included in the future
    
    #combustor quantities
    
    #[fb,lambda]=Fb(alpha,beta,gammaa,Tt3,htf,Tt4)
    
    fb=(((Tt4/To)-tau_r*(Tt3/Tt1_9))/(eta_b*tau_f-(Tt4/To)))
    
    f=fb*(1-alphac)  #alphac=0 if cooling mass flow not included
    lmbda=1
    
    #combustor exit conditions
    
    ht4=lmbda*Cpp*Tt4#h(Tt4)
    #sigmat4=lambda*sigma(Tt4)
    pt4=pt3*pib
    
    #Station 4.1 without IGV cooling
    
    Tt4_1=Tt4
    pt4_1=pt4
    lambdad=lmbda
    
    #Station 4.1 with IGV cooling
    
    #complete later
    
    
    #Turbine quantities
    
    #high pressure turbine
    
    deltah_ht=-1/(1+f)*1/0.99*(ht3-ht2_5)
    
    Tt4_5=Tt4_1+deltah_ht/Cpp
    
    pt4_5=pt4_1*(Tt4_5/Tt4_1)**(gamma/((gamma-1)*etapolt))
    ht4_5=Cpp*Tt4_5#h(Tt4_5)
    
    #low pressure turbine
    
    deltah_lt=-1/(1+f)*md_lc_by_md_hc*(1/0.99*(ht2_5-ht1_9)+aalpha*1/0.99*(ht2_1-ht2))
    
    Tt4_9=Tt4_5+deltah_lt/Cpp
    
    pt4_9=pt4_5*(Tt4_9/Tt4_5)**(gamma/((gamma-1)*etapolt))
    ht4_9=Cpp*Tt4_9#h(Tt4_9)
    
    #turbine nozzle conditions (assumed that p8=po)
    
    pt5=pt4_9*pitn
    Tt5=Tt4_9*pitn**((gamma-1)/(gamma)*etapoltn)
    ht5=Cpp*Tt5#h(Tt5)
    
    
    #Fan exhaust quantities
    
    pt8=pt7
    Tt8=Tt7
    ht8=Cpp*Tt8#h(Tt8)
    
    M8=numpy.sqrt((((pt8/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
    T8=Tt8/(1+(gamma-1)/2*M8**2)
    h8=Cpp*T8#h(T8)
    u8=numpy.sqrt(2*(ht8-h8))

    #Core exhaust quantities (assumed that p6=po)
    
    pt6=pt5
    Tt6=Tt5
    ht6=Cpp*Tt6#h(Tt6)
    
    M6=numpy.sqrt((((pt6/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
    T6=Tt6/(1+(gamma-1)/2*M6**2)
    h6=Cpp*T6#h(T6)
    u6=numpy.sqrt(2*(ht6-h6))

    
    if M8 < 1.0:
    # nozzle unchoked
  
        p7=po
        
        M7=numpy.numpy.sqrt((((pt7/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
        T7=Tt7/(1+(gamma-1)/2*M7**2)
        h7=Cpp*T7
  
    
    else:
        M7=1
        T7=Tt7/(1+(gamma-1)/2*M7**2)
        p7=pt7/(1+(gamma-1)/2*M7**2)**(gamma/(gamma-1))
        h7=Cpp*T7
    

    u7=numpy.sqrt(2*(ht7-h7))
    rho7=p7/(R*T7)
  # 
  # 
  # #core nozzle ppties
  # 


    if M6 < 1.0:  #nozzle unchoked
      
        p5=po
        M5=numpy.numpy.sqrt((((pt5/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
        T5=Tt5/(1+(gamma-1)/2*M5**2)
        h5=Cpp*T5
    
   
    else:
        M5=1
        T5=Tt5/(1+(gamma-1)/2*M5**2)
        p5=pt5/(1+(gamma-1)/2*M5**2)**(gamma/(gamma-1))
        h5=Cpp*T5
    

  # #core nozzle area
  # 
    u5=numpy.sqrt(2*(ht5-h5))
    rho5=p5/(R*T5)
  # 
  # #-------------------------
  # #Thrust calculation based on that
  # 
  # 
   #Fsp=1/(ao*(1+aalpha))*((u5-uo)+aalpha*(u7-uo)+(p5-po)/(rho5*u5)+aalpha*(p7-po)/(rho7*u7)+f)
  
    Ae_b_Ao=1/(1+aalpha)*(fm_id(Mo)/fm_id(M5)*(1/(pt5/pto))*(numpy.sqrt(Tt5/Tto)))
    
    A1e_b_A1o=aalpha/(1+aalpha)*(fm_id(Mo)/fm_id(M7))*(1/(pt7/pto))*numpy.sqrt(Tt7/Tto)
     
    Thrust_nd=gamma*Mo**2*(1/(1+aalpha)*(u5/uo-1)+(aalpha/(1+aalpha))*(u7/uo-1))+Ae_b_Ao*(p5/po-1)+A1e_b_A1o*(p7/po-1)
    
    #calculate actual value of thrust 
    
    Fsp=1/(gamma*Mo)*Thrust_nd

  
  #overall engine quantities
    
    Isp=Fsp*ao*(1+aalpha)/(f*g)
    TSFC=3600/Isp  # for the test case 
    #print TSFC
  #-----Design sizing-------------------------------------------------------
  #--------------calculation pass-------------------------------------------
  
  
#---------sizing if thrust is provided-----------------------------------  
    
    #print TSFC
  
  #mass flow sizing
  
    mdot_core=FD/(Fsp*ao*(1+aalpha))
    


    #Component area sizing---------------------------------------------------
    
    #Component area sizing----------------------------------------------------
    
    #fan area
    
    T2=Tt2/(1+(gamma-1)/2*M2**2)
    h2=Cpp*T2
    p2=pt2/(1+(gamma-1)/2*M2**2)**(gamma/(gamma-1))
    
    
    #[p2,T2,h2]=FM(alpha,po,To,Mo,M2,etapold)  #non ideal case
    rho2=p2/(R*T2)
    u2=M2*numpy.sqrt(gamma*R*T2)
    A2=(1+aalpha)*mdot_core/(rho2*u2)
    df=numpy.sqrt(4*A2/(numpy.pi*(1-HTRf**2))) #if HTRf- hub to tip ratio is specified
    
    #hp compressor fan area
     
    T2_5=Tt2_5/(1+(gamma-1)/2*M2_5**2)
    h2_5=Cpp*T2_5
    p2_5=pt2_5/(1+(gamma-1)/2*M2_5**2)**(gamma/(gamma-1))
    
    #[p2_5,T2_5,h2_5]=FM(alpha,pt2_5,Tt2_5,0,M2_5,etapold)
    rho2_5=p2_5/(R*T2_5)
    u2_5=M2_5*numpy.sqrt(gamma*R*T2_5)
    A2_5=(1+alpha)*mdot_core/(rho2_5*u2_5)
    dhc=numpy.sqrt(4*A2_5/(numpy.pi*(1-HTRhc**2))) #if HTRf- hub to tip ratio is specified
     
    
    
    #fan nozzle area
    
    M8=u8/numpy.sqrt(Cp(T8)*R/(Cp(8)-R)*T8)
  
    if M8<1: # nozzle unchoked
    
        p7=po
        M7=numpy.sqrt((((pt7/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
        T7=Tt7/(1+(gamma-1)/2*M7**2)
        h7=Cpp*T7
  
    else:
      
        M7=1
        T7=Tt7/(1+(gamma-1)/2*M7**2)
        p7=pt7/(1+(gamma-1)/2*M7**2)**(gamma/(gamma-1))
        h7=Cpp*T7
      
  #end
  
    u7=numpy.sqrt(2*(ht7-h7))
    rho7=p7/(R*T7)
    A7=aalpha*mdot_core/(rho7*u7)
  
  #core nozzle area
    
    M6=u6/numpy.sqrt(Cp(T6)*R*T6/(Cp(T6)-R))
    
    if M6<1:  #nozzle unchoked
      
        p5=po
        M5=numpy.sqrt((((pt5/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
        T5=Tt5/(1+(gamma-1)/2*M5**2)
        h5=Cpp*T5
      
    else:
        M5=1
        T5=Tt5/(1+(gamma-1)/2*M5**2)
        p5=pt5/(1+(gamma-1)/2*M5**2)**(gamma/(gamma-1))
        h5=Cpp*T5
          
  #end
  
  #core nozzle area
  
    u5=numpy.sqrt(2*(ht5-h5))
    rho5=p5/(R*T5)
    A5=mdot_core/(rho5*u5)
  
    Ao=mdot_core*(1+aalpha)/(rhoo*uo)  
    
   
    mhtD=(1+f)*mdot_core*numpy.sqrt(Tt4_1/Tref)/(pt4_1/Pref)
    mltD=(1+f)*mdot_core*numpy.sqrt(Tt4_5/Tref)/(pt4_5/Pref)
    
    mdfD=(1+aalpha)*mdot_core*numpy.sqrt(Tt2/Tref)/(pt2/Pref)
    mdlcD=mdot_core*numpy.sqrt(Tt1_9/Tref)/(pt1_9/Pref)
    mdhcD=mdot_core*numpy.sqrt(Tt2_5/Tref)/(pt2_5/Pref) 
  
  
  #-------if areas specified-----------------------------
    #fuel_rate=mdot_core*f
    
    #FD2=Fsp*ao*(1+aalpha)*mdot_core  
  #-------------------------------------------------------
  
    
  
  #--------------geometry functions-----------------------------------------------
  #-------------------------------------------------------------------------------
  
  #CD_nacelle=nacelle_drag(df,uo,u2,rhoo,mu_o)
  
  
  ##-----engine weight normal engine----------------------
  
  ##-----direct way------------------------------
  
  #a=1.809*10*aalpha**2+4.769*10**2*aalpha+701.3
  #b=1.077*10**-3*aalpha**2-3.716*10**-2*aalpha+1.19
  #c=-1.058*10**-2*aalpha+0.326
  
  #mdot_lb=mdot_core*2.20462
  #OPR=pif*pilc*pihc
  
  #W_direct=a*(mdot_lb/100)**b*(OPR/40)**c
  
  ##--------nacelle weight-----------------------------
  
  #mdot_t=(1+aalpha)*mdot_core   #mass flow
  
  ##-------- inlet--------------------
  
  #f_M=(((gamma+1)/2)**((gamma+1)/(2*(gamma-1))))*Mo/((1+(gamma-1)/2*Mo**2)**((gamma+1)/(2*(gamma-1))))
  
  #temp=gamma/(((gamma+1)/2)**((gamma+1)/(2*(gamma-1))))*pto/numpy.sqrt(gamma*Ro*Tto)
  
  #Ao=mdot_t/(temp*f_M)
  
  #do_o=numpy.sqrt(Ao/(pi/4))
  ##----------exit--------------------------
  
  
  #aa=6.156*10**2*aalpha**2+1.357*10*aalpha+27.51
  #bb=6.8592*10**-4*aalpha**2-2.714*10**-2*aalpha+0.505
  #cc=0.129
  
  #L_eng_in=aa*(mdot_lb/100)**bb*(OPR/40)**cc   #engine length in inches
  #L_eng_m=0.0254*L_eng_in          #engine length in metres
  
  ##n_od=1.1*df
  
  ##-------------------nacelle volume----------------------------------
  
  ##---inlet---------
  
  #inlet_outV=pi*0.6*df/12*1.1**2*(df**2+df*do_o+do_o**2)
  #inlet_inV=pi*0.6*df/12*(df**2+df*do_o+do_o**2)
  #inlet_V=inlet_outV-inlet_inV
  
  ##---eng--------
  
  #eng_outV=pi/4*(1.1*df)**2*L_eng_m
  #eng_inV=pi/4*(df)**2*L_eng_m
  #eng_V=eng_outV-eng_inV
  
  #nacelle_vol=inlet_V+eng_V
  
  #nacelle_wt=nc_den*nacelle_vol*2.20462
  
  #Wtot=W_direct+nacelle_wt
  
  #end
    Turbofan.A2= A2
    Turbofan.df= df
    Turbofan.A2_5= A2_5
    Turbofan.dhc= dhc
    Turbofan.A7= A7
    Turbofan.A5= A5
    Turbofan.Ao=Ao
    Turbofan.mdt= mhtD
    Turbofan.mlt= mltD
    Turbofan.mdf=mdfD
    Turbofan.mdlc=mdlcD

  
  
    #Turbofan.sfc=sfc
    #Turbofan.thrust=th  
    Turbofan.mdhc=mdhcD
  
    #return Fsp,TSFC,mdhcD
      
