""" Propulsion.py: Methods for Propulsion Analysis """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data
from SUAVE.Attributes import Constants
# import SUAVE

# ----------------------------------------------------------------------
#  Mission Methods
# ----------------------------------------------------------------------

def evaluate(eta,segment):

    """  CF, Isp, Psp = evaluate(eta,segment): sum the performance of all the propulsion systems
    
         Inputs:    eta = array of throttle values                          (required)  (floats)    
                    segment                                                 (required)  (Mission Segment instance)

         Outputs:   CF = array of thurst coefficient values                 (floats)
                    Isp = array of specific impulse values (F/(dm/dt))      (floats)
                    Psp = array of specific power values (P/F)              (floats)                                                                                            

        """

         

    return

# ----------------------------------------------------------------------
#  Sizing Methods
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
#  Utility Methods
# ----------------------------------------------------------------------


#--necessary functions---------------------------------

#have a base engine enalysis method which reads in the type of fidelity you want along with the inputs

def initialize(propulsor,F_max,alt_design=0.0,M_design=0.0):

    """  CF, Isp, Psp = evaluate(eta,segment): sum the performance of all the propulsion systems
    
         Inputs:    eta = array of throttle values                          (required)  (floats)    
                    segment                                                 (required)  (Mission Segment instance)

         Outputs:   CF = array of thurst coefficient values                 (floats)
                    Isp = array of specific impulse values (F/(dm/dt))      (floats)
                    Psp = array of specific power values (P/F)              (floats)                                                                                            

    """

    # determine pt and Tt throughout


    # shaft work balance 


    # find 


    return

 
def engine_analysis_pass(Turbofan,State):    
    
    
    mach=State.M
    a=State.alt    
    
    #mach=Segment.mach
    #a=Segment.alt
    sfc_sfcref=Turbofan.sfc_sfcref
    sls_thrust=Turbofan.thrust_sls
    eng_type=Turbofan.type


    if eng_type==1:
        sfc= sfc_sfcref*(0.335780271274146 +  0.029873325206606*mach +  0.081209003474378*a- 0.821844062318631*mach**2 - 0.012313371529160*a*mach- 0.013980789615436*a**2 + 1.941911084998007*mach**3 - 0.058052828837187*mach**2*a +  0.004256576580281*mach*a**2 +  0.001331152570038*a**3 - 1.201666666666645*mach**4 + 0.044941876252323*mach**3*a + 0.000021022540684*mach**2*a**2 - 0.000294083634720*mach*a**3 - 0.000032451442616*a**4)
        th=sls_thrust*(0.868009508296048  -0.166557104971858*mach  -0.144651575623525*a  -1.407030667976488*mach**2  + 0.093905630521542*mach*a +  0.027666192971266*a**2 +  0.853125587544063*mach**3 +  0.118638337242548*mach**2*a  -0.019855919345256*mach*a**2  -0.002651612710499*a**3  -0.112812499999986*mach**4  -0.052144616158441*mach**3*a -0.000980638640291*mach**2*a**2   +0.001005413843178*mach*a**3 +  0.000094582673830*a**4)

    if eng_type==2:
        sfc= sfc_sfcref*(0.607700000000000  -0.000000000000003*mach+   0.054425447119591*a +  0.000000000000018*mach**2  -0.000000000000000*mach*a  -0.009605314839191*a**2  -0.000000000000037*mach**3  -0.000000000000000*mach**2*a +  0.000000000000000*mach*a**2 +  0.000949737253235*a**3+   0.000000000000024*mach**4  -0.000000000000000*mach**3*a  -0.000000000000000*mach**2*a**2  -0.000000000000000*mach*a**3  -0.000025397493590*a**4)
        th=sls_thrust*(0.804113755084826  -0.035627902201493*mach  -0.096184884986570*a  -2.779712131707290*mach**2  + 0.095737606320651*mach*a +  0.016854969440649*a**2 +  3.691579501909484*mach**3+   0.153160834493603*mach**2*a -0.018080377409672*mach*a**2  -0.001507189240658*a**3  -1.577283906249969*mach**4  -0.153161698513016*mach**3*a+   0.002018571573980*mach**2*a**2+   0.000869874382566*mach*a**3+   0.000048066992847*a**4)

    if eng_type==3:
        sfc= sfc_sfcref*(0.285436237761663 +  0.024502655477171*mach +  0.068999277904472*a  -0.697385662078613*mach**2 -0.010544773458370*mach*a  -0.011884083398418*a**2 +  1.653048305914588*mach**3  -0.049001864268300*mach**2*a +   0.003616272596893*mach*a**2 +  0.001131761236922*a**3 -1.024651041666653*mach**4 +  0.037773643074563*mach**3*a +   0.000062961864908*mach**2*a**2  -0.000252876560383*mach*a**3  -0.000027565941442*a**4)
        th=sls_thrust*(0.860753545648466  -0.254090878749431*mach  -0.108037058286478*a  -1.010034245425557*mach**2+  0.069717277528921*mach*a +0.011645535013526*a**2+  0.423541666666671*mach**3 + 0.047130921483669*mach**2*a +  -0.006331867045162*mach*a**2  -0.000460002484144*a**3)

    if eng_type==5:
        sfc= sfc_sfcref*(0)
        th=sls_thrust*(0.860707560213307  -0.251373072375285*mach  -0.107900030324592*a  -1.019215393654531*mach**2+  0.069827313560592*mach*a +  0.011615366286529*a**2+  0.430875000000004*mach**3 +  0.046861671760037*mach**2*a -0.006319498957339*mach*a**2  -0.000458427089846*a**3)
 

    if eng_type==6:
        sfc= sfc_sfcref*(1.039279928346275+ 0.154354343003498*mach+ 0.117486771130496*a -0.265820962970648*mach**2 + 0.011679904232747*mach*a  -0.065853428512314*a**2 +  0.112879463923913*mach**3  -0.005232011798524*mach**2*a  -0.000864238935027*mach*a**2 +  0.009811438002540*a**3  -0.015448922821971*mach**4  + 0.000896085056928*mach**3*a +  0.000148921068721*mach**2*a**2  -0.000014581672329*a**3*mach  -0.000387827542944*mach**4)
        th=sls_thrust*(0.871003609000932  -0.503449519567159*mach + 0.245774798277167*a + 0.038477468960137*mach**2  -0.139342839995028*mach*a +  0.000585397434489*a**2 +  0.023113033234127*mach**3  -0.000098763071611*mach**2*a+  0.011975525050602*mach*a**2  -0.001938941125545*a**3)


#-------------put in engine geometry specifications----

    Turbofan.sfc=sfc
    Turbofan.thrust=th


    return Turbofan

#--------------engine sizing -----------------------------------------------------------

def engine_sizing_pass(Turbofan,State):

    #sls_thrust=Turbofan.sls_thrust
    
    ''' outputs = engine_dia,engine_length,nacelle_dia,inlet_length,eng_area,inlet_area
        inputs  engine related : sls_thrust - basic pass input

  '''    
    #unpack inputs
    
    sls_thrust=Turbofan.thrust_sls
    
    
    #calculate

    engine_dia=1.0827*sls_thrust**0.4134
    engine_length=2.4077*sls_thrust**0.3876
    nacelle_dia=1.1*engine_dia
    inlet_length=0.6*engine_dia
    eng_maxarea=3.14*0.25*engine_dia**2
    inlet_area=0.7*eng_maxarea

    
    
    
    #Pack results
    
    Turbofan.bare_engine_dia=engine_dia
    Turbofan.bare_engine_length=engine_length
    Turbofan.nacelle_dia=nacelle_dia
    Turbofan.inlet_length=inlet_length
    Turbofan.eng_maxarea=eng_maxarea
    Turbofan.inlet_area= inlet_area    
    
    #Vehicle.Turbofan.bare_engine_dia=engine_dia
    #Vehicle.Turbofan.bare_engine_length=engine_length
    #Vehicle.Turbofan.nacelle_dia=nacelle_dia
    #Vehicle.Turbofan.inlet_length=inlet_length
    #Vehicle.Turbofan.eng_maxarea=eng_maxarea
    #Vehicle.Turbofan.inlet_area= inlet.area



    #return (engine_dia,engine_length,nacelle_dia,inlet_length,eng_maxarea,inlet_area)



    #engine analysis based on TASOPT
    #constant Cp is not assumed 
    #pressure ratio prescribed
    #MAIN CODE
     
def fm_id(M):

    R=287.87
    g=1.4
    m0=(g+1)/(2*(g-1))
    m1=((g+1)/2)**m0
    m2=(1+(g-1)/2*M**2)**m0
    fm=m1*M/m2
    return fm
      
def Cp(T):

    #gamma=1.4
    Thetav=3354
    R=287
    Cpt=R*(7/2+((Thetav/(2*T))/(numpy.sinh(Thetav/(2*T))))**2)
    
    #Cpt = 1.9327e-10*T**4 - 7.9999e-7*T**3 + 1.1407e-3*T**2 - 4.4890e-1*T + 1.0575e+3
    return Cpt

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
      



  #engine analysis based on TASOPT
  #constant Cp is not assumed 
  #pressure ratio prescribed
  #MAIN CODE
  

def engine_analysis_1d(vehicle,throttles,State):  
#def engine_analysis_1d(Turbofan,State):
#def engine_analysis_1d(Minf,Tinf,pinf,pid,pif,pifn,pilc,pihc,pib,pitn,Tt4,aalpha,mdhc): 
   
   #(self,eta,segment)
    
    Minf_a=State.M
    Tinf_a=State.T
    pinf_a=State.p
    
    
    
    arr_len=len(Minf_a)
    
    TSFC_a=numpy.empty(arr_len)
    FD_a=numpy.empty(arr_len)
    mdot_core_a=numpy.empty(arr_len)   
    mfuel_a=numpy.empty(arr_len)
    CF=numpy.empty(arr_len)
    Isp=numpy.empty(arr_len)
    
    
    
    
    pid=State.config.Propulsors[0].diffuser_pressure_ratio
    pif=State.config.Propulsors[0].fan_pressure_ratio
    pifn=State.config.Propulsors[0].fan_nozzle_pressure_ratio
    pilc=State.config.Propulsors[0].lpc_pressure_ratio
    pihc=State.config.Propulsors[0].hpc_pressure_ratio
    pib=State.config.Propulsors[0].burner_pressure_ratio
    pitn=State.config.Propulsors[0].turbine_nozzle_pressure_ratio
    Tt4=State.config.Propulsors[0].Tt4
    aalpha=State.config.Propulsors[0].bypass_ratio
    mdhc=State.config.Propulsors[0].mdhc  
    A22=State.config.Propulsors[0].A2 
    no_eng=State.config.Propulsors[0].no_of_engines 
   
   
   
    #print mdhc
    for ln in range(0,arr_len):
        
        Minf=Minf_a[ln]
        Tinf=Tinf_a[ln]
        pinf=pinf_a[ln]
        throttle=throttles[ln]
        #print throttle
        #throttle=1.0
    
   
   
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
           
        alpha=1
        beta=1
        gammaa=1
        alphac=0
     
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
        #Tt4=1380#K
        #Cppp=1005
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
        
        #delh=0.5*uo**2
       
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
            
            M7=numpy.sqrt((((pt7/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
            T7=Tt7/(1+(gamma-1)/2*M7**2)
            h7=Cpp*T7
        
        else:
            M7=1
            T7=Tt7/(1+(gamma-1)/2*M7**2)
            p7=pt7/(1+(gamma-1)/2*M7**2)**(gamma/(gamma-1))
            h7=Cpp*T7
          
        # 
        u7=numpy.sqrt(2*(ht7-h7))
        rho7=p7/(R*T7)
    
        # #core nozzle ppties
    
        # 
        if M6 < 1.0:  #nozzle unchoked
            
            p5=po
            M5=numpy.sqrt((((pt5/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
            T5=Tt5/(1+(gamma-1)/2*M5**2)
            h5=Cpp*T5
          
      
        else:
            M5=1
            T5=Tt5/(1+(gamma-1)/2*M5**2)
            p5=pt5/(1+(gamma-1)/2*M5**2)**(gamma/(gamma-1))
            h5=Cpp*T5
          
    
        # 
        # #core nozzle area
        # 
        u5=numpy.sqrt(2*(ht5-h5))
        rho5=p5/(R*T5)
        # 
        # #-------------------------
        # #Thrust calculation based on that
        # 
       
        Ae_b_Ao=1/(1+aalpha)*(fm_id(Mo)/fm_id(M5)*(1/(pt5/pto))*(numpy.sqrt(Tt5/Tto)))
        
        A1e_b_A1o=aalpha/(1+aalpha)*(fm_id(Mo)/fm_id(M7))*(1/(pt7/pto))*numpy.sqrt(Tt7/Tto)
         
        Thrust_nd=gamma*Mo**2*(1/(1+aalpha)*(u5/uo-1)+(aalpha/(1+aalpha))*(u7/uo-1))+Ae_b_Ao*(p5/po-1)+A1e_b_A1o*(p7/po-1)
        
        
        #Fsp=((1+f)*u6-uo+aalpha*(u8-uo))/((1+aalpha)*ao);
        #calculate actual value of thrust 
        
        Fsp=1/(gamma*Mo)*Thrust_nd
    
        #overall engine quantities
        
        Ispp=Fsp*ao*(1+aalpha)/(f*g)
    
        TSFC=3600/Ispp  # for the test case 
        
        #mass flow sizing
        mdot_core=mdhc*numpy.sqrt(Tref/Tt2_5)*(pt2_5/Pref)
        
        #mdot_core=FD/(Fsp*ao*(1+aalpha))
        #print mdot_core
      
      
        #-------if areas specified-----------------------------
        fuel_rate=mdot_core*f*no_eng
        
        FD2=Fsp*ao*(1+aalpha)*mdot_core*no_eng*throttle  
        mfuel=0.1019715*FD2*TSFC/3600
        State.config.A_engine=A22
        
        CF[ln] = FD2/(State.q[ln]*State.config.A_engine)
        Isp[ln] = FD2/(mfuel*State.g0)        
        
        #-------------------------------------------------------
        #Turbofan.sfc = TSFC
        #Turbofan.thrust = FD2
        #Turbofan.mdot_core = mdot_core
        
        TSFC_a[ln]=TSFC
        FD_a[ln]=FD2
        mdot_core_a[ln]=mdot_core
        mfuel_a[ln]=mfuel
        
  
    #return FD_a,TSFC_a,mfuel_a
    return CF, Isp

#-------ducted fan-----------------------------------------------------------------------

def engine_sizing_ductedfan(Ductedfan,State):    
#def engine_sizing_1d(Minf,Tinf,pinf,pid,pif,pifn,pilc,pihc,pib,pitn,Tt4,aalpha,FD):

    Minf=State.M
    Tinf=State.T
    pinf=State.p
    pid=Ductedfan.diffuser_pressure_ratio
    pif=Ductedfan.fan_pressure_ratio
    pifn=Ductedfan.fan_nozzle_pressure_ratio
    FD=Ductedfan.design_thrust




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
    
    
    ##LPC exit conditions
    
    ##etapollc=Feta(pilc,pilc,1,1)
    ##etapollc=1
    
    #pt2_5=pt1_9*pilc
    #Tt2_5=Tt1_9*pilc**((gamma-1)/(gamma*etapollc))
    #ht2_5=Cpp*Tt2_5  #h(Tt2_5)
    
    
    ##HPC exit calculations
    
    ##etapolhc=Feta(pihc,pihc,1,1)
    ##etapolhc=1
    
    #pt3=pt2_5*pihc
    #Tt3=Tt2_5*pihc**((gamma-1)/(gamma*etapolhc))
    #ht3=Cpp*Tt3#h(Tt3)
    
   
    ##cooling mass flow
    
    ##to be included in the future
    
    ##combustor quantities
    
    ##[fb,lambda]=Fb(alpha,beta,gammaa,Tt3,htf,Tt4)
    
    #fb=(((Tt4/To)-tau_r*(Tt3/Tt1_9))/(eta_b*tau_f-(Tt4/To)))
    
    #f=fb*(1-alphac)  #alphac=0 if cooling mass flow not included
    #lmbda=1
    
    ##combustor exit conditions
    
    #ht4=lmbda*Cpp*Tt4#h(Tt4)
    ##sigmat4=lambda*sigma(Tt4)
    #pt4=pt3*pib
    
    ##Station 4.1 without IGV cooling
    
    #Tt4_1=Tt4
    #pt4_1=pt4
    #lambdad=lmbda
    
    ##Station 4.1 with IGV cooling
    
    ##complete later
    
    
    ##Turbine quantities
    
    ##high pressure turbine
    
    #deltah_ht=-1/(1+f)*1/0.99*(ht3-ht2_5)
    
    #Tt4_5=Tt4_1+deltah_ht/Cpp
    
    #pt4_5=pt4_1*(Tt4_5/Tt4_1)**(gamma/((gamma-1)*etapolt))
    #ht4_5=Cpp*Tt4_5#h(Tt4_5)
    
    ##low pressure turbine
    
    #deltah_lt=-1/(1+f)*md_lc_by_md_hc*(1/0.99*(ht2_5-ht1_9)+aalpha*1/0.99*(ht2_1-ht2))
    
    #Tt4_9=Tt4_5+deltah_lt/Cpp
    
    #pt4_9=pt4_5*(Tt4_9/Tt4_5)**(gamma/((gamma-1)*etapolt))
    #ht4_9=Cpp*Tt4_9#h(Tt4_9)
    
    ##turbine nozzle conditions (assumed that p8=po)
    
    #pt5=pt4_9*pitn
    #Tt5=Tt4_9*pitn**((gamma-1)/(gamma)*etapoltn)
    #ht5=Cpp*Tt5#h(Tt5)
    
    
    #Fan exhaust quantities
    
    pt8=pt7
    Tt8=Tt7
    ht8=Cpp*Tt8#h(Tt8)
    
    M8=numpy.sqrt((((pt8/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
    T8=Tt8/(1+(gamma-1)/2*M8**2)
    h8=Cpp*T8#h(T8)
    u8=numpy.sqrt(2*(ht8-h8))

    ##Core exhaust quantities (assumed that p6=po)
    
    #pt6=pt5
    #Tt6=Tt5
    #ht6=Cpp*Tt6#h(Tt6)
    
    #M6=numpy.sqrt((((pt6/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
    #T6=Tt6/(1+(gamma-1)/2*M6**2)
    #h6=Cpp*T6#h(T6)
    #u6=numpy.sqrt(2*(ht6-h6))

    
    if M8 < 1.0:
    # nozzle unchoked
  
        p7=po
        
        M7=numpy.sqrt((((pt7/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
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


    #if M6 < 1.0:  #nozzle unchoked
      
        #p5=po
        #M5=numpy.numpy.sqrt((((pt5/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
        #T5=Tt5/(1+(gamma-1)/2*M5**2)
        #h5=Cpp*T5
    
   
    #else:
        #M5=1
        #T5=Tt5/(1+(gamma-1)/2*M5**2)
        #p5=pt5/(1+(gamma-1)/2*M5**2)**(gamma/(gamma-1))
        #h5=Cpp*T5
    

  ## #core nozzle area
  ## 
    #u5=numpy.sqrt(2*(ht5-h5))
    #rho5=p5/(R*T5)
  # 
  # #-------------------------
  # #Thrust calculation based on that
  # 
  # 
   #Fsp=1/(ao*(1+aalpha))*((u5-uo)+aalpha*(u7-uo)+(p5-po)/(rho5*u5)+aalpha*(p7-po)/(rho7*u7)+f)
  
   # Ae_b_Ao=1/(1+aalpha)*(fm_id(Mo)/fm_id(M5)*(1/(pt5/pto))*(numpy.sqrt(Tt5/Tto)))
    
    A1e_b_A1o=(fm_id(Mo)/fm_id(M7))*(1/(pt7/pto))*numpy.sqrt(Tt7/Tto)
     
    Thrust_nd=gamma*Mo**2*((u7/uo-1))+A1e_b_A1o*(p7/po-1)
    
    #calculate actual value of thrust 
    
    Fsp=1/(gamma*Mo)*Thrust_nd

  
  #overall engine quantities
    
    #Isp=Fsp*ao*(1+aalpha)/(f*g)
    #TSFC=3600/Isp  # for the test case 
  
  #-----Design sizing-------------------------------------------------------
  #--------------calculation pass-------------------------------------------
  
  
#---------sizing if thrust is provided-----------------------------------  
    #print Fsp
    #print TSFC
  
  #mass flow sizing
  
    mdot_df=FD/(Fsp*ao)
    #print mdot_core


    #Component area sizing---------------------------------------------------
    
    #Component area sizing----------------------------------------------------
    
    #fan area
    
    T2=Tt2/(1+(gamma-1)/2*M2**2)
    h2=Cpp*T2
    p2=pt2/(1+(gamma-1)/2*M2**2)**(gamma/(gamma-1))
    
    
    #[p2,T2,h2]=FM(alpha,po,To,Mo,M2,etapold)  #non ideal case
    rho2=p2/(R*T2)
    u2=M2*numpy.sqrt(gamma*R*T2)
    A2=mdot_df/(rho2*u2)
    df=numpy.sqrt(4*A2/(numpy.pi*(1-HTRf**2))) #if HTRf- hub to tip ratio is specified
    
    ##hp compressor fan area
     
    #T2_5=Tt2_5/(1+(gamma-1)/2*M2_5**2)
    #h2_5=Cpp*T2_5
    #p2_5=pt2_5/(1+(gamma-1)/2*M2_5**2)**(gamma/(gamma-1))
    
    ##[p2_5,T2_5,h2_5]=FM(alpha,pt2_5,Tt2_5,0,M2_5,etapold)
    #rho2_5=p2_5/(R*T2_5)
    #u2_5=M2_5*numpy.sqrt(gamma*R*T2_5)
    #A2_5=(1+alpha)*mdot_core/(rho2_5*u2_5)
    #dhc=numpy.sqrt(4*A2_5/(numpy.pi*(1-HTRhc**2))) #if HTRf- hub to tip ratio is specified
     
    
    
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
    A7=mdot_df/(rho7*u7)
  
  ##core nozzle area
    
    #M6=u6/numpy.sqrt(Cp(T6)*R*T6/(Cp(T6)-R))
    
    #if M6<1:  #nozzle unchoked
      
        #p5=po
        #M5=numpy.sqrt((((pt5/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
        #T5=Tt5/(1+(gamma-1)/2*M5**2)
        #h5=Cpp*T5
      
    #else:
        #M5=1
        #T5=Tt5/(1+(gamma-1)/2*M5**2)
        #p5=pt5/(1+(gamma-1)/2*M5**2)**(gamma/(gamma-1))
        #h5=Cpp*T5
          
  ##end
  
  #core nozzle area
  
    #u5=numpy.sqrt(2*(ht5-h5))
    #rho5=p5/(R*T5)
    #A5=mdot_core/(rho5*u5)
  
    Ao=mdot_df/(rhoo*uo)  
    
   
    #mhtD=(1+f)*mdot_core*numpy.sqrt(Tt4_1/Tref)/(pt4_1/Pref)
    #mltD=(1+f)*mdot_core*numpy.sqrt(Tt4_5/Tref)/(pt4_5/Pref)
    
    mdfD=mdot_df*numpy.sqrt(Tt2/Tref)/(pt2/Pref)
    #mdlcD=mdot_core*numpy.sqrt(Tt1_9/Tref)/(pt1_9/Pref)
    #mdhcD=mdot_core*numpy.sqrt(Tt2_5/Tref)/(pt2_5/Pref) 
  
  
  
  ##end
    Ductedfan.A2= A2
    #Turbofan.df= df
    #Turbofan.A2_5= A2_5
    #Turbofan.dhc= dhc
    Ductedfan.A7= A7
    #Turbofan.A5= A5
    #Turbofan.Ao=Ao
    #Turbofan.mdt= mhtD
    #Turbofan.mlt= mltD
    #Turbofan.mdf=mdfD
    #Turbofan.mdlc=mdlcD

  
  
    #Turbofan.sfc=sfc
    #Turbofan.thrust=th  
    #Turbofan.mdhc=mdhcD
  
    #return Fsp,TSFC,mdhcD
      



  #engine analysis based on TASOPT
  #constant Cp is not assumed 
  #pressure ratio prescribed
  #MAIN CODE

def engine_analysis_ductedfan(Ductedfan,State):
#def engine_analysis_1d(Minf,Tinf,pinf,pid,pif,pifn,pilc,pihc,pib,pitn,Tt4,aalpha,mdhc): 
   
   
    Minf=State.M
    Tinf=State.T
    pinf=State.p
    pid=Ductedfan.diffuser_pressure_ratio
    pif=Ductedfan.fan_pressure_ratio
    pifn=Ductedfan.fan_nozzle_pressure_ratio
    A7=Ductedfan.A7  
   
   
   
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
       
    alpha=1
    beta=1
    gammaa=1
    alphac=0
 
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
    #Tt4=1380#K
    #Cppp=1005
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
    
    #delh=0.5*uo**2
   
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
    
    ##etapollc=Feta(pilc,pilc,1,1)
    ##etapollc=1
    
    #pt2_5=pt1_9*pilc
    #Tt2_5=Tt1_9*pilc**((gamma-1)/(gamma*etapollc))
    #ht2_5=Cpp*Tt2_5  #h(Tt2_5)
    
    
    
    ##HPC exit calculations
    
    ##etapolhc=Feta(pihc,pihc,1,1)
    ##etapolhc=1
    
    #pt3=pt2_5*pihc
    #Tt3=Tt2_5*pihc**((gamma-1)/(gamma*etapolhc))
    #ht3=Cpp*Tt3#h(Tt3)
    
    
    
    ##cooling mass flow
    
    ##to be included in the future
    
    ##combustor quantities
    
    #fb=(((Tt4/To)-tau_r*(Tt3/Tt1_9))/(eta_b*tau_f-(Tt4/To)))
    
    #f=fb*(1-alphac)  #alphac=0 if cooling mass flow not included
    #lmbda=1
    
    ##combustor exit conditions
    
    #ht4=lmbda*Cpp*Tt4#h(Tt4)
    ##sigmat4=lambda*sigma(Tt4)
    #pt4=pt3*pib
    
    ##Station 4.1 without IGV cooling
    
    #Tt4_1=Tt4
    #pt4_1=pt4
    #lambdad=lmbda
    
    ##Station 4.1 with IGV cooling
    
    ##complete later
    
    
    ##Turbine quantities
    
    ##high pressure turbine
    
    #deltah_ht=-1/(1+f)*1/0.99*(ht3-ht2_5)
    
    #Tt4_5=Tt4_1+deltah_ht/Cpp
    
    #pt4_5=pt4_1*(Tt4_5/Tt4_1)**(gamma/((gamma-1)*etapolt))
    #ht4_5=Cpp*Tt4_5#h(Tt4_5)
    
    ##low pressure turbine
    
    #deltah_lt=-1/(1+f)*md_lc_by_md_hc*(1/0.99*(ht2_5-ht1_9)+aalpha*1/0.99*(ht2_1-ht2))
    
    #Tt4_9=Tt4_5+deltah_lt/Cpp
    
    #pt4_9=pt4_5*(Tt4_9/Tt4_5)**(gamma/((gamma-1)*etapolt))
    #ht4_9=Cpp*Tt4_9#h(Tt4_9)
    
    ##turbine nozzle conditions (assumed that p8=po)
    
    #pt5=pt4_9*pitn
    #Tt5=Tt4_9*pitn**((gamma-1)/(gamma)*etapoltn)
    #ht5=Cpp*Tt5#h(Tt5)
    
    
    #Fan exhaust quantities
    
    pt8=pt7
    Tt8=Tt7
    ht8=Cpp*Tt8#h(Tt8)
    
    M8=numpy.sqrt((((pt8/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
    T8=Tt8/(1+(gamma-1)/2*M8**2)
    h8=Cpp*T8#h(T8)
    u8=numpy.sqrt(2*(ht8-h8))

    
    #Core exhaust quantities (assumed that p6=po)
    
    #pt6=pt5
    #Tt6=Tt5
    #ht6=Cpp*Tt6#h(Tt6)
    
    #M6=numpy.sqrt((((pt6/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
    #T6=Tt6/(1+(gamma-1)/2*M6**2)
    #h6=Cpp*T6#h(T6)
    #u6=numpy.sqrt(2*(ht6-h6))

    if M8 < 1.0:
    # nozzle unchoked
    
        p7=po
        
        M7=numpy.sqrt((((pt7/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
        T7=Tt7/(1+(gamma-1)/2*M7**2)
        h7=Cpp*T7
    
    else:
        M7=1
        T7=Tt7/(1+(gamma-1)/2*M7**2)
        p7=pt7/(1+(gamma-1)/2*M7**2)**(gamma/(gamma-1))
        h7=Cpp*T7
      
    # 
    u7=numpy.sqrt(2*(ht7-h7))
    rho7=p7/(R*T7)

    # #core nozzle ppties

    # 
    #if M6 < 1.0:  #nozzle unchoked
        
        #p5=po
        #M5=numpy.numpy.sqrt((((pt5/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
        #T5=Tt5/(1+(gamma-1)/2*M5**2)
        #h5=Cpp*T5
      
  
    #else:
        #M5=1
        #T5=Tt5/(1+(gamma-1)/2*M5**2)
        #p5=pt5/(1+(gamma-1)/2*M5**2)**(gamma/(gamma-1))
        #h5=Cpp*T5
      

    # 
    # #core nozzle area
    # 
    #u5=numpy.sqrt(2*(ht5-h5))
    #rho5=p5/(R*T5)
    # 
    # #-------------------------
    # #Thrust calculation based on that
    # 
   
    #Ae_b_Ao=1/(1+aalpha)*(fm_id(Mo)/fm_id(M5)*(1/(pt5/pto))*(numpy.sqrt(Tt5/Tto)))
    
    A1e_b_A1o=(fm_id(Mo)/fm_id(M7))*(1/(pt7/pto))*numpy.sqrt(Tt7/Tto)
     
    Thrust_nd=gamma*Mo**2*((u7/uo-1))+A1e_b_A1o*(p7/po-1)
    
    #calculate actual value of thrust 
    
    Fsp=1/(gamma*Mo)*Thrust_nd

    #overall engine quantities
    
    #Isp=Fsp*ao*(1+aalpha)/(f*g)

    #TSFC=3600/Isp  # for the test case 
    
    #mass flow sizing
    #mdot_core=mdhc*numpy.sqrt(Tref/Tt2_5)*(pt2_5/Pref)
    
    #mdot_core=FD/(Fsp*ao*(1+aalpha))
    #print mdot_core
  
  
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
    #A7=mdot_df/(rho7*u7)    
    mdot_df=A7*rho7*u7
    
    FD=Fsp*ao*mdot_df
    
  
  
  
  
  
  
  
    #-------if areas specified-----------------------------
    #fuel_rate=mdot_core*f
    
    #FD2=Fsp*ao*(1+aalpha)*mdot_core  
    #-------------------------------------------------------
    #Ductedfan.sfc = TSFC
    Ductedfan.thrust = FD
    #Turbofan.mdot_core = mdot_core
  
    #return FD2,TSFC,mdot_core

def engine_analysis(Turbofan,State):  

#def engine_analysis(mach,a,sfc_sfcref,sls_thrust,eng_type):
    ''' outputs = sfc, th (thrust)  - both cruise values 
        inputs  engine related : sfc_sfcref - basic pass input
                                 sls_thrust - basic pass input
                                   
                                   
                                   
                                   eng_type  - Engine Type Engine Type from the following list:
                                                1. High bypass turbofan (PW 2037 : sfc_sls = 0.33)  
                                                2. Low bypass turbofan (JT8-D  : sfc_sls = 0.53 )
                                                3. UDF (Propfan)
                                                4. Generic Turboprop
                                                5. Reserved
                                                6. SST Engine
                                                7. SST Engine with improved lapse rate 
                                  
              mission related  : mach (Mach number)
                                 a (altitude)
                
    '''
    
    #unpack
    
    
    if Turbofan.analysis_type == 'pass' :
        engine_analysis_pass(Turbofan,State)
    elif Turbofan.analysis_type == '1D' :
        engine_analysis_1d(Turbofan,State) 
        
def engine_sizing(Turbofan,State):

    #sls_thrust=Turbofan.sls_thrust
    
    ''' outputs = engine_dia,engine_length,nacelle_dia,inlet_length,eng_area,inlet_area
        inputs  engine related : sls_thrust - basic pass input

  '''
    
    if Turbofan.analysis_type == 'pass' :
        engine_sizing_pass(Turbofan,State)
    elif Turbofan.analysis_type == '1D' :
        engine_sizing_1d(Turbofan,State)          