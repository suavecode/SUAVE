""" Turbojet_Supersonic.py: Turbojet 1D gasdynamic engine model """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy #as np
from SUAVE.Core import Data, Container
from Propulsor import Propulsor
import Segments

# ----------------------------------------------------------------------
#  Turbojet
# ----------------------------------------------------------------------

class Turbojet_Supersonic(Propulsor):

    def __defaults__(self):
        
        self.tag = 'Turbojet Variable Nozzle'
        self.Compressors = Segments.Segment.Container()
        self.Compressors.HP = Segments.Compressor()
        self.Compressors.LP = Segments.Compressor()
        self.Burner = Segments.Burner()
        self.Turbines = Segments.Segment.Container()
        self.Turbines.HP = Segments.Compressor()
        self.Turbines.LP = Segments.Compressor()
        self.Afterburner = Segments.Burner()
        self.Fan = Segments.Fan()


class Turbojet_SupersonicPASS(Propulsor):
    def __defaults__(self):
        self.tag = 'Turbojet Variable Nozzle'
        self.thrust_sls = 0.0
        self.sfc_TF     = 0.0
        self.kwt0_eng   = 0.0
        self.type               = 0
        self.length             = 0.0
        self.cowl_length        = 0.0
        self.eng_thrt_ratio     = 0.0
        self.hilight_thrt_ratio = 0.0
        self.lip_finess_ratio   = 0.0
        self.height_width_ratio = 0.0
        self.upper_surf_shape_factor = 0.0
        self.lower_surf_shape_factor = 0.0
        self.dive_flap_ratio    = 0.0
        self.tip      = Data()
        self.inlet    = Data()
        self.diverter = Data()
        self.nozzle   = Data()
        self.analysis_type= 'pass'
        
        
        #--geometry pass like
        
        self.engine_dia= 0.0
        self.engine_length= 0.0
        self.nacelle_dia= 0.0
        self.inlet_length= 0.0
        self.eng_maxarea= 0.0
        self.inlet_area= 0.0   
        
        #newly added for 1d engine analysis
        self.diffuser_pressure_ratio = 1.0
        self.fan_pressure_ratio = 1.0
        self.fan_nozzle_pressure_ratio = 1.0
        self.lpc_pressure_ratio = 1.0
        self.hpc_pressure_ratio = 1.0
        self.burner_pressure_ratio = 1.0
        self.turbine_nozzle_pressure_ratio = 1.0
        self.Tt4 = 1400
        self.bypass_ratio = 0.0
        self.mdhc = 0.0        
        self.thrust = Data()
        self.thrust.design = 1.0
        self.no_of_engines=0.0
        
        #----geometry
        self.A2 = 0.0
        self.df = 0.0
        self.A2_5 = 0.0
        self.dhc = 0.0
        self.A7 = 0.0
        self.A5 = 0.0
        self.Ao = 0.0
        self.mdt = 0.0
        self.mlt = 0.0
        self.mdf = 0.0
        self.mdlc = 0.0     
        self.mdot_core = 0.0
        self.D=0.0
        
    def engine_sizing_1d(self,State):
        #------------------------------engine sizing method-----------------------------------------------     
        Minf=State.M
        Tinf=State.T
        pinf=State.p
        pid=self.diffuser_pressure_ratio
        pif=self.fan_pressure_ratio
        pifn=self.fan_nozzle_pressure_ratio
        pilc=self.lpc_pressure_ratio
        pihc=self.hpc_pressure_ratio
        pib=self.burner_pressure_ratio
        pitn=self.turbine_nozzle_pressure_ratio
        Tt4=self.Tt4
        aalpha=0 # Changes from a turbofan to turbojet
        FD=self.thrust.design
    
    
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
        
        
        pt6=pt5
        Tt6=Tt5
        ht6=Cpp*Tt6#h(Tt6)
        
        M6=numpy.sqrt((((pt6/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
        T6=Tt6/(1+(gamma-1)/2*M6**2)
        h6=Cpp*T6#h(T6)
        u6=numpy.sqrt(2*(ht6-h6))
    
        
        if isinstance(Mo,float) or isinstance(Mo,int):
            Mo = numpy.array([Mo])  
            array_flag = True
        else:
            array_flag = False      
    
        if numpy.linalg.norm(M6) < 1.0:  #nozzle unchoked
            
            p5=po
            M5=numpy.sqrt((((pt5/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
            T5=Tt5/(1+(gamma-1)/2*M5**2)
            h5=Cpp*T5
          
      
        else:
              
            M5=numpy.array([1.0]*len(Mo))
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
      
      
        if array_flag:
            M5 = numpy.array([M5])
            pt5 = numpy.array([pt5])
            pto = numpy.array([pto])
            Tt5 = numpy.array([Tt5])
            Tto = numpy.array([Tto])
            po = numpy.array([po])
        
        
        ue = numpy.array([0.0]*len(Mo))
        pe = numpy.array([0.0]*len(Mo))
        Ae_b_Ao = numpy.array([0.0]*len(Mo))
            
        
        for ii in range(len(Mo)):
            if Mo[ii] < 0.95:
                Ae_b_Ao[ii]=(fm_id(Mo[ii])/fm_id(M5[ii])*(1/(pt5[ii]/pto[ii]))*(numpy.sqrt(Tt5[ii]/Tto[ii])))
                ue[ii] = u5 # this is an array later
            else:
                # sudo code start
                p9 = po[ii]
                pt9 = pt5[ii]
                M9 = numpy.sqrt((((pt9/po[ii])**((gamma-1)/gamma))-1)*2/(gamma-1))
                Tt9 = Tt5[ii]
                T9 = Tt9/(1+(gamma-1)/2*M9**2)
                ht9 = Cpp*Tt9
                h9 = Cpp*T9
                u9=numpy.sqrt(2*(ht9-h9))
                rho9=p9/(R*T9)                
                # sudo code end
                Ae_b_Ao[ii]=(fm_id(Mo[ii])/fm_id(M9)*(1/(pt9/pto[ii]))*(numpy.sqrt(Tt9/Tto[ii])))
                ue[ii] = u9
                pe[ii] = p9
             
        Thrust_nd=gamma*Mo**2*(ue/uo-1)+Ae_b_Ao*(pe/po-1)
        
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
      
      
      
      #end
        self.A2= A2
        self.df= df
        self.nacelle_dia = df
        self.A2_5= A2_5
        self.dhc= dhc
        #self.A7= A7
        self.A5= A5
        self.Ao=Ao
        self.mdt= mhtD
        self.mlt= mltD
        self.mdf=mdfD
        self.mdlc=mdlcD
        self.D=numpy.sqrt(A2/(numpy.pi/4))
    
      
        print ' Areas ', self.Ao
        #Turbofan.sfc=sfc
        #Turbofan.thrust=th  
        self.mdhc=mdhcD
      
    def __call__(self,eta,State):
         
        Minf_a=State.M
        Tinf_a=State.T
        pinf_a=State.p
        
        #print Minf_a
        #print 'Tinf',Tinf_a
        
        #grav=State.g0
        #print 'grav',grav
        #arr_len=len(grav)
        
        
        arr_len=len(Minf_a)
        
        
        
        TSFC_a=numpy.empty(arr_len)
        FD_a=numpy.empty(arr_len)
        mdot_core_a=numpy.empty(arr_len)   
        mfuel_a=numpy.empty(arr_len)
        CF=numpy.empty(arr_len)
        Isp=numpy.empty(arr_len)
        #eta_Pe=numpy.empty(arr_len)
        
        
        
        
        pid=self.diffuser_pressure_ratio
        pif=self.fan_pressure_ratio
        pifn=self.fan_nozzle_pressure_ratio
        pilc=self.lpc_pressure_ratio
        pihc=self.hpc_pressure_ratio
        pib=self.burner_pressure_ratio
        pitn=self.turbine_nozzle_pressure_ratio
        Tt4=self.Tt4
        #aalpha=self.bypass_ratio
        aalpha=0 # TURBOJETT
        mdhc=self.mdhc  
        A22=self.A2 
        no_eng=self.number_of_engines
        
        throttle=eta
       
            
        Minf=Minf_a
        Tinf=Tinf_a
        pinf=pinf_a            
        #throttle=eta[ln]
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
    
        if numpy.linalg.norm(M8) < 1.0:
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
        if isinstance(Mo,float) or isinstance(Mo,int):
            Mo = numpy.array([Mo]) 
            array_flag = True
        else:
            array_flag = False   
            
        if numpy.linalg.norm(M6) < 1.0:  #nozzle unchoked
            
            p5=po
            M5=numpy.sqrt((((pt5/po)**((gamma-1)/gamma))-1)*2/(gamma-1))
            T5=Tt5/(1+(gamma-1)/2*M5**2)
            h5=Cpp*T5
          
      
        else:
            M5=numpy.array([1.0]*len(Mo))
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
        
        
        if array_flag:
            M5 = numpy.array([M5])
            pt5 = numpy.array([pt5])
            pto = numpy.array([pto])
            Tt5 = numpy.array([Tt5])
            Tto = numpy.array([Tto])
            po = numpy.array([po])
        
        
        ue = numpy.array([0.0]*len(Mo))
        pe = numpy.array([0.0]*len(Mo))
        Ae_b_Ao = numpy.array([0.0]*len(Mo))
            
        
        for ii in range(len(Mo)):
            if Mo[ii] < 0.95:
                Ae_b_Ao[ii]=(fm_id(Mo[ii])/fm_id(M5[ii])*(1/(pt5[ii]/pto[ii]))*(numpy.sqrt(Tt5[ii]/Tto[ii])))
                ue[ii] = u5[ii]
            else:
                # Create supersonic exit conditions
                p9 = po[ii]
                pt9 = pt5[ii]
                M9 = numpy.sqrt((((pt9/po[ii])**((gamma-1)/gamma))-1)*2/(gamma-1))
                Tt9 = Tt5[ii]
                T9 = Tt9/(1+(gamma-1)/2*M9**2)
                ht9 = Cpp*Tt9
                h9 = Cpp*T9
                u9=numpy.sqrt(2*(ht9-h9))
                rho9=p9/(R*T9)                
                Ae_b_Ao[ii]=(fm_id(Mo[ii])/fm_id(M9)*(1/(pt9/pto[ii]))*(numpy.sqrt(Tt9/Tto[ii])))
                ue[ii] = u9
                pe[ii] = p9
             
        Thrust_nd=gamma*Mo**2*(ue/uo-1)+Ae_b_Ao*(pe/po-1)
        
        
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
        
        #if min(throttle) < 0:
            #print throttle
            #print('Negative Throttle')
            ##raise('Negative Throttle')

        
        
        FD2=Fsp*ao*(1+aalpha)*mdot_core*no_eng*throttle
        mfuel=0.1019715*FD2*TSFC/3600
        #State.config.A_engine=A22
        
        #CF[ln] = FD2/(State.q[ln]*self.A2)
        #Isp[ln] = FD2/(mfuel*State.g0)   
        
        CF = FD2/(State.q*self.A2)
        if throttle.all() == 0.0:
            Isp = 0.0
        else:
            Isp = FD2/(mfuel*State.g0*State.q/State.q)        
        
        
        #-------------------------------------------------------
        ##Turbofan.sfc = TSFC
        ##Turbofan.thrust = FD2
        ##Turbofan.mdot_core = mdot_core
        ##print TSFC
        #TSFC_a[ln]=TSFC
        #FD_a[ln]=FD2
        #mdot_core_a[ln]=mdot_core
        #mfuel_a[ln]=mfuel
        ##eta_Pe[ln]=0.0
        
        #deindent indentation
        
        #dei
      
        #return FD_a,TSFC_a,mfuel_a
        eta_Pe = 0.0
        
        return CF, Isp, eta_Pe    
    
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