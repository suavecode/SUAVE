# propeller_noise_frequency.py
# 
# Created:  May 2016, Carlos
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import numpy as np
import scipy
import matplotlib.pyplot as plt

from scipy import special
from SUAVE.Core import Units, Data


def propeller_noise_frequency(noise_data):
    
    #-----------------------------------------------------
    #unpack
    
    number_sections    = noise_data.number_sections               #number of section on the blade
    blades_number      = noise_data.blades_number                  #number of blades
    propeller_diameter = noise_data.propeller_diameter           #propeller diameter [ft]
    thrust             = np.max(noise_data.thrust)       #Thrust [lbs]
    hpower             = np.max(noise_data.hpower)    #horsepower [hp]
    rpm                = np.max(noise_data.rpm)                        #shaft rotation frequency
    velocity           = np.max(noise_data.velocity)                       #flight speed
    
    MCA = 0.5*np.ones(number_sections)  #MCA (midchord of alignment or sweep) of the blade for each section
    FA = np.zeros(number_sections)  #face alignment or offset    
    
    r0 = noise_data.r0
    airfoil_chord = noise_data.airfoil_chord 
    lift_coefficient = noise_data.lift_coefficient[:][1]
    drag_coefficient = noise_data.drag_coefficient[:][1]
    

    n_harmonic = noise_data.n_harmonic #Number of harmonics to consider (Default = 10)
    
    sound_speed = np.max(noise_data.sound_speed) 
    density = np.max(noise_data.density)
    
    p_base=2.0*1e-5  
    zeff = 0.8
    
   # x_obs=5*propeller_diameter
   # y_obs=5*propeller_diameter

    dist_obser = noise_data.distance    #np.sqrt((x_obs*x_obs)+(y_obs*y_obs))
    theta = (noise_data.angle)*np.pi/180  #radiation angle from propeller axis to observer point
    
    
    dist_axis = dist_obser * np.sin(noise_data.angle) #y_obs                   #observer distance from propeller axis    
    
    
    tip_radius = 0.3048*(propeller_diameter/2.0) #propeller tip radius [m]
    
    Mx = velocity/sound_speed                   #flight Mach number
    
    frequency = 2*np.pi*(rpm/60.)                       #Frequency of the rotation blade
    frequency_doppler = frequency/(1-Mx*np.cos(theta))    #Doppler effect
    
    Mach_tip = frequency*tip_radius/sound_speed #tip rotational Mach number
    
    convection_factor = (1.0-Mx*np.cos(theta))  #convection factor
    
    #print "f = ", frequency
    #print "fd = ", frequency_doppler
    #print "Mach_tip = ", Mach_tip
    #print "convec = ", convection_factor
    #print "Mach_x = ", Mx
    
    
    #Sampling signal
    max_frequency = n_harmonic*blades_number*rpm/60
    nyquest_frequency = 2*max_frequency
    dt=1.0/nyquest_frequency
    
    tinit = 0.1
    sampling=np.ceil(1-tinit)/dt
    timej= np.linspace(tinit,0.5,num=sampling, endpoint=True) #[0.0001, 0.006, 0.011, 0.016, 0.021]
    
    # Loading the initial vectors
    total_pressure = 0.0
    total_monopole = 0.0
    total_drag = 0.0
    total_lift = 0.0
    SPL = np.zeros(n_harmonic+1)
    pressure = np.zeros(n_harmonic+1)
    pt_total=np.zeros(np.size(timej))
    pt_monopole=np.zeros(np.size(timej))
    pt_drag=np.zeros(np.size(timej))
    pt_lift=np.zeros(np.size(timej))
    
    p_2_total = 0.0
    p_2_monopole = 0.0
    p_2_drag = 0.0
    p_2_lift = 0.0
    
    #*********************
    # Loop over the time 
    for jj in xrange(0,np.size(timej)):
    #*********************
        time = timej[jj]    #time

        #print "time = ", time
        
        #************************************************
        # Loop over the number of harmonics of interest
        for m in xrange (1,n_harmonic+1):
        #************************************************
            
            Pvm_section = np.zeros(number_sections)
            Plm_section = np.zeros(number_sections)
            Pdm_section = np.zeros(number_sections)
            I_pdm = 0.0
            I_plm = 0.0
            I_pvm = 0.0
            Pmb = 0.0
           
            
    #        print "Harmonic mode = ", m, "BPF = ", m*blades_number*rpm/60 
            
            #Calculation of the constant term on the integral
            const = - density*sound_speed*sound_speed*blades_number*np.sin(theta)*np.exp(1j*m*blades_number*((frequency_doppler*dist_obser/sound_speed)-(np.pi/2.0))) \
                / (8.0*np.pi*(dist_axis/propeller_diameter)*convection_factor)        
    
      #      print "const = ", const
            
            #********************************************
            #Loop over the blade geometry - sections
            for i in xrange (0,number_sections):    
            #********************************************
                
                max_thickness = 0.094*airfoil_chord[i]
                z = r0[i]/tip_radius                        #normalized radial coordinate
                tb = max_thickness/airfoil_chord[i]         #section thickness to chord ratio
                Bd = airfoil_chord[i]/propeller_diameter    #section chord to diameter ratio
                
                Mach_section = np.sqrt((Mx*Mx)+(z*z*Mach_tip*Mach_tip))
            
               
                #Chordwise wave number
                kx = (2.0*m*blades_number*Bd*Mach_tip)/(Mach_section*convection_factor)
                
                #Second wave number
                ky = ((2*m*blades_number*Bd)/(z*Mach_section))*((Mach_section*Mach_section*np.cos(theta)-Mx)/convection_factor)
                
                #Phase shift due to sweep
                phi_zero = (2*m*blades_number/z*Mach_section)*((Mach_section*Mach_section*np.cos(theta)-Mx)/convection_factor)*FA[i]/propeller_diameter
                
                #Phase shift due to offset or face alignment
                phi_s = (2*m*blades_number*Mach_tip/Mach_section*convection_factor)*MCA[i]/propeller_diameter    
        
                
                #Calculation of the Bessel function:
                Arg    = m*blades_number*z*Mach_tip*np.sin(theta)/convection_factor #argument
                NOrder = m*blades_number    #order
                XJn    = scipy.special.jn(NOrder, Arg)
                
                if (kx==0):
                    Gama_v = 2.0/3.0
                    Gama_l = 1.0
                    Gama_d = 1.0
                else:
                    Gama_v = (8.0/(kx*kx))*((2.0/kx)*np.sin(kx*0.5)-np.cos(kx*0.5))
                    Gama_l = (2.0/kx)*np.sin(kx*0.5)
                    Gama_d = (2.0/kx)*np.sin(kx*0.5)
                
                
                pvm_factor = kx*kx*tb*Gama_v
                pdm_factor = 1j*kx*(drag_coefficient[i]*0.5)*Gama_d
                plm_factor = -1j*ky*(lift_coefficient[i]*0.5)*Gama_l
                
                const_integral = Mach_section*Mach_section*np.exp(1j*(phi_s+phi_zero))*XJn
                
                Pvm_section[i] = const_integral*pvm_factor 
                Pdm_section[i] = const_integral*pdm_factor 
                Plm_section[i] = const_integral*plm_factor
                
                #DIv_section[i] = XJn*Gama_v
                #DIl_section[i] = XJn*Gama_l
                            
           #     print "SECTION = ", i, "Max_thick = ", max_thickness, "tb = ", tb, "Bd = ", Bd, "z = ",z, "  Mach section = ", Mach_section, "cl = ", lift_coefficient[i], "drag = ", drag_coefficient[i]
               # print "z = ",z, "  Mach section = ", Mach_section
         #       print "Phi0 = ", phi_zero, "  Phis = ", phi_s
            
            for i in xrange(1,number_sections-2):
                I_pvm = I_pvm + Pvm_section[i]
                I_pdm = I_pdm + Pdm_section[i]
                I_plm = I_plm + Plm_section[i]
            
    #integration using trapezoidal rule       
            h = (r0[number_sections-1]-r0[0])/number_sections 
            Pvm = const*0.5*h*(Pvm_section[0]+Pvm_section[number_sections-1]+2.0*I_pvm)
            Pdm = const*0.5*h*(Pdm_section[0]+Pdm_section[number_sections-1]+2.0*I_pdm)
            Plm = const*0.5*h*(Plm_section[0]+Plm_section[number_sections-1]+2.0*I_plm)
            
            Pmb = (Pvm + Pdm + Plm)
            
            p_monopole = 2*np.real(Pvm*np.exp(-1j*m*blades_number*frequency_doppler*time))
            p_drag = 2*np.real(Pdm*np.exp(-1j*m*blades_number*frequency_doppler*time))
            p_lift = 2*np.real(Plm*np.exp(-1j*m*blades_number*frequency_doppler*time))
            
            p_total = 2*np.real(Pmb*np.exp(-1j*m*blades_number*frequency_doppler*time))
            
         #   print "monopole = ", p_monopole, "p_drag = ", p_drag, "p_lift = ", p_lift
            
          #  print "Freq = ", m*blades_number*rpm/60, "Pressure = ", pressure[m], "SPL = ", (20.0*np.log10(np.sqrt(pressure[m]**2)/(4.1784*1e-7))) #p_base)
            
            SPL_m_total = 20*np.log10(np.sqrt(p_total**2)/p_base)
            SPL_m_monopole = 20*np.log10(np.sqrt(p_monopole**2)/p_base)
            SPL_m_drag = 20*np.log10(np.sqrt(p_drag**2)/p_base)
            SPL_m_lift = 20*np.log10(np.sqrt(p_lift**2)/p_base)         
            
            if SPL_m_total<0.0:
                SPL_m_total = 0.0
            
            if SPL_m_drag<0.0:
                SPL_m_drag=0.0
            if SPL_m_lift<0.0:
                SPL_m_lift=0.0
            if SPL_m_monopole<0.0:
                SPL_m_monopole=0.0
            
           # print "frequency = ", m*blades_number*rpm/60, "SPL_total =", SPL_m_total, "SPL_monopole =", SPL_m_monopole, \
           #        "SPL_drag =", SPL_m_drag, "SPL_lift =", SPL_m_lift 
            
            total_monopole = p_monopole + total_monopole
            total_drag = p_drag + total_drag
            total_lift = p_lift + total_lift
            total_pressure = p_total + total_pressure
            
    #************************************************************************************************************************* 
    #***********************************************************************************************************************        
            #SPL Calculation
            
            Xbig = (m*blades_number*Mach_tip*Bd)/(Mach_section*convection_factor)
            
            plm = np.sin(Xbig)/Xbig
            
            #Calculation of the Bessel function:
            Arg    = m*blades_number*zeff*Mach_tip*np.sin(theta)/convection_factor #argument
            NOrder = m*blades_number    #order
            Jmb    = scipy.special.jn(NOrder, Arg)        
            sound_speed_ft = 1115.49
           
            A = 538673*m*blades_number*Mach_tip*np.sin(theta)/(dist_axis*propeller_diameter*convection_factor)
           
            B = (np.cos(theta)/convection_factor)*thrust-(550/(zeff*zeff*Mach_tip*Mach_tip*sound_speed_ft)*hpower)
                                                               
            SPL[m] = 20*np.log10(A*B*plm*Jmb)
            
         #   SPL1 = 20*np.log10(538673*m*blades_number*Mach_tip*np.sin(theta)/(dist_axis*propeller_diameter*convection_factor) \
         #                     *((np.cos(theta)*thrust/convection_factor)-(550*hpower/(zeff*zeff*Mach_tip*Mach_tip*sound_speed_ft))) \
         #                     *plm*Jmb)
            
           # print "A = ", A, "B = ", B,"plm = ",plm," bessel = ", Jmb, 
         #   print " noise = ",SPL, "noise1 = ", SPL1
    #*************************************************************************************************************************            
    #************************************************************************************************************************* 
    
    #        print "total_pressure",total_pressure
    #        print "total_monopole",total_monopole
    #        print "total_drag",total_drag
    #        print "total_lift",total_lift
            
        pt_total[jj]=total_pressure
        pt_monopole[jj]=total_monopole
        pt_drag[jj]=total_drag
        pt_lift[jj]=total_lift
        
        p_2_total = (total_pressure*total_pressure) + p_2_total
        p_2_monopole = (total_monopole*total_monopole) + p_2_monopole
        p_2_drag = (total_drag*total_drag) + p_2_drag
        p_2_lift = (total_lift*total_lift) + p_2_lift
        
       # print p_time[time]
        
        total_pressure = 0.0
        total_monopole = 0.0
        total_drag = 0.0
        total_lift = 0.0    
        
       
    
    p_rms_total = np.sqrt(np.mean(p_2_total))
    p_rms_monopole = np.sqrt(np.mean(p_2_monopole))
    p_rms_drag = np.sqrt(np.mean(p_2_drag))
    p_rms_lift = np.sqrt(np.mean(p_2_lift))
    
    # pack
    noise_result = Data()
    noise_result.p_rms_total = p_rms_total
    noise_result.p_rms_monopole = p_rms_monopole
    noise_result.p_rms_drag = p_rms_drag
    noise_result.p_rms_lift = p_rms_lift
    noise_result.SPL_total = 20*np.log10(p_rms_total/p_base)
    noise_result.SPL_monopole = 20*np.log10(p_rms_monopole/p_base)
    noise_result.SPL_drag = 20*np.log10(p_rms_drag/p_base)
    noise_result.SPL_lift = 20*np.log10(p_rms_lift/p_base)    
    noise_result.time = timej
    noise_result.pt_total=pt_total
    noise_result.pt_monopole=pt_monopole
    noise_result.pt_drag=pt_drag
    noise_result.pt_lift=pt_lift
    
    return (noise_result)

    #print "SPL_total =", 20*np.log10(p_rms_total/p_base)
    #print "SPL_monopole =", 20*np.log10(p_rms_monopole/p_base)
    #print "SPL_drag =", 20*np.log10(p_rms_drag/p_base)
    #print "SPL_lift =", 20*np.log10(p_rms_lift/p_base)
    
    #print SPL
    
    #plt.plot(timej,pt_total/p_base,'k', linewidth = 4, label="total")
    #plt.plot(timej,pt_drag/p_base,'r-',label="drag")
    #plt.plot(timej,pt_lift/p_base,'k-.',label="lift")
    #plt.plot(timej,pt_monopole/p_base,'b--',label="monopole")
    #plt.xlabel('time [s]')
    #plt.ylabel('Pressure')
    #plt.legend(["Total", "Drag", "Lift", "Monopole"])
    #plt.show()