# noise_propeller_sae.py
#
# Created:  Oct 2016, C. Ilario
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np

def noise_propeller_sae():

    diameter = 13.5 #ft
    n_blades = 4
    HP = 3260 #SHP
    rpm = 1020
    dist = 584 #ft
    azim = 105 #deg
    speed = 180 #kts
    temperature = 86 #F
    n_propellers = 4
    tip_speed = 720 #ft/s
    tip_mach = 0.64
    sound_speed = 1116.4698 #ft/s
    
    #Correction for the number of propellers:
    if n_propellers == 1:
        NC = 0
    elif n_propellers == 2:
        NC = 3
    elif n_propellers == 4:
        NC = 6
    
    #Calculation of the helical tip speed
    Vtip = np.sqrt(speed**2 + tip_speed**2)
    Vtip_Mach = Vtip/sound_speed
    
    
    
    #***** Figure 3 *********************
    if tip_mach <= 0.4:
        FL1 = 6.7810306101*np.log(HP)+32.4536847808
    elif tip_mach <= 0.5:
        FL1 = 6.7656398511*np.log(HP)+36.4477233368
    elif tip_mach <= 0.6:
        FL1 = 6.6832881888*np.log(HP)+40.5886386121
    elif tip_mach <= 0.7:
        FL1 = 6.7439113333*np.log(HP)+43.5568035724
    elif tip_mach <= 0.8:
        FL1 = 6.8401765646*np.log(HP)+46.8530951754
    elif tip_mach <= 0.9:
        FL1 = 6.8447630205*np.log(HP)+50.7481250789
        
    #***** Figure 4 *********************
            
    if n_blades == 2:
        FL2 = -8.683066391*np.log(diameter)+26.6099964054
    elif n_blades == 3:
        FL2 = -8.7158230649*np.log(diameter)+23.7536138121
    elif n_blades == 4:
        FL2 = -8.7339566501*np.log(diameter)+20.6709100907
    elif n_blades == 6:
        FL2 = -8.6437428565*np.log(diameter)+17.6059581647
    elif n_blades == 8:
        FL2 = -8.7158865083*np.log(diameter)+15.2385549706
    
    #***** Figure 5 *********************
    #Perceived Noise Level and dbA
    FL3_1 = -10.6203648721*np.log(dist) + 63.9095683153
    #Overall sound pressure level
    FL3_2 = -9.0388679389*np.log(dist) + 55.9177440082
        
    #***** Figure 6 *********************
    
    DI = 0.000000003*(azim**5) - 1.5330661136843e-6*(azim**4) + 0.0002748446*(azim**3) - 0.0224465579*(azim**2) + 0.8841215644*azim -15.0851264829
    
    #***** Figure 7 *********************
    if Vtip_Mach <= 0.5:
        PNL_factor = -0.0002194325*(diameter)**4 + 0.0119121744*diameter**3 - 0.1715062582*diameter**2 - 0.440364203*diameter + 6.2098100491
    elif Vtip_Mach <= 0.6:   
        PNL_factor = -0.000244323*(diameter)**4 + 0.0152241341*diameter**3 - 0.2925615622*diameter**2 + 1.1285073819*diameter + 1.4783088763
    elif Vtip_Mach <= 0.7:   
        PNL_factor = -9.19201576386925e-5*(diameter)**4 + 0.007125071*diameter**3 - 0.1605227186*diameter**2 +0.5589157992*diameter + 2.4062117898
    elif Vtip_Mach <= 0.8:
        PNL_factor = -9.71380207504857e-6*(diameter)**4 + 0.0013799409*diameter**3 - 0.0240602419*diameter**2 -0.6106811004*diameter + 6.8595599903
    elif Vtip_Mach <= 0.85:
        PNL_factor = 0.0005259053*(diameter)**3 - 0.0058445988*diameter**2 -0.6296392506*diameter + 7.3325512636
    elif Vtip_Mach <= 0.9:
        PNL_factor = 0.0001869235*(diameter)**4 - 0.0108618115*diameter**3 + 0.2292260658*diameter**2 - 2.468037691*diameter + 13.2430172278
    
    OASPL = FL1+FL2+FL3_2+DI+NC
    PNL = OASPL + PNL_factor
    PNL_dBA = PNL - 14
    EPNdB_takeoff = PNL - 4
    EPNdB_landing = PNL - 2
    
    print PNL , PNL_dBA, OASPL, EPNdB_landing,EPNdB_takeoff
    
    return

