## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# noise_tone_correction.py
#
# Created:  Jul 2015, C. Ilario
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Noise Tone Correction
# ----------------------------------------------------------------------

## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
def noise_tone_correction(SPL):
    """This method calculates de correction for spectral irregularities by means of
        a correction tone factor
        
    Assumptions:
        None
    
    Source:
        None 

    Inputs:
        SPL                     - Sound Pressure Level in 1/3 octave band

    Outputs: 
        tone_correction_max     - Maximum tone correction for a time history signal 
        
    Properties Used:
        N/A     
    """
        
        
    # Defining the necessary arrays for the tone correction procedure
    nsteps              = len(SPL)
    slope               = np.zeros(23)
    aux_ds              = np.zeros(23)
    delta_slope         = np.zeros(23)
    tone_correction_max = np.zeros(nsteps)
    
    for j in range(0,nsteps):
    
        #------------------------------------------------------------
        #STEP 1 - Calculation of slopes in the one-third octave bands
        #------------------------------------------------------------
        for i in range(3,23):
            slope[i] = SPL[j][i]-SPL[j][i-1]
        
        #------------------------------------------------------------
        #STEP 2 - Encircle the necessary values of the slope
        #------------------------------------------------------------    
        for i in range(3,23):        
            aux_ds[i] = np.abs(slope[i]-slope[i-1])
            
            if aux_ds[i]>5:
                delta_slope[i]=1
            else:
                delta_slope[i]=0
        #------------------------------------------------------------
        #STEP 3 - Encircle the slope
        #------------------------------------------------------------
        step3  = np.zeros(23)
        step3a = np.zeros(23)
        step3b = np.zeros(23)
        for i in range(3,23):
            if delta_slope[i]==1 and slope[i]>0 and slope[i]>slope[i-1]:
                step3a[i] = 1
            if  delta_slope[i]==1 and slope[i]<=0 and slope[i-1]>0:
                step3b[i] = 1
        step3 = step3a + step3b
        
        #------------------------------------------------------------
        #STEP 4 - Compute new adjusted sound pressure level
        #------------------------------------------------------------        
        step4 = np.zeros(23)
        for i in range(1,23):
            if step3[i]!=0 and i<23:
                step4[i] = (SPL[j][i-1]+SPL[j][i+1])/2
            if step3[i]!=0 and i==23:
                step4[i] = SPL[j][i-1]+slope[i-1]
            if step3[i]==0:
                step4[i] = SPL[j][i]
                
        #------------------------------------------------------------
        #STEP 5 - Recompute new slope
        #------------------------------------------------------------    
        step5 = np.zeros(25)
        for i in range(3,23):
            step5[i]=step4[i]-step4[i-1]
        step5[2]  = step5[3]
        step5[24] = step5[23]
        
        #------------------------------------------------------------
        #STEP 6 - Compute the arithmetic average of the three adjacent slopes
        #------------------------------------------------------------
        step6 = np.zeros(23)
        for i in range(2,22):
            if i==22:
                step6[i] = (step5[i]+step5[i+1])/3.
            else:
                step6[i] = (step5[i]+step5[i+1]+step5[i+2])/3.
        
        #------------------------------------------------------------
        #STEP 7 - Compute the final 1/3 octave band
        #------------------------------------------------------------
        step7 = np.zeros(24)
        step7[2]=SPL[j][2]
        for i in range(3,23):
            step7[i] = step7[i-1]+step6[i-1]
        
        #------------------------------------------------------------
        #STEP 8 - Compute the differences between original SPL and final SPL
        #------------------------------------------------------------    
        step8 = np.zeros(24)
        step8_aux = np.zeros(24)
        for i in range(2,16):
            step8_aux[i] = SPL[j][i]-step7[i]
            if step8_aux[i]>=1.5:
                step8[i] = step8_aux[i]
            else:
                step8[i]=0.
        for i in range(17,22):
            step8_aux[i] = SPL[j][i]-step7[i]
            if step8_aux[i]>=1.5 and SPL[j][i]>0 and SPL[j][i+1]>0 and SPL[j][i-1]>0:
                step8[i] = step8_aux[i]
            else:
                step8[i] = 0.
        
        step8_aux[23] = SPL[j][23]-step7[23]
        if step8_aux[23]>=1.5 and SPL[j][23]>0 and SPL[j][22]>0:
            step8[23] = step8_aux[23]
        else:
            step8[23]=0.
            
        #------------------------------------------------------------
        #STEP 9 - Determine tone correction factors for each 1/3 octave band
        #------------------------------------------------------------
        tone_correction = np.zeros(23)
        for i in range(2,9):
            if step8[i]>=1.5 and step8[i]<3:
                tone_correction = (step8[i]/3)-0.5
            if step8[i]>=3 and step8[i]<20:
                tone_correction = step8[i]/6.
            if step8[i]>20:
                tone_correction = 3+(1/3)
        for i in range(10,20):
            if step8[i]>=1.5 and step8[i]<3:
                tone_correction = (2/3)*(step8[i])-1
            if step8[i]>=3 and step8[i]<20:
                tone_correction = step8[i]/3.
            if step8[i]>20:
                tone_correction = 6+(2/3)
        for i in range(21,23):
            if step8[i]>=1.5 and step8[i]<3:
                tone_correction = (step8[i]/3)-(1/2)
            if step8[i]>=3 and step8[i]<20:
                tone_correction = step8[i]/6.
            if step8[i]>20:
                tone_correction = 3+(1/3)
                
        #------------------------------------------------------------
        #STEP 10 - Largest tone correction factor
        #------------------------------------------------------------
        tone_correction_max[j] = np.max(tone_correction)
    
        return tone_correction_max
    