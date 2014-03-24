
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
        