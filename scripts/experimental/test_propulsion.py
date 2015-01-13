# test_propulsion.py

import SUAVE
#import test_b737_pass
#import Vehicle
from SUAVE.Analyses.Missions import Segment
from SUAVE.Attributes.Results import State
from SUAVE.Components.Propulsors import Turbo_Fan
from SUAVE.Components.Propulsors import Ductedfan
from SUAVE.Methods.Propulsion import engine_analysis
from SUAVE.Methods.Propulsion import engine_sizing
from SUAVE.Methods.Propulsion import engine_analysis_ductedfan
from SUAVE.Methods.Propulsion import engine_sizing_ductedfan
from SUAVE.Methods.Propulsion import engine_sizing_1d  
from SUAVE.Methods.Propulsion import engine_analysis_1d

# MAIN


# TEST PROPULSION
def test():
    
    #Vehicle = test_b737_pass.create_av()
    Turbofan=Turbo_Fan()
    #print 'Sea level static thrust - ' , Turbofan.thrust_sls
    #Turbofan.thrust_sls=20000
    #Turbofan.sfc_sfcref=1.0
    #Turbofan.type=1
    #Seg=Segment()
    #Seg.mach=0.8
    #Seg.alt=8
    
    st=State()
    
    st.M=0.8
    st.alt=8
    st.T=218
    st.p=0.239*10**5
    
    Turbofan.analysis_type == '1D'
    Turbofan.diffuser_pressure_ratio = 0.98
    Turbofan.fan_pressure_ratio = 1.7
    Turbofan.fan_nozzle_pressure_ratio = 0.99
    Turbofan.lpc_pressure_ratio = 1.14
    Turbofan.hpc_pressure_ratio = 13.415
    Turbofan.burner_pressure_ratio = 0.95
    Turbofan.turbine_nozzle_pressure_ratio = 0.99
    Turbofan.Tt4 = 1350
    Turbofan.bypass_ratio = 5.4
    Turbofan.design_thrust = 25000
    
    engine_sizing_1d(Turbofan,st) 
    #engine_sizing(Turbofan,st)
    
    
    #print Turbofan.mdhc
    
    #print('Engine Geometry')
    #print 'Engine bare_engine_dia - ' , Turbofan.bare_engine_dia
    #print 'Engine bare_engine_length - ' ,Turbofan.bare_engine_length
    #print 'Engine nacelle dia - ' ,Turbofan.nacelle_dia
    #print 'Engine inlet length - ' ,Turbofan.inlet_length
    #print 'Engine max area - ' ,Turbofan.eng_maxarea
    #print 'Engine inlet area - ' ,Turbofan.inlet_area      
    #print 'Engine upper surf shape factor - ' ,Turbofan.upper_surf_shape_factor    
    
    st2=State()
    
    st2.M=0.3
    st2.alt=0.5
    st2.T=258
    st2.p=10**5    
    
    
    #Turbofan2=SUAVE.Methods.Propulsion.engine_analysis(Turbo_Fan,Segment)
    #engine_analysis(Turbofan,st2)
    engine_analysis_1d(Turbofan,st2)
     
    
    #print('Engine analysis results')    
    print 'mission sfc -' , Turbofan.sfc
    print 'mission thrust - ' ,Turbofan.thrust  
    
    
    
    
    
    
    
    
    
    
    
    ductedfan=Ductedfan()
    ductedfan.diffuser_pressure_ratio = 0.98
    ductedfan.fan_pressure_ratio = 1.7
    ductedfan.fan_nozzle_pressure_ratio = 0.99
    ductedfan.design_thrust = 1.0  
    
    engine_sizing_ductedfan(ductedfan,st)
    
    engine_analysis_ductedfan(ductedfan,st2)
    print 'Ducted Fan thrust ', ductedfan.thrust

def main():
    
    test()
if __name__ == '__main__':
    main()      

    
   
#main()
    

