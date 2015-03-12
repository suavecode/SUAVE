# test_propulsion.py

import SUAVE
#import test_b737_pass
#import Vehicle
#from SUAVE.Analyses.Missions import Segment
#from SUAVE.Attributes.Results import State
from SUAVE.Components.Propulsors.Turbofan import TurboFanPASS
#from SUAVE.Components.Propulsors import Ductedfan
from SUAVE.Methods.Propulsion import engine_analysis
from SUAVE.Methods.Propulsion import engine_sizing
#from SUAVE.Methods.Propulsion import engine_analysis_ductedfan
#from SUAVE.Methods.Propulsion import engine_sizing_ductedfan
from SUAVE.Methods.Propulsion import engine_sizing_1d  
from SUAVE.Methods.Propulsion import engine_analysis_1d

from SUAVE.Core import (
Data, Container, Data_Exception, Data_Warning,
)


import numpy as np
# MAIN


# TEST PROPULSION
def test():
    
    #Vehicle = test_b737_pass.create_av()
    Turbofan=TurboFanPASS()
    #print 'Sea level static thrust - ' , Turbofan.thrust_sls
    #Turbofan.thrust_sls=20000
    #Turbofan.sfc_sfcref=1.0
    #Turbofan.type=1
    #Seg=Segment()
    #Seg.mach=0.8
    #Seg.alt=8
    
    #st=State()
    
    st=Data()
    
    st.M=0.8
    st.alt=8.0
    st.T=218.0
    st.p=0.239*10**5
    
    Turbofan.analysis_type == '1D'
    Turbofan.diffuser_pressure_ratio = 0.98
    Turbofan.fan_pressure_ratio = 1.7
    Turbofan.fan_nozzle_pressure_ratio = 0.99
    Turbofan.lpc_pressure_ratio = 1.14
    Turbofan.hpc_pressure_ratio = 13.415
    Turbofan.burner_pressure_ratio = 0.95
    Turbofan.turbine_nozzle_pressure_ratio = 0.99
    Turbofan.Tt4 = 1450.0 #1350
    Turbofan.bypass_ratio = 5.4
    Turbofan.design_thrust = 24000
    
    Turbofan.engine_sizing_1d(st) 
    #engine_sizing(Turbofan,s
    
    
    print Turbofan.mdhc
    
    #print('Engine Geometry')
    #print 'Engine bare_engine_dia - ' , Turbofan.bare_engine_dia
    #print 'Engine bare_engine_length - ' ,Turbofan.bare_engine_length
    #print 'Engine nacelle dia - ' ,Turbofan.nacelle_dia
    #print 'Engine inlet length - ' ,Turbofan.inlet_length
    #print 'Engine max area - ' ,Turbofan.eng_maxarea
    #print 'Engine inlet area - ' ,Turbofan.inlet_area      
    #print 'Engine upper surf shape factor - ' ,Turbofan.upper_surf_shape_factor    
    
    # --- Conditions        
    ones_1col = np.ones([1,1])      
    
    st2=Data()
    
    st2.M=ones_1col*0.3
    st2.alt=ones_1col*0.5
    st2.T=ones_1col*258.0
    st2.p=ones_1col*10.0**5    
    st2.rho= st2.p/(287.87*st2.T)
    st2.a0 = np.sqrt(1.4*287.87*st2.T)
    st2.v = st2.M*st2.a0
    st2.q = 0.5* st2.rho*st2.v*st2.v
    st2.g0 =  9.81
    eta = ones_1col*1.0
    
    #Turbofan2=SUAVE.Methods.Propulsion.engine_analysis(Turbo_Fan,Segment)
    #engine_analysis(Turbofan,st2)
    [F,mdot,Isp] = Turbofan(eta,st2)
     
    
    #print('Engine analysis results')
    print 'thrust', F
    #print 'mission sfc -' , Turbofan.sfc
    #print 'mission thrust - ' ,Turbofan.thrust  
    
    
    
    
    
    
    
    
    
    
    
    #ductedfan=Ductedfan()
    #ductedfan.diffuser_pressure_ratio = 0.98
    #ductedfan.fan_pressure_ratio = 1.7
    #ductedfan.fan_nozzle_pressure_ratio = 0.99
    #ductedfan.design_thrust = 1.0  
    
    #engine_sizing_ductedfan(ductedfan,st)
    
    #engine_analysis_ductedfan(ductedfan,st2)
    #print 'Ducted Fan thrust ', ductedfan.thrust

def main():
    
    test()
if __name__ == '__main__':
    main()      

    
   
#main()
    

