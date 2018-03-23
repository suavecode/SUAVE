## @ingroup Input_Output-OpenVSP
# write_vsp_fea.py
# 
# Created:  Mar 2018, T. MacDonald
# Modified: 

try:
    import vsp_g as vsp
except ImportError:
    pass # This allows SUAVE to build without OpenVSP
import numpy as np
import time
import fileinput

## @ingroup Input_Output-OpenVSP
def write_vsp_fea(geometry,tag):
    """
    
    Assumptions:
    None

    Source:
    N/A

    Inputs:

    Outputs:                             

    Properties Used:
    N/A
    """      
    
    # Reset OpenVSP to avoid including a previous vehicle
    vsp.ClearVSPModel()    
    
    vsp.ReadVSPFile(tag + '.vsp3')
    vehicle_id = vsp.FindContainersWithName('Vehicle')[0]
    
    wing_id = vsp.FindContainersWithName('main_wing')[0]
    
    wing_fea_id = vsp.AddFeaStruct(wing_id)
    
    spar_id = vsp.AddFeaPart(wing_id,wing_fea_id,vsp.FEA_SPAR)
    spar_limit_parm = vsp.FindParm(spar_id,'LimitSparToSectionFlag','FeaSpar')
    spar_start_parm = vsp.FindParm(spar_id,'StartWingSection','FeaSpar')
    spar_end_parm = vsp.FindParm(spar_id,'StartWingSection','FeaSpar')
    vsp.SetParmVal(spar_limit_parm,1.0)
    vsp.SetParmVal(spar_start_parm,1)
    vsp.SetParmVal(spar_end_parm,2)
    vsp.GetParmVal(spar_limit_parm)
    
    vsp.WriteVSPFile(tag + "_fea_test_write.vsp3")
    
    pass
    
if __name__ == '__main__':
    tag = '/home/tim/Documents/SST_Design/Wing_Weights_Testing/Wing_Creation/SUAVE_CRM'
    write_vsp_fea(None,tag)