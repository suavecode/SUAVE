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
    
    wing_tag = 'main_wing'
    
    wing_tags = [wing_tag]
    
    for wt in wing_tags:
        write_wing(geometry,wt)
    
    vsp.WriteVSPFile(tag + "_fea_test_write.vsp3")
    
    pass

def write_wing(geometry,wing_tag):
    
    wing_id = vsp.FindContainersWithName(wing_tag)[0]
    
    wing_fea_id = vsp.AddFeaStruct(wing_id)
    
    rib_locs = geometry.wings[wing_tag].structure.ribs.locations
    
    for rib_loc in rib_locs:
        rib_id = vsp.AddFeaPart(wing_id,wing_fea_id,vsp.FEA_RIB)
        rib_pos_type_flag = vsp.FindParm(rib_id,'AbsRelParmFlag','FeaPart')
        rib_loc_parm      = vsp.FindParm(rib_id,'AbsCenterLocation','FeaPart')
        vsp.SetParmVal(rib_pos_type_flag,0.0) # make rib position absolute
        vsp.SetParmVal(rib_loc_parm,float(rib_loc))        
    
    
    spar_locs = geometry.wings[wing_tag].structure.spars.locations
    num_segs = len(geometry.wings[wing_tag].Segments)-1
    
    for spar_loc in spar_locs:
        for i in range(num_segs):
            spar_id = vsp.AddFeaPart(wing_id,wing_fea_id,vsp.FEA_SPAR)
            spar_limit_parm = vsp.FindParm(spar_id,'LimitSparToSectionFlag','FeaSpar')
            spar_start_parm = vsp.FindParm(spar_id,'StartWingSection','FeaSpar')
            spar_end_parm   = vsp.FindParm(spar_id,'EndWingSection','FeaSpar')
            pos_parm        = vsp.FindParm(spar_id,'RelCenterLocation','FeaPart')
            vsp.SetParmVal(pos_parm,spar_loc)
            vsp.SetParmVal(spar_limit_parm,1.0) # set spars to be limited to specified sections
            vsp.SetParmVal(spar_end_parm,i+1)
            vsp.Update()
            vsp.SetParmVal(spar_start_parm,i+1)    
    
if __name__ == '__main__':
    tag = '/home/tim/Documents/SST_Design/Wing_Weights_Testing/Wing_Creation/SUAVE_CRM'
    write_vsp_fea(None,tag)