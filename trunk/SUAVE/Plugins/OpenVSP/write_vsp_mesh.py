import vsp_g as vsp
import numpy as np
import time
import fileinput

def write_vsp_mesh(tag,half_mesh_flag):
    
    vsp.ClearVSPModel()
    
    f = fileinput.input(tag + '.vsp3',inplace=1)
    for line in f:
        if 'SymmetrySplitting' in line:
            print line[0:34] + '1' + line[35:-1]
        else:
            print line
    
    vsp.ReadVSPFile(tag + '.vsp3')
    
    file_type = vsp.CFD_STL_TYPE + vsp.CFD_KEY_TYPE
    set_int   = vsp.SET_ALL

    vsp.SetComputationFileName(vsp.CFD_STL_TYPE, tag + '.stl')
    vsp.SetComputationFileName(vsp.CFD_KEY_TYPE, tag + '.key')
    
    vehicle_cont = vsp.FindContainer('Vehicle',0)
    STL_multi    = vsp.FindParm(vehicle_cont, 'MultiSolid', 'STLSettings')
    vsp.SetParmVal(STL_multi, 1.0)
    
    vsp.SetCFDMeshVal(vsp.CFD_FAR_FIELD_FLAG,1)
    if half_mesh_flag == True:
        vsp.SetCFDMeshVal(vsp.CFD_HALF_MESH_FLAG,1)
        
    # Figure out the size of the bounding box
    vehicle_id = vsp.FindContainersWithName('Vehicle')[0]
    xlen = vsp.GetParmVal(vsp.FindParm(vehicle_id,"X_Len","BBox"))
    ylen = vsp.GetParmVal(vsp.FindParm(vehicle_id,"Y_Len","BBox"))
    zlen = vsp.GetParmVal(vsp.FindParm(vehicle_id,"Z_Len","BBox"))
    
    # Max length
    max_len = np.max([xlen,ylen,zlen])
    far_length = 10.*max_len
        
    #vsp.SetCFDMeshVal(vsp.CFD_FAR_MAX_GAP, 0.005) # to prevent half mesh tail errors
    vsp.SetCFDMeshVal(vsp.CFD_FAR_SIZE_ABS_FLAG,1)
    vsp.SetCFDMeshVal(vsp.CFD_FAR_LENGTH,far_length)
    vsp.SetCFDMeshVal(vsp.CFD_FAR_WIDTH,far_length)
    vsp.SetCFDMeshVal(vsp.CFD_FAR_HEIGHT,far_length)    
    vsp.SetCFDMeshVal(vsp.CFD_FAR_MAX_EDGE_LEN, 30)
    
    vsp.AddDefaultSources()    
    
    print 'Starting mesh for ' + tag
    ti = time.time()
    vsp.ComputeCFDMesh(set_int,file_type)
    tf = time.time()
    dt = tf-ti
    print 'VSP meshing for ' + tag + ' completed in ' + str(dt) + ' s'
    
    
if __name__ == '__main__':
    
    tag = '/home/tim/Documents/SUAVE/scripts/experimental/SU2_link/cruise'
    write_vsp_mesh(tag,True)