# write_vsp_mesh.py
# 
# Created:  Oct 2016, T. MacDonald
# Modified: Jan 2017, T. MacDonald

try:
    import vsp_g as vsp
except ImportError:
    pass # This allows SUAVE to build without OpenVSP
import numpy as np
import time
import fileinput

def write_vsp_mesh(geometry,tag,half_mesh_flag,growth_ratio,growth_limiting_flag):
    
    # Reset OpenVSP to avoid including a previous vehicle
    vsp.ClearVSPModel()

    # Turn on symmetry plane splitting to improve robustness of meshing process
    if half_mesh_flag == True:
        f = fileinput.input(tag + '.vsp3',inplace=1)
        for line in f:
            if 'SymmetrySplitting' in line:
                print line[0:34] + '1' + line[35:-1]
            else:
                print line
    
    vsp.ReadVSPFile(tag + '.vsp3')
    
    # Set output file types and what will be meshed
    file_type = vsp.CFD_STL_TYPE + vsp.CFD_KEY_TYPE
    set_int   = vsp.SET_ALL

    vsp.SetComputationFileName(vsp.CFD_STL_TYPE, tag + '.stl')
    vsp.SetComputationFileName(vsp.CFD_KEY_TYPE, tag + '.key')
    
    # Set to create a tagged STL mesh file
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
        
    vsp.SetCFDMeshVal(vsp.CFD_FAR_SIZE_ABS_FLAG,1)
    vsp.SetCFDMeshVal(vsp.CFD_FAR_LENGTH,far_length)
    vsp.SetCFDMeshVal(vsp.CFD_FAR_WIDTH,far_length)
    vsp.SetCFDMeshVal(vsp.CFD_FAR_HEIGHT,far_length)    
    vsp.SetCFDMeshVal(vsp.CFD_FAR_MAX_EDGE_LEN, max_len)
    vsp.SetCFDMeshVal(vsp.CFD_GROWTH_RATIO, growth_ratio)
    if growth_limiting_flag == True:
        vsp.SetCFDMeshVal(vsp.CFD_LIMIT_GROWTH_FLAG, 1.0)
    
    # Set the max edge length so we have on average 50 elements per chord length
    MAC     = geometry.wings.main_wing.chords.mean_aerodynamic
    min_len = MAC/50.
    vsp.SetCFDMeshVal(vsp.CFD_MAX_EDGE_LEN,min_len)
    
    # vsp.AddDefaultSources()   
    SetSources(geometry)
    
    vsp.Update()
    
    vsp.WriteVSPFile(tag + '_premesh.vsp3')
    
    print 'Starting mesh for ' + tag
    ti = time.time()
    vsp.ComputeCFDMesh(set_int,file_type)
    tf = time.time()
    dt = tf-ti
    print 'VSP meshing for ' + tag + ' completed in ' + str(dt) + ' s'
    
def SetSources(geometry):
    # Extract information on geometry type (for some reason it seems VSP doesn't have a simple 
    # way to do this)
    comp_type_dict = dict()
    comp_dict      = dict()
    for wing in geometry.wings:
        comp_type_dict[wing.tag] = 'wing'
        comp_dict[wing.tag] = wing
    for fuselage in geometry.fuselages:
        comp_type_dict[fuselage.tag] = 'fuselage'
        comp_dict[fuselage.tag] = fuselage
    # Propulsor sources have not been implemented
    #for propulsor in geometry.propulsors:
        #comp_type_dict[propulsor.tag] = 'turbojet'
        #comp_dict[propulsor.tag] = propulsor
        
    components = vsp.FindGeoms()
    
    # The default source values are (mostly) based on the OpenVSP scripts, wing for example:
    # https://github.com/OpenVSP/OpenVSP/blob/a5ac5302b320e8e318830663bb50ba0d4f2d6f64/src/geom_core/WingGeom.cpp
    
    for comp in components:
        comp_name = vsp.GetGeomName(comp)
        comp_type = comp_type_dict[comp_name]
        # Nacelle sources are not implemented
        #if comp_name[0:8] == 'turbofan':
            #comp_type = comp_type_dict[comp_name[0:8]]
        #else:
            #comp_type = comp_type_dict[comp_name]
        if comp_type == 'wing':
            wing = comp_dict[comp_name]
            if len(wing.Segments) == 0: # check if segments exist
                num_secs = 1
                use_base = True
            else:
                if wing.Segments[0].percent_span_location == 0.: # check if first segment starts at the root
                    num_secs = len(wing.Segments)
                    use_base = False
                else:
                    num_secs = len(wing.Segments) + 1
                    use_base = True
                    
            u_start = 0.
            base_root = wing.chords.root
            base_tip  = wing.chords.tip            
            for ii in xrange(0,num_secs):
                if (ii==0) and (use_base == True): # create sources on root segment
                    cr = base_root
                    if len(wing.Segments) > 0:
                        ct = base_root  * wing.Segments[0].root_chord_percent
                        seg = wing.Segments[ii]
                    else:
                        if wing.has_key('vsp_mesh'):
                            custom_flag = True
                        else:
                            custom_flag = False
                        ct = base_tip           
                        seg = wing
                    # extract CFD source parameters
                    if len(wing.Segments) == 0:
                        wingtip_flag = True
                    else:
                        wingtip_flag = False
                    AddSegmentSources(comp,cr, ct, ii, u_start, num_secs, custom_flag, 
                                  wingtip_flag,seg)                        
                elif (ii==0) and (use_base == False): 
                    cr = base_root * wing.Segments[0].root_chord_percent
                    if num_secs > 1:
                        ct = base_root  * wing.Segments[1].root_chord_percent
                    else:
                        ct = base_tip
                    # extract CFD source parameters
                    seg = wing.Segments[ii]
                    if wing.Segments[ii].has_key('vsp_mesh'):
                        custom_flag = True
                    else:
                        custom_flag = False
                    wingtip_flag = False
                    AddSegmentSources(comp,cr, ct, ii, u_start, num_secs, custom_flag, 
                                  wingtip_flag,seg)
                elif ii < num_secs - 1:
                    if use_base == True:
                        jj = 1
                    else:
                        jj = 0
                    cr = base_root * wing.Segments[ii-jj].root_chord_percent
                    ct = base_root * wing.Segments[ii+1-jj].root_chord_percent
                    seg = wing.Segments[ii-jj]
                    if wing.Segments[ii-jj].has_key('vsp_mesh'):
                        custom_flag = True
                    else:
                        custom_flag = False
                    wingtip_flag = False
                    AddSegmentSources(comp,cr, ct, ii, u_start, num_secs, custom_flag, 
                                  wingtip_flag,seg)                   
                else:     
                    if use_base == True:
                        jj = 1
                    else:
                        jj = 0                    
                    cr = base_root * wing.Segments[ii-jj].root_chord_percent
                    ct = base_tip
                    seg = wing.Segments[ii-jj]
                    if wing.Segments[ii-jj].has_key('vsp_mesh'):
                        custom_flag = True
                    else:
                        custom_flag = False
                    wingtip_flag = True
                    AddSegmentSources(comp,cr, ct, ii, u_start, num_secs, custom_flag, 
                                  wingtip_flag,seg)  
                pass
                    
        elif comp_type == 'fuselage':
            fuselage = comp_dict[comp_name]
            if fuselage.has_key('vsp_mesh'):
                len1 = fuselage.vsp_mesh.length
                rad1 = fuselage.vsp_mesh.radius
            else:
                len1 = 0.1 * 0.5 # not sure where VSP is getting this value
                rad1 = 0.2 * fuselage.lengths.total
            uloc = 0.0
            wloc = 0.0
            vsp.AddCFDSource(vsp.POINT_SOURCE,comp,0,len1,rad1,uloc,wloc) 
            uloc = 1.0
            vsp.AddCFDSource(vsp.POINT_SOURCE,comp,0,len1,rad1,uloc,wloc) 
            pass
        
        # This is a stub for the nacelle implementation. It will create sources
        # as is but they will not be appropriate for the nacelle shape.
        
        #elif comp_type == 'turbofan':
            #propulsor = comp_dict[comp_name[0:8]]
            #if propulsor.has_key('vsp_mesh'):
                #len1 = propulsor.vsp_mesh.length
                #rad1 = propulsor.vsp_mesh.radius
            #else:
                #len1 = 0.1 * 0.5 # not sure where VSP is getting this value
                #rad1 = 0.2 * propulsor.engine_length
            #uloc = 0.0
            #wloc = 0.0
            #vsp.AddCFDSource(vsp.POINT_SOURCE,comp,0,len1,rad1,uloc,wloc) 
            #uloc = 1.0
            #vsp.AddCFDSource(vsp.POINT_SOURCE,comp,0,len1,rad1,uloc,wloc) 
            #pass        
    
        

def AddSegmentSources(comp,cr,ct,ii,u_start,num_secs,custom_flag,wingtip_flag,seg):
    if custom_flag == True:
        len1 = seg.vsp_mesh.inner_length
        len2 = seg.vsp_mesh.outer_length
        rad1 = seg.vsp_mesh.inner_radius
        rad2 = seg.vsp_mesh.outer_radius
    else:
        len1 = 0.01 * cr
        len2 = 0.01 * ct
        rad1 = 0.2 * cr
        rad2 = 0.2 * ct
    uloc1 = ((ii+1)+u_start-1 +1)/(num_secs+2) # index additions are shown explicitly for cross-referencing with VSP code
    wloc1 = 0.5
    uloc2 = ((ii+1)+u_start +1)/(num_secs+2)
    wloc2 = 0.5
    vsp.AddCFDSource(vsp.LINE_SOURCE,comp,0,len1,rad1,uloc1,wloc1,len2,rad2,uloc2,wloc2)
    wloc1 = 0.
    wloc2 = 0.
    TE_match = True
    if (custom_flag == True) and (seg.vsp_mesh.has_key('matching_TE')):
        if seg.vsp_mesh.matching_TE == False: # use default values if so
            vsp.AddCFDSource(vsp.LINE_SOURCE,comp,0,0.01 * cr,0.2 * cr,uloc1,wloc1,0.01 * ct,0.2 * ct,uloc2,wloc2) 
            TE_match = False
        else:
            vsp.AddCFDSource(vsp.LINE_SOURCE,comp,0,len1,rad1,uloc1,wloc1,len2,rad2,uloc2,wloc2)
    else:
        vsp.AddCFDSource(vsp.LINE_SOURCE,comp,0,len1,rad1,uloc1,wloc1,len2,rad2,uloc2,wloc2)  
    if wingtip_flag == True:
        len1 = len2
        rad1 = rad2
        wloc1 = 0.0
        wloc2 = 0.5
        uloc1 = uloc2
        if TE_match == False: # to match not custom TE if indicated
            len1 = 0.01 * ct
            rad1 = 0.2 * ct
        vsp.AddCFDSource(vsp.LINE_SOURCE,comp,0,len1,rad1,uloc1,wloc1,len2,rad2,uloc2,wloc2)    
    
if __name__ == '__main__':
    write_vsp_mesh(tag,True)
