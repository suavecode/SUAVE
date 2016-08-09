# vsp_write_tube_and_wing.py
# 
# Created:  Jul 2016, T. MacDonald
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data

import vsp_g as vsp
import numpy as np

# ----------------------------------------------------------------------
#  write
# ----------------------------------------------------------------------

def write(vehicle):
    
    # Wings
    
    for wing in vehicle.wings:
    
        wing_x = wing.origin[0]    
        wing_y = wing.origin[1]
        wing_z = wing.origin[2]
        if wing.symmetric == True:
            span   = wing.spans.projected/2. # span of one side
        else:
            span   = wing.spans.projected
        root_chord = wing.chords.root
        tip_chord  = wing.chords.tip
        sweep      = wing.sweeps.quarter_chord / Units.deg
        sweep_loc  = 0.25
        root_twist = wing.twists.root / Units.deg
        tip_twist  = wing.twists.tip  / Units.deg
        root_tc    = wing.thickness_to_chord 
        tip_tc     = wing.thickness_to_chord 
        dihedral   = wing.dihedral / Units.deg
        
        wing_id = vsp.AddGeom( "WING" )
        
        if wing.symmetric == False:
            vsp.SetParmVal( wing_id,'Sym_Planar_Flag','Sym',0)
        if wing.vertical == True:
            vsp.SetParmVal( wing_id,'X_Rel_Rotation','XForm',90)           
        
        vsp.SetParmVal( wing_id,'X_Rel_Location','XForm',wing_x)
        vsp.SetParmVal( wing_id,'Y_Rel_Location','XForm',wing_y)
        vsp.SetParmVal( wing_id,'Z_Rel_Location','XForm',wing_z)
        vsp.SetParmVal( wing_id,'Span','XSec_1',span)
        vsp.SetParmVal( wing_id,'Root_Chord','XSec_1',root_chord)
        vsp.SetParmVal( wing_id,'Tip_Chord','XSec_1',tip_chord)
        vsp.SetParmVal( wing_id,'Sweep','XSec_1',sweep)
        vsp.SetParmVal( wing_id,'Sweep_Location','XSec_1',sweep_loc)
        vsp.SetParmVal( wing_id,'Twist','XSec_1',tip_twist) # tip
        vsp.SetParmVal( wing_id,'Twist','XSec_0',root_twist) # root
        vsp.SetParmVal( wing_id,'ThickChord','XSecCurve_0',root_tc)
        vsp.SetParmVal( wing_id,'ThickChord','XSecCurve_1',tip_tc)
        vsp.SetParmVal( wing_id,'Dihedral','XSec_1',dihedral)
        
        if wing.tag == 'main_wing':
            main_wing_id = wing_id
    
    # Pylons
    
    wing = vehicle.wings.main_wing
    
    pylon_wing_pos = 0.25
    pylon_y_off    = .4
    pylon_x_off    = 1    
    engine_nac_length = 2.5
    pylon_tc       = .2
    
    root_chord = wing.chords.root
    tip_chord  = wing.chords.tip
    sweep      = wing.sweeps.quarter_chord
    span       = wing.spans.projected
    
    dx_tip   = .25*root_chord + span/2.*np.tan(sweep)-.25*tip_chord
    dx_pylon = dx_tip*pylon_wing_pos
    
    pylon_id = vsp.AddGeom( "WING",main_wing_id)
    
    pylon_x = dx_pylon + wing.origin[0]
    pylon_y = span/2.*pylon_wing_pos
    pylon_z = wing.origin[2]
    
    pylon_chord = engine_nac_length/2.
    pylon_sweep = -np.arctan(pylon_x_off/pylon_y_off) / Units.deg
    
    vsp.SetParmVal( pylon_id,'X_Rel_Location','XForm',pylon_x)
    vsp.SetParmVal( pylon_id,'Y_Rel_Location','XForm',pylon_y)
    vsp.SetParmVal( pylon_id,'Z_Rel_Location','XForm',pylon_z)
    vsp.SetParmVal( pylon_id,'Span','XSec_1',pylon_y_off)
    vsp.SetParmVal( pylon_id,'Root_Chord','XSec_1',pylon_chord)
    vsp.SetParmVal( pylon_id,'Tip_Chord','XSec_1',pylon_chord)  
    vsp.SetParmVal( pylon_id,'X_Rel_Rotation','XForm',-90) 
    vsp.SetParmVal( pylon_id,'Sweep','XSec_1',pylon_sweep)
    vsp.SetParmVal( pylon_id,'Sweep_Location','XSec_1',0)   
    vsp.SetParmVal( pylon_id,'ThickChord','XSecCurve_0',pylon_tc)
    vsp.SetParmVal( pylon_id,'ThickChord','XSecCurve_1',pylon_tc) 
    
    # Engines
    
    nac_id = vsp.AddGeom( "FUSELAGE",pylon_id )
    
    # unpack the turbofan
    turbofan  = vehicle.propulsors.turbofan
    n_engines = turbofan.number_of_engines
    length    = turbofan.engine_length
    width     = turbofan.nacelle_diameter
    origins   = turbofan.origin
    bpr       = turbofan.bypass_ratio
    
    if n_engines == 2:
        symmetric = 1
    else:
        symmetric = 0
        
    z = pylon_z - width/2 - pylon_y_off
    x = pylon_x - pylon_y_off -  length/2
        
    # Length and overall diameter
    vsp.SetParmVal(nac_id,"Length","Design",length)
    vsp.SetParmVal(nac_id,"Diameter","Design",width)   
    vsp.SetParmVal(nac_id,'X_Rel_Location','XForm',x)
    vsp.SetParmVal(nac_id,'Y_Rel_Location','XForm',pylon_y)
    vsp.SetParmVal(nac_id,'Z_Rel_Location','XForm',z)        
    
    # The inside of the nacelle
    inside = vsp.AddSubSurf(nac_id, 1)
    vsp.SetParmVal(inside,"Const_Line_Type",inside,0.)
    vsp.SetParmVal(inside,"Const_Line_Value",inside,0.5)



    # Fuselage
    
    # Unpack the fuselage
    fuselage = vehicle.fuselages.fuselage
    width    = fuselage.width
    length   = fuselage.lengths.total
    hmax     = fuselage.heights.maximum
    height1  = fuselage.heights.at_quarter_length
    height2  = fuselage.heights.at_wing_root_quarter_chord 
    height3  = fuselage.heights.at_three_quarters_length
    effdia   = fuselage.effective_diameter
    n_fine   = fuselage.fineness.nose 
    t_fine   = fuselage.fineness.tail  
    
    w_origin = vehicle.wings.main_wing.origin
    
    # Figure out the location x location of each section, 3 sections, end of nose, wing origin, and start of tail
    
    x1 = n_fine*width/length
    x2 = w_origin[0]/length
    x3 = 1-t_fine*width/length
    
    fuse_id = vsp.AddGeom("FUSELAGE")    

    vsp.SetParmVal(fuse_id,"Length","Design",length)
    vsp.SetParmVal(fuse_id,"Diameter","Design",width)
    vsp.SetParmVal(fuse_id,"XLocPercent","XSec_1",x1)
    vsp.SetParmVal(fuse_id,"XLocPercent","XSec_2",x2)
    vsp.SetParmVal(fuse_id,"XLocPercent","XSec_3",x3)
    vsp.SetParmVal(fuse_id,"ZLocPercent","XSec_4",.02)
    vsp.SetParmVal(fuse_id, "Ellipse_Width", "XSecCurve_1", width)
    vsp.SetParmVal(fuse_id, "Ellipse_Width", "XSecCurve_2", width)
    vsp.SetParmVal(fuse_id, "Ellipse_Width", "XSecCurve_3", width)
    vsp.SetParmVal(fuse_id, "Ellipse_Height", "XSecCurve_1", height1);
    vsp.SetParmVal(fuse_id, "Ellipse_Height", "XSecCurve_2", height2);
    vsp.SetParmVal(fuse_id, "Ellipse_Height", "XSecCurve_3", height3);   
    
    
    # Write the vehicle to the file
    
    vsp.WriteVSPFile(vehicle.tag + ".vsp3")
    
    return