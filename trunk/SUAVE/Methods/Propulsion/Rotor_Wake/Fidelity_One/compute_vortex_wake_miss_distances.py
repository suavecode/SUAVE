## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
# compute_vortex_wake_miss_distances.py
# 
# Created:  Jan 2023, R. Erhard 
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
from SUAVE.Core import Data
import numpy as np 
from DCode.Common.plottingFunctions import *
from DCode.Common.generalFunctions import *
from DCode.Common.Visualization_Tools.plane_contour_field_vtk import plane_contour_field_vtk
from SUAVE.Input_Output.VTK.save_evaluation_points_vtk import save_evaluation_points_vtk

## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
def compute_vortex_wake_miss_distances(wake,rotor, h_c=0.6):  
    """
    This uses a planar geometric method to compute the miss-distances of the rotor wake system
    in the vertical plane of each blade.
    
    Inputs
       clip_tip - flag to clip the interactions only for the tip vortices [Boolean]
    
    """
    # Extract wake parameters and vortex distribution
    Na_high_res = wake.wake_settings.high_resolution_azimuthals
    
    VD_low_res = wake.vortex_distribution
    VD_lr_shape = np.shape(VD_low_res.reshaped_wake.XA1)
    m = VD_lr_shape[1]
    B = VD_lr_shape[2]
    nmax = VD_lr_shape[3]
    nts = VD_lr_shape[4]
    Nr = nmax + 1
    
    VD = wake.vortex_distribution.reshaped_wake
    h = h_c * np.max(rotor.chord_distribution)
    
    # 
    psi = np.linspace(0,2*np.pi,Na_high_res+1)[:-1]
    Rh = rotor.hub_radius
    R = rotor.tip_radius
    
    tipMissDistances = np.ones((Na_high_res, m, B, nts-1)) * np.nan
    tipPlaneIntersectionPoints = np.ones((3, Na_high_res, m, B, nts-1)) * np.nan
    bvi_radial_locations = np.ones((Na_high_res, m, B, nts-1)) * np.nan
    
    for a in range(Na_high_res):
        # neglect lifting line panel rows, and clip_tip/ only consider the Nr'th vortex panels
        XB1 = VD.XB1[a,:,:,-1,1:]
        XB2 = VD.XB2[a,:,:,-1,1:]
        YB1 = VD.YB1[a,:,:,-1,1:]
        YB2 = VD.YB2[a,:,:,-1,1:]      
        ZB1 = VD.ZB1[a,:,:,-1,1:]
        ZB2 = VD.ZB2[a,:,:,-1,1:] 
        
        # compute distance of all filaments to rotor blade
        x_min_rotor_frame = -h
        x_max_rotor_frame = h
        y_min_rotor_frame = Rh * np.sin(psi[a])
        y_max_rotor_frame = R  * np.sin(psi[a])
        z_min_rotor_frame = Rh * np.cos(psi[a])
        z_max_rotor_frame = R  * np.cos(psi[a])
        
        # compute rotor plane equation vertices, 
        pA = np.array([x_min_rotor_frame, y_min_rotor_frame, z_min_rotor_frame])
        pB = np.array([x_min_rotor_frame, y_max_rotor_frame, z_max_rotor_frame ])
        pC = np.array([x_max_rotor_frame, y_max_rotor_frame, z_max_rotor_frame ])
        pD = np.array([x_max_rotor_frame, y_min_rotor_frame, z_min_rotor_frame ])
        
        # convert points from rotor frame into vehicle reference frame
        rot_mat = rotor.vehicle_body_to_prop_vel()[0]
        pA = np.matmul(rot_mat, pA)
        pB = np.matmul(rot_mat, pB)
        pC = np.matmul(rot_mat, pC)
        pD = np.matmul(rot_mat, pD)
        
        # get rotor blade zero line
        Yb   = wake.vortex_distribution.reshaped_wake.Yblades_cp[a,0,0,:,0]
        Zb   = wake.vortex_distribution.reshaped_wake.Zblades_cp[a,0,0,:,0]
        Xb   = wake.vortex_distribution.reshaped_wake.Xblades_cp[a,0,0,:,0]        
        
        rotor_line = np.concatenate([Xb[None,:], Yb[None,:],  Zb[None,:]]) #0.5*np.concatenate([(Xb[1:] + Xb[:-1])[None,:], (Yb[1:] + Yb[:-1])[None,:],  (Zb[1:] + Zb[:-1])[None,:]]) 
        
        n_hat = np.cross((pB-pA), (pC-pB))
        d = -np.dot(n_hat, pA)
        
        # compute line equations
        PB1 = np.reshape(np.concatenate([XB1[None,:,:,:], YB1[None,:,:,:], ZB1[None,:,:,:]]), (3, np.size(XB1)))
        PB2 = np.reshape(np.concatenate([XB2[None,:,:,:], YB2[None,:,:,:], ZB2[None,:,:,:]]), (3, np.size(XB1)))
    
        # solve for t (inf if no intersection, as denominator will go to zero)
        tB = (np.matmul(n_hat, PB1) - d) / -(np.matmul(n_hat, (PB2-PB1)))
        
        # find intersection points with plane
        xB_I = PB1 + (PB2 - PB1) * tB
        
        # check if intersection point lies between point 1 and point 2 (not just along the line created by these points)
        dLB = np.linalg.norm(PB1 - PB2, axis=0)
        dLB_sum = np.linalg.norm(PB1 - xB_I, axis=0) + np.linalg.norm(PB2 - xB_I, axis=0)
        inLineB = np.isclose(dLB, dLB_sum)      
        
        # ------------------------------------------------------------------------------------------------
        # remove intersection points that are outside the blade projection plane
        # ------------------------------------------------------------------------------------------------
        
        # B points
        d_IA = np.linalg.norm((xB_I.T - pA).T, axis=0)
        d_IB = np.linalg.norm((xB_I.T - pB).T, axis=0)
        d_IC = np.linalg.norm((xB_I.T - pC).T, axis=0)
        d_ID = np.linalg.norm((xB_I.T - pD).T, axis=0)
        
        d_AD = np.linalg.norm(pA-pD)
        d_AB = np.linalg.norm(pA-pB)
        d_CD = np.linalg.norm(pC-pD)
        d_BC = np.linalg.norm(pB-pC)
        
        s_ADI = 0.5 * (d_AD + d_IA + d_ID)
        s_ABI = 0.5 * (d_AB + d_IA + d_IB)
        s_CDI = 0.5 * (d_CD + d_IC + d_ID)
        s_BCI = 0.5 * (d_BC + d_IB + d_IC)
        area_ADI = np.sqrt( s_ADI * (s_ADI - d_AD) * (s_ADI - d_IA) * (s_ADI - d_ID) )
        area_ABI = np.sqrt( s_ABI * (s_ABI - d_AB) * (s_ABI - d_IA) * (s_ABI - d_IB) )
        area_CDI = np.sqrt( s_CDI * (s_CDI - d_CD) * (s_CDI - d_IC) * (s_CDI - d_ID) )
        area_BCI = np.sqrt( s_BCI * (s_BCI - d_BC) * (s_BCI - d_IB) * (s_BCI - d_IC) )
        point_rect_area = area_ADI + area_ABI + area_CDI + area_BCI
        rect_area = d_AD * d_AB
        in_rect = np.isclose(point_rect_area, rect_area)
        
        inLineB[~in_rect] = False
        inLineB = np.reshape(inLineB, np.shape(XB1))          
        
        
        # record distances from each intersecting element
        # find position of closest blade element to each xA_I
        # distance of each xA_I element to each radial station
        
        dX = (xB_I[0].reshape(1,-1) - rotor_line[0].reshape(-1,1))
        dY = (xB_I[1].reshape(1,-1) - rotor_line[1].reshape(-1,1))
        dZ = (xB_I[2].reshape(1,-1) - rotor_line[2].reshape(-1,1))
        distances = np.sqrt(dX**2 + dY**2 + dZ**2)
        indices = np.abs(distances).argmin(axis=0)
        #residual = np.diagonal(distances[indices,])
        
        blade_element_position_B = rotor_line[:,indices]        
        
        d_B_I = np.reshape(np.linalg.norm(xB_I - blade_element_position_B, axis=0), np.shape(XB1))# fix this: distance not to corresponding radial station but shortest distance to rotor disc plane
        xB_I = np.reshape(xB_I, np.shape(np.repeat(XB1[None,:,:,:],3,axis=0)))   
        blade_element_position_B = np.reshape(blade_element_position_B, np.shape(np.repeat(XB1[None,:,:,:],3,axis=0)))
        
        tipMissDistances[a][inLineB] = d_B_I[inLineB]
        tipPlaneIntersectionPoints[0][a][inLineB] = xB_I[0][inLineB]  
        tipPlaneIntersectionPoints[1][a][inLineB] = xB_I[1][inLineB]  
        tipPlaneIntersectionPoints[2][a][inLineB] = xB_I[2][inLineB]     

        # record closest radial disc location of BVI
        bvi_event = np.argwhere(~np.isnan(tipMissDistances[a]) )
        radial_stations_bvi = np.linalg.norm(blade_element_position_B, axis=0)
        bvi_radial_locations[a][inLineB] = radial_stations_bvi[tuple(bvi_event.T)]
    
        # record points with miss distance values and output the VTK for each time step
        saveDir = "/Users/rerha/Desktop/missDistanceDebug/"
        for b in range(B):
            points = Data()
            points.XC = np.ravel(tipPlaneIntersectionPoints[0,a,0,b,:])
            points.YC = np.ravel(tipPlaneIntersectionPoints[1,a,0,b,:])
            points.ZC = np.ravel(tipPlaneIntersectionPoints[2,a,0,b,:])
            
            point_data = Data()
            point_data.bladeOrigin = np.ones_like(points.XC) * b
            point_data.missDistance = np.ravel(tipMissDistances[a,0,b,:])
            points.point_data = point_data
            
            save_evaluation_points_vtk(points, filename=saveDir+"bvi_blade_{}.vtk".format(b),time_step=a)
        
        # DEBUG: plot vtk of rotor line
        pts = Data()
        pts.XC = rotor_line[0,:]
        pts.YC = rotor_line[1,:]
        pts.ZC = rotor_line[2,:]
        pts.point_data = Data()
        save_evaluation_points_vtk(pts, filename=saveDir+"rotor_line.vtk",time_step=a)
        
            
        
        points=Data()
        points.XC = np.array([pA[0], pB[0], pD[0], pC[0]])
        points.YC = np.array([pA[1], pB[1], pD[1], pC[1]])
        points.ZC = np.array([pA[2], pB[2], pD[2], pC[2]])
        save_evaluation_points_vtk(points, filename=saveDir+"bvi_plane.vtk",time_step=a)
        
        planeData = Data()
        planeData.Position = np.array([points.XC, points.YC, points.ZC]).T
        #planeData.Velocity = np.zeros_like(planeData.Position)
        planeData.Nw = 2
        planeData.Nh = 2
        stateData = Data()
        stateData.vFreestream = 20
        stateData.alphaDeg = 0
        plane_contour_field_vtk(planeData, stateData, filename=saveDir + "bvi_contour_plane.vtk", iteration=a)
        
    # debug: polar plot of disk contour bvi events
    theta = 2*np.pi*np.linspace(0, 1, Na_high_res+1)[:-1]
    cols = ['black','red','blue','green']
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    for a in range(Na_high_res):
        for b in range(B):
            # plot bvis that originate from blade b
            r_locs = bvi_radial_locations[a,0,b,:][~np.isnan(bvi_radial_locations[a,0,b,:])]
            axis0.plot(np.ones_like(r_locs)*theta[a], r_locs, 'x',color=cols[b])    # -np.pi+psi turns it 

    axis0.set_title("BVI Occurrence") 
    axis0.set_rorigin(0)
    axis0.set_theta_zero_location("S")
    axis0.set_yticklabels([])
    plt.show()
    return
  