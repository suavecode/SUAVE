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

## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
def compute_vortex_wake_miss_distances(wake,rotor, h=0.1):  
    """
    This uses a planar geometric method to compute the miss-distances of the rotor wake system
    in the vertical plane of each blade.
    
    """
    # Extract wake parameters and vortex distribution
    Na_high_res = wake.wake_settings.high_resolution_azimuthals
    
    VD_low_res = wake.vortex_distribution
    VD_lr_shape = np.shape(VD_low_res.reshaped_wake.XA1)
    Na_low_res = VD_lr_shape[0]
    m = VD_lr_shape[1]
    B = VD_lr_shape[2]
    nmax = VD_lr_shape[3]
    nts = VD_lr_shape[4]
    Nr = nmax + 1
    
    VD = wake.vortex_distribution.reshaped_wake
    tol = 1e-3
    
    # 
    psi = np.linspace(0,2*np.pi,Na_high_res+1)[:-1]
    Rh = rotor.hub_radius
    R = rotor.tip_radius
    
    missDistances = np.ones((Na_high_res, m, B, Nr-1, nts-1)) * np.nan #np.ones_like(VD.XA1) * np.nan
    planeIntersectionPoints = np.ones((3, Na_high_res, m, B, Nr-1, nts-1)) * np.nan #np.ones_like(np.repeat(VD.XA1[None,:,:,:,:], 3, axis=0)) * np.nan
    for a in range(Na_high_res):
        # neglect lifting line panel rows
        XA1 = VD.XA1[a,:,:,:,1:]
        XA2 = VD.XA2[a,:,:,:,1:]
        XB1 = VD.XB1[a,:,:,:,1:]
        XB2 = VD.XB2[a,:,:,:,1:]
        YA1 = VD.YA1[a,:,:,:,1:]
        YA2 = VD.YA2[a,:,:,:,1:]
        YB1 = VD.YB1[a,:,:,:,1:]
        YB2 = VD.YB2[a,:,:,:,1:]      
        ZA1 = VD.ZA1[a,:,:,:,1:]
        ZA2 = VD.ZA2[a,:,:,:,1:]
        ZB1 = VD.ZB1[a,:,:,:,1:]
        ZB2 = VD.ZB2[a,:,:,:,1:] 
        
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
        
        rotor_line = np.matmul(rot_mat, ((Xb[1:] + Xb[:-1])/2, (Yb[1:] + Yb[:-1])/2,  (Zb[1:] + Zb[:-1])/2))
        rotor_line = np.tile(rotor_line[:,None,None,:,None], (1, m, B, 1, nts-1))
        
        n_hat = np.cross((pB-pA), (pC-pB))
        d = -np.dot(n_hat, pA)
        
        # compute line equations
        PA1 = np.reshape(np.concatenate([XA1[None,:,:,:,:], YA1[None,:,:,:,:], ZA1[None,:,:,:,:]]), (3, np.size(XA1)))
        PA2 = np.reshape(np.concatenate([XA2[None,:,:,:,:], YA2[None,:,:,:,:], ZA2[None,:,:,:,:]]), (3, np.size(XA1)))
        PB1 = np.reshape(np.concatenate([XB1[None,:,:,:,:], YB1[None,:,:,:,:], ZB1[None,:,:,:,:]]), (3, np.size(XA1)))
        PB2 = np.reshape(np.concatenate([XB2[None,:,:,:,:], YB2[None,:,:,:,:], ZB2[None,:,:,:,:]]), (3, np.size(XA1)))
        
        # solve for t (inf if no intersection, as denominator will go to zero)
        tA = (np.matmul(n_hat, PA1) - d) / -(np.matmul(n_hat, (PA2-PA1)))
        tB = (np.matmul(n_hat, PB1) - d) / -(np.matmul(n_hat, (PB2-PB1)))
        
        # find intersection points with plane
        xA_I = PA1 + (PA2 - PA1) * tA
        xB_I = PB1 + (PB2 - PB1) * tB
        
        # check if intersection point lies between point 1 and point 2 (not just along the line created by these points)
        dLA = np.linalg.norm(PA1 - PA2, axis=0)
        dLA_sum = np.linalg.norm(PA1 - xA_I, axis=0) + np.linalg.norm(PA2 - xA_I, axis=0)
        inLineA = np.isclose(dLA, dLA_sum)
        
        dLB = np.linalg.norm(PB1 - PB2, axis=0)
        dLB_sum = np.linalg.norm(PB1 - xB_I, axis=0) + np.linalg.norm(PB2 - xB_I, axis=0)
        inLineB = np.isclose(dLB, dLB_sum)      
        
        # ------------------------------------------------------------------------------------------------
        # remove intersection points that are outside the blade projection plane
        # ------------------------------------------------------------------------------------------------
        # A points
        d_IA = np.linalg.norm((xA_I.T - pA).T, axis=0)
        d_IB = np.linalg.norm((xA_I.T - pB).T, axis=0)
        d_IC = np.linalg.norm((xA_I.T - pC).T, axis=0)
        d_ID = np.linalg.norm((xA_I.T - pD).T, axis=0)
        
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
        
        inLineA[~in_rect] = False
        inLineA = np.reshape(inLineA, np.shape(XA1))
        
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
        xA_I = np.reshape(xA_I, np.shape(np.repeat(XA1[None,:,:,:,:],3,axis=0)))
        d_A_I = np.linalg.norm(xA_I - rotor_line, axis=0) # fix this: distance not to corresponding radial station but shortest distance to rotor disc plane

        xB_I = np.reshape(xB_I, np.shape(np.repeat(XA1[None,:,:,:,:],3,axis=0)))
        d_B_I = np.linalg.norm(xB_I - rotor_line, axis=0)        
        
        missDistances[a][inLineA] = d_A_I[inLineA]
        planeIntersectionPoints[0][a][inLineA] = xA_I[0][inLineA]
        planeIntersectionPoints[1][a][inLineA] = xA_I[1][inLineA]
        planeIntersectionPoints[2][a][inLineA] = xA_I[2][inLineA]

        missDistances[a][inLineB] = d_B_I[inLineB]
        planeIntersectionPoints[0][a][inLineB] = xB_I[0][inLineB]  
        planeIntersectionPoints[1][a][inLineB] = xB_I[1][inLineB]  
        planeIntersectionPoints[2][a][inLineB] = xB_I[2][inLineB]        
    
        # record points with miss distance values and output the VTK for each time step
        saveDir = "/Users/rerha/Desktop/missDistanceDebug/"
        for b in range(B):
            points = Data()
            points.XC = np.ravel(planeIntersectionPoints[0,a,0,b,:,:])
            points.YC = np.ravel(planeIntersectionPoints[1,a,0,b,:,:])
            points.ZC = np.ravel(planeIntersectionPoints[2,a,0,b,:,:])
            
            point_data = Data()
            point_data.radialOrigin = np.ravel(np.tile(np.arange(Nr-1)[:,None], (1,nts)))
            point_data.bladeOrigin = np.ones_like(point_data.radialOrigin) * b
            point_data.missDistance = np.ravel(missDistances[a,0,b,:,:])
            points.point_data = point_data
            
            from SUAVE.Input_Output.VTK.save_evaluation_points_vtk import save_evaluation_points_vtk
            save_evaluation_points_vtk(points, filename=saveDir+"bvi_blade_{}.vtk".format(b),time_step=a)
            
        
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
        
        
    return
  