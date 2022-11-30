## @ingroup Plots-Performance-Aerodynamics
# plot_surface_pressure_contours.py
# 
# Created:    Nov 2022, E. Botero
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

import numpy as np

# ---------------------------------------------------------------------- 
#   Aerodynamic Forces
# ---------------------------------------------------------------------- 

## @ingroup Plots-Performance-Aerodynamics
def plot_surface_pressure_contours(results,vehicle):
    """This plots the surface pressure distrubtion at all control points
    on all lifting surfaces of the aircraft
    
    Assumptions:
    None
    
    Source:
    None
    
    Inputs:
    results.segments.aerodynamics.
        pressure_coefficient
    vehicle.vortex_distribution.
       n_cw
       n_sw
       n_w
    
    Outputs:
    Plots
    
    Properties Used:
    N/A
    """

    VD         = vehicle.vortex_distribution
    n_cw       = VD.n_cw
    n_cw       = VD.n_cw
    n_sw       = VD.n_sw
    n_w        = VD.n_w
    b_pts      = np.concatenate(([0],np.cumsum(VD.n_sw*VD.n_cw)))

    # Create a boolean for not plotting vertical wings
    idx        = 0
    plot_flag  = np.ones(n_w)
    for wing in vehicle.wings:
        if wing.vertical:
            plot_flag[idx] = 0
            idx += 1
        else:
            idx += 1
        if wing.vertical and wing.symmetric:
            plot_flag[idx] = 0
            idx += 1
        else:
            idx += 1

    img_idx    = 1
    seg_idx    = 1
    for segment in results.segments.values():
        num_ctrl_pts = len(segment.conditions.frames.inertial.time)
        for ti in range(num_ctrl_pts):
            CP         = segment.conditions.aerodynamics.pressure_coefficient[ti]

            fig        = plt.figure()
            axes       = plt.subplot(1, 1, 1)
            x_max      = max(VD.XC) + 2
            y_max      = max(VD.YC) + 2
            axes.set_ylim(x_max, 0)
            axes.set_xlim(-y_max, y_max)
            fig.set_size_inches(8,8)
            for i in range(n_w):
                n_pts     = (n_sw[i] + 1) * (n_cw[i]+ 1)
                xc_pts    = VD.X[i*(n_pts):(i+1)*(n_pts)]
                x_pts     = np.reshape(np.atleast_2d(VD.XC[b_pts[i]:b_pts[i+1]]).T, (n_sw[i],-1))
                y_pts     = np.reshape(np.atleast_2d(VD.YC[b_pts[i]:b_pts[i+1]]).T, (n_sw[i],-1))
                z_pts     = np.reshape(np.atleast_2d(CP[b_pts[i]:b_pts[i+1]]).T, (n_sw[i],-1))
                x_pts_p   = x_pts*((n_cw[i]+1)/n_cw[i]) - x_pts[0,0]*((n_cw[i]+1)/n_cw[i])  +  xc_pts[0]
                points    = np.linspace(0.001,1,50)
                A         = np.cumsum(np.sin(np.pi/2*points))
                levals    = -(np.concatenate([-A[::-1],A[1:]])/(2*A[-1])  + A[-1]/(2*A[-1]) )[::-1]*0.015
                color_map = plt.cm.get_cmap('jet')
                rev_cm    = color_map.reversed()
                if plot_flag[i] == 1:
                    CS  = axes.contourf(y_pts,x_pts_p, z_pts, cmap = rev_cm,levels=levals,extend='both')

            # Set Color bar
            cbar = fig.colorbar(CS, ax=axes)
            cbar.ax.set_ylabel('$C_{P}$', rotation =  0)
            plt.axis('off')
            plt.grid(None)

            if save_figure:
                plt.savefig( save_filename + '_' + str(img_idx) + file_type)
            img_idx += 1
        seg_idx +=1

    return
