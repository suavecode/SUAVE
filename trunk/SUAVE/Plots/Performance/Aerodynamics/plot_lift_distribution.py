## @ingroup Plots-Performance-Aerodynamics
# plot_lift_distribution.py
# 
# Created:    Nov 2022, E. Botero
# Modified:   

# ----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------- 

import numpy as np

# ---------------------------------------------------------------------- 
#   Sectional Lift Distribution
# ---------------------------------------------------------------------- 

## @ingroup Plots-Performance-Aerodynamics
def plot_lift_distribution(results,vehicle):
   """This plots the sectional lift distrubtion at all control points
    on all lifting surfaces of the aircraft
    
    Assumptions:
    None
    
    Source:
    None
    
    Inputs:
    results.segments.aerodynamics.
        inviscid_wings_sectional_lift
    vehicle.vortex_distribution.
       n_sw
       n_w
       
    Outputs:
    Plots
    
    Properties Used:
    N/A
    """
   VD         = vehicle.vortex_distribution
   n_w        = VD.n_w
   b_sw       = np.concatenate(([0],np.cumsum(VD.n_sw)))

   axis_font  = {'size':'12'}
   img_idx    = 1
   seg_idx    = 1
   for segment in results.segments.values():
      num_ctrl_pts = len(segment.conditions.frames.inertial.time)
      for ti in range(num_ctrl_pts):
         cl_y = segment.conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional[ti]
         line = ['-b','-b','-r','-r','-k']
         fig  = plt.figure()
         fig.set_size_inches(8,8)
         axes = plt.subplot(1,1,1)
         for i in range(n_w):
            y_pts = VD.Y_SW[b_sw[i]:b_sw[i+1]]
            z_pts = cl_y[b_sw[i]:b_sw[i+1]]
            axes.plot(y_pts, z_pts, line[i] )
         axes.set_xlabel("Spanwise Location (m)",axis_font)
         axes.set_title('$C_{Ly}$',axis_font)

         if save_figure:
            plt.savefig( save_filename + '_' + str(img_idx) + file_type)
         img_idx += 1
      seg_idx +=1


   return
