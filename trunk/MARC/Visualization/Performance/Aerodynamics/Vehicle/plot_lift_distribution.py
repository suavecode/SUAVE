## @ingroup Visualization-Performance-Aerodynamics
# plot_lift_distribution.py
# 
# Created:    Dec 2022, M. Clarke
# Modified:   

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------  
from MARC.Visualization.Performance.Common import plot_style, save_plot 
import numpy as np 

import plotly.graph_objects as go
from plotly.subplots import make_subplots
# ---------------------------------------------------------------------- 
#   Sectional Lift Distribution
# ---------------------------------------------------------------------- 

## @ingroup Visualization-Performance-Aerodynamics
def plot_lift_distribution(results,vehicle,
                           save_figure=False,
                            save_filename="Lift_Distribution",
                            file_type=".png",
                            width = 1200, height = 600,
                            *args, **kwargs):   
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
   
   wing_names     = []
   symmetric_flag = []
   num_wings      = len(vehicle.wings)
   for wing in vehicle.wings:
      wing_names.append(wing.tag) 
      if wing.symmetric:
         symmetric_flag.append(1)  
      else:
         symmetric_flag.append(0)
      
   for segment in results.segments.values():
      num_ctrl_pts = len(segment.conditions.frames.inertial.time)
      for ti in range(num_ctrl_pts): 
         fig       = make_subplots(rows=1, cols=1) 
         cl_y      = segment.conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional[ti]
         time_step = segment.conditions.frames.inertial.time[ti]  
         start     = 0 
         end       = 1 
         for i in range(num_wings): 
            cl_vals  = cl_y[b_sw[start]:b_sw[end]]
            y_pts    = VD.Y_SW[b_sw[start]:b_sw[end]]             
            if symmetric_flag[i]: 
               start += 1     
               end   += 1 
               cl_vals  = np.hstack(( cl_y[b_sw[start]:b_sw[end]][::-1],cl_vals ))
               y_pts    = np.hstack(( VD.Y_SW[b_sw[start]:b_sw[end]][::-1], y_pts)) 
            
            fig.add_trace(go.Scatter(x=y_pts,
                                    y=cl_vals  ,
                                    name=wing_names[i]),
                                    row=1, col=1)     
 
            start += 1     
            end   += 1             
            
         fig.update_yaxes(title_text='$C_{Ly}$', row=1, col=1)  
         fig.update_xaxes(title_text='Spanwise Location (m)', row=1, col=1) 
      
         fig = plot_style(fig)    
         fig.update_xaxes(rangemode="normal") # update plot so that negative axis is plotted 
         
         # Set overall figure layout style and legend title
         fig.update_layout(
             width=width, height=height,
              legend_title_text='Wing',
              title_text = 'Segment: '+ segment.tag + 'Times: ' + str(time_step) 
          )
         
         if save_figure:
            save_plot(fig, save_filename, file_type)     
      
         fig.show() 

   return
