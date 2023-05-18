## @ingroup Visualization-Performance-Energy-Battery
# plot_electric_motor_and_rotor_efficiencies.py
# 
# Created: Feb 2023 M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

from MARC.Core import Units
from MARC.Visualization.Performance.Common import set_axes, plot_style
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 

## @ingroup Visualization-Performance-Energy-Battery
def plot_electric_motor_and_rotor_efficiencies(results,
                                  save_figure = False,
                                  show_legend=True,
                                  save_filename = "eMotor_Prop_Efficiencies",
                                  file_type = ".png",
                                  width = 12, height = 7):
    """This plots the electric driven network propeller efficiencies 

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.conditions.propulsion. 
         etap
         etam
         fom
        
    Outputs: 
    Plots

    Properties Used:
    N/A	
    """	   
    # get plotting style 
    ps      = plot_style()  

    parameters = {'axes.labelsize': ps.axis_font_size,
                  'xtick.labelsize': ps.axis_font_size,
                  'ytick.labelsize': ps.axis_font_size,
                  'axes.titlesize': ps.title_font_size}
    plt.rcParams.update(parameters)
    
    # determine the number of propulsor groups 
    number_of_propulsor_groups = results.segments[0].conditions.propulsion.number_of_propulsor_groups 
    
    # get line colors for plots 
    line_colors   = cm.inferno(np.linspace(0,0.9,len(results.segments)))     
    
    for pg in range(number_of_propulsor_groups): 
        fig   = plt.figure(save_filename + '_Propulsor_Group_' + str(pg))
        fig.set_size_inches(width,height)
        
        for i in range(len(results.segments)): 
            time   = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
            effp   = results.segments[i].conditions.propulsion['propulsor_group_' + str(pg)].rotor.efficiency[:,0]
            fom    = results.segments[i].conditions.propulsion['propulsor_group_' + str(pg)].rotor.figure_of_merit[:,0]
            effm   = results.segments[i].conditions.propulsion['propulsor_group_' + str(pg)].motor.efficiency[:,0]        
                
            segment_tag  =  results.segments[i].tag
            segment_name = segment_tag.replace('_', ' ')
            axes_1 = plt.subplot(2,2,1)
            axes_1.plot(time, effp, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width, label = segment_name)
            axes_1.set_ylabel(r'$\eta_{rotor}$')
            set_axes(axes_1)         
            plt.ylim((0,1.1))
            
            axes_2 = plt.subplot(2,2,2)
            axes_2.plot(time, fom, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
            axes_2.set_xlabel('Time (mins)')
            axes_2.set_ylabel(r'FoM')
            set_axes(axes_2)
            plt.ylim((0,1.1)) 
    
            axes_3 = plt.subplot(2,2,3)
            axes_3.plot(time, effm, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
            axes_3.set_xlabel('Time (mins)')
            axes_3.set_ylabel(r'$\eta_{motor}$')
            set_axes(axes_3)
            plt.ylim((0,1.1)) 
       
        
        if show_legend:            
            leg =  fig.legend(bbox_to_anchor=(0.5, 0.95), loc='upper center', ncol = 5) 
            leg.set_title('Flight Segment', prop={'size': ps.legend_font_size, 'weight': 'heavy'})    
        
        # Adjusting the sub-plots for legend 
        fig.subplots_adjust(top=0.75)
        
        # set title of plot 
        title_text    = 'Propulsor Group ' + str(pg) + ': Rotor and Motor Conditions'      
        fig.suptitle(title_text)
        
        if save_figure:
            plt.savefig(save_filename + '_Propulsor_Group_' + str(pg)  + file_type)   
    return