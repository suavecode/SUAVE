## @ingroup Visualization-Performance-Energy-Battery
# plot_lift_cruise_network.py
# 
# Created:    Nov 2022, J. Smart
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

from SUAVE.Core import Units
from SUAVE.Visualization.Performance.Common import plot_style, save_plot
from SUAVE.Visualization.Performance.Energy.Battery import *
from SUAVE.Visualization.Performance.Energy.Common import *

import numpy as np
import pandas as pd

import plotly.graph_objects as go

from plotly.subplots import make_subplots

## @ingroup Visualization-Performance-Energy-Battery
def plot_lift_cruise_network(results,
                             save_figure=False,
                             battery_save_filename="Battery_Pack_Conditions",
                             propeller_save_filename="Propeller_Conditions",
                             lift_rotor_save_filename="Lift_Rotor_Conditions",
                             mach_save_filename="Tip_Mach_Numbers",
                             file_type=".png",
                             width = 1200, height = 600,
                             *args, **kwargs):
    """Plots the electric and propulsion network performance of a vehicle
    with a lift-cruise network.

    Assumptions:
    None

    Deprecated SUAVE Mission Plots Functions

    Created:    Mar 2020, M. Clarke
    Modified:   Apr 2020, M. Clarke
                Sep 2020, M. Clarke
                Apr 2021, M. Clarke
                Dec 2021, S. Claridge

    Inputs:
    results.segments.conditions.propulsion
         throttle
         lift_rotor_throttle
         battery_energy
         battery_specific_power
         voltage_under_load
         voltage_open_circuit

    Outputs: 
    Plots

    Properties Used:
    N/A	
    """

    # Create empty dataframe to be populated with the segment data

    plot_cols = [
        'Throttle',
        'Lift Throttle',
        'Battery Energy',
        'Battery Specific Power',
        'Voltage Under Load',
        'Voltage Open Circuit',
        'Prop RPM',
        'Prop Thrust',
        'Prop Torque',
        'Prop Efficiency',
        'Prop Motor Efficiency',
        'Prop Power Coefficient',
        'Prop Tip Mach',
        'Lift RPM',
        'Lift Thrust',
        'Lift Torque',
        'Lift FoM',
        'Lift Motor Efficiency',
        'Lift Power Coefficient',
        'Lift Tip Mach',
        'Segment',
    ]

    df = pd.DataFrame(columns=plot_cols)

    # Get the segment-by-segment results

    for segment in results.segments.values():

        time                = segment.conditions.frames.inertial.time[:,0] / Units.min
        eta                 = segment.conditions.propulsion.throttle[:,0]
        eta_l               = segment.conditions.propulsion.throttle_lift[:,0]
        
        energy              = segment.conditions.propulsion.battery.pack.energy[:,0]/ Units.Wh
        specific_power      = segment.conditions.propulsion.battery.pack.specfic_power[:,0]
        volts               = segment.conditions.propulsion.battery.pack.voltage_under_load[:,0]
        volts_oc            = segment.conditions.propulsion.battery.pack.voltage_open_circuit[:,0]
        
        prop_thrust         = segment.conditions.frames.body.thrust_force_vector[:,0]
        lift_rotor_thrust   =-segment.conditions.frames.body.thrust_force_vector[:,2]
        
        prop_torque         = segment.conditions.propulsion.propeller_motor.torque[:,0]
        prop_effm           = segment.conditions.propulsion.propeller_motor.efficiency[:,0]
        
        prop_rpm            = segment.conditions.propulsion.propeller.rpm[:,0]
        prop_effp           = segment.conditions.propulsion.propeller.efficiency[:,0]
        prop_Cp             = segment.conditions.propulsion.propeller.power_coefficient[:,0]
        ptm                 = segment.conditions.propulsion.propeller.tip_mach[:, 0]
        
        lift_rotor_rpm      = segment.conditions.propulsion.lift_rotor.rpm[:,0]
        lift_rotor_effp     = segment.conditions.propulsion.lift_rotor.efficiency[:,0]
        lift_rotor_effm     = segment.conditions.propulsion.lift_rotor.figure_of_merit[:,0]
        lift_rotor_Cp       = segment.conditions.propulsion.lift_rotor.power_coefficient[:,0]
        rtm                 = segment.conditions.propulsion.lift_rotor.tip_mach[:, 0]
        
        lift_rotor_torque   = segment.conditions.propulsion.lift_rotor_motor.torque[:,0]

        # Assemble the data into temporary holding dataframe

        segment_frame = pd.DataFrame(
            np.column_stack((
                eta               ,
                eta_l             ,
                energy            ,
                specific_power    ,
                volts             ,
                volts_oc          ,
                prop_rpm          ,
                prop_thrust       ,
                prop_torque       ,
                prop_effp         ,
                prop_effm         ,
                prop_Cp           ,
                ptm               ,
                lift_rotor_rpm    ,
                lift_rotor_thrust ,
                lift_rotor_torque ,
                lift_rotor_effp   ,
                lift_rotor_effm   ,
                lift_rotor_Cp     ,
                rtm
            )),
            columns=plot_cols[:-1], index=time
        )

        segment_frame['Segment'] = [segment.tag for i in range(len(time))]

        # Append to collecting dataframe

        df = df.append(segment_frame)

        # Set plot parameters

        batt_fig = make_subplots(rows=3, cols=2)
        prop_fig = make_subplots(rows=3, cols=2)
        lift_fig = make_subplots(rows=3, cols=2)
        mach_fig = make_subplots(rows=2, cols=1)

        for seg, data in df.groupby("Segment", sort=False):

            seg_name = ' '.join(seg.split("_")).capitalize()

            # Battery Traces

            batt_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Throttle'],
                name=seg_name),
                row=1, col=1)

            batt_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Lift Throttle'],
                name=seg_name,
                showlegend=False),
                row=1, col=2)

            batt_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Battery Energy'],
                name=seg_name,
                showlegend=False),
                row=2, col=1)

            batt_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Battery Specific Power'],
                name=seg_name,
                showlegend=False),
                row=2, col=2)

            batt_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Voltage Under Load'],
                name=seg_name,
                showlegend=False),
                row=3, col=1)

            batt_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Voltage Open Circuit'],
                name=seg_name,
                showlegend=False),
                row=3, col=2)

            # Propeller Traces

            prop_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Prop RPM'],
                name=seg_name),
                row=1, col=1)

            prop_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Prop Motor Efficiency'],
                name=seg_name,
                showlegend=False),
                row=1, col=2)

            prop_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Prop Thrust'],
                name=seg_name,
                showlegend=False),
                row=2, col=1)

            prop_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Prop Torque'],
                name=seg_name,
                showlegend=False),
                row=2, col=2)

            prop_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Prop Efficiency'],
                name=seg_name,
                showlegend=False),
                row=3, col=1)

            prop_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Prop Power Coefficient'],
                name=seg_name,
                showlegend=False),
                row=3, col=2)

            # Lift Traces 
            lift_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Lift RPM'],
                name=seg_name),
                row=1, col=1)

            lift_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Lift Motor Efficiency'],
                name=seg_name,
                showlegend=False),
                row=1, col=2)

            lift_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Lift Thrust'],
                name=seg_name,
                showlegend=False),
                row=2, col=1)

            lift_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Lift Torque'],
                name=seg_name,
                showlegend=False),
                row=2, col=2)

            lift_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Lift FoM'],
                name=seg_name,
                showlegend=False),
                row=3, col=1)

            lift_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Lift Power Coefficient'],
                name=seg_name,
                showlegend=False),
                row=3, col=2)

            # Mach Traces

            mach_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Prop Tip Mach'],
                name=seg_name),
                row=1, col=1)

            mach_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Lift Tip Mach'],
                name=seg_name,
                showlegend=False),
                row=2, col=1)


        # Add Titles

        # Battery Axes

        batt_fig.update_yaxes(title_text='Prop. Throttle', row=1, col=1)
        batt_fig.update_yaxes(title_text='Lift Throttle', row=1, col=2)
        batt_fig.update_yaxes(title_text='Energy (W-hr)', row=2, col=1)
        batt_fig.update_yaxes(title_text='Specific Power', row=2, col=2)
        batt_fig.update_yaxes(title_text='Voltage (V)', row=3, col=1)
        batt_fig.update_yaxes(title_text='Voltage OC (V)', row=3, col=1)

        batt_fig.update_xaxes(title_text='Time (min)', row=3, col=1)
        batt_fig.update_xaxes(title_text='Time (min)', row=3, col=2)  

        batt_fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment',
        title_text = 'Battery Pack Conditions',
        )

        

        # Prop Axes

        prop_fig.update_yaxes(title_text='RPM', row=1, col=1)
        prop_fig.update_yaxes(title_text='Motor Efficiency', row=1, col=2)
        prop_fig.update_yaxes(title_text='Thrust (N)', row=2, col=1)
        prop_fig.update_yaxes(title_text='Torque (N-m)', row=2, col=2)
        prop_fig.update_yaxes(title_text='Efficiency', row=3, col=1)
        prop_fig.update_yaxes(title_text='Power Coefficient', row=3, col=2)

        prop_fig.update_xaxes(title_text='Time (min)', row=3, col=1)
        prop_fig.update_xaxes(title_text='Time (min)', row=3, col=2) 


        prop_fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment',
        title_text = 'Propeller Conditions',
        )
 
        # Lift Axes

        lift_fig.update_yaxes(title_text='RPM', row=1, col=1)
        lift_fig.update_yaxes(title_text='Motor Efficiency', row=1, col=2)
        lift_fig.update_yaxes(title_text='Thrust (N)', row=2, col=1)
        lift_fig.update_yaxes(title_text='Torque (N-m)', row=2, col=2)
        lift_fig.update_yaxes(title_text='Efficiency', row=3, col=1)
        lift_fig.update_yaxes(title_text='Power Coefficient', row=3, col=2)

        lift_fig.update_xaxes(title_text='Time (min)', row=3, col=1)
        lift_fig.update_xaxes(title_text='Time (min)', row=3, col=2) 


        lift_fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment',
        title_text = 'Lift Rotor Conditions',
        )
 
        # Tip Mach Axes

        mach_fig.update_yaxes(title_text='Propeller', row=1, col=1)
        mach_fig.update_yaxes(title_text='Lift Rotor', row=2, col=1)

        mach_fig.update_xaxes(title_text='Time (min)', row=2, col=1) 

        mach_fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment',
        title_text = 'Tip Mach Numbers',
        )

        # Set Style

        batt_fig = plot_style(batt_fig)
        batt_fig.show()

        prop_fig = plot_style(prop_fig)
        prop_fig.show()

        lift_fig = plot_style(lift_fig)
        lift_fig.show()

        mach_fig = plot_style(mach_fig)
        mach_fig.show()

        if save_figure:
            save_plot(batt_fig, battery_save_filename, file_type)
            save_plot(prop_fig, propeller_save_filename, file_type)
            save_plot(lift_fig, lift_rotor_save_filename, file_type)
            save_plot(mach_fig, mach_save_filename, file_type)
         
        return
