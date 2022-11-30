## @ingroup [ADD DOCUMENTATION GROUP]
# plot_eMotor_Prop_efficiencies.py
# 
# Created:    Nov 2022, J. Smart
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

# TODO: ADD IMPORTS

## @ingroup [ADD DOCUMENTATION GROUP]
def plot_eMotor_Prop_efficiencies(results,
                                  save_figure = False,
                                  save_filename = "eMotor_Prop_Efficiencies",
                                  file_type = ".png",
                                  width = 800, height = 500,
                                  *args, **kwargs):
    """This plots the electric driven network propeller efficiencies

    Assumptions:
    None

    Source:

    Deprecated SUAVE Mission Plots Functions

    Created:    Mar 2020, M. Clarke
    Modified:   Apr 2020, M. Clarke
                Sep 2020, M. Clarke
                Apr 2021, M. Clarke
                Dec 2021, S. Claridge

    Inputs:
    results.segments.conditions.propulsion.
         etap
         etam

    Outputs:
    Plots

    Properties Used:
    N/A
    """
    # Create empty dataframe to be populated by the segment data

    plot_cols = [
        "Propeller Efficiency",
        "Figure of Merit",
        "Motor Efficiency",
        "Segment"
    ]

    df = pd.DataFrame(columns=plot_cols)

    # Get the segment-by-segment results

    for segment in results.segments.values():

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        effp   = segment.conditions.propulsion.propeller_efficiency[:,0]
        fom    = segment.conditions.propulsion.figure_of_merit[:,0]
        effm   = segment.conditions.propulsion.propeller_motor_efficiency[:,0]

        segment_frame = pd.DataFrame(
            np.column_stack((
                effp,
                fom,
                effm,

            )),
            columns=plot_cols[:-1], index=time
        )

        segment_frame['Segment'] = [segment.tag for i in range(len(time))]

        # Append to collecting dataframe

        df = df.append(segment_frame)

    # Set plot properties

    fig = make_subplots(rows=3, cols=1)

    # Add traces to the figure for each value by segment

    for seg, data in df.groupby("Segment", sort=False):

        seg_name = ' '.join(seg.split("_")).capitalize()

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Propeller Efficiency'],
            name=seg_name),
            row=1, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Figure of Merit'],
            name=seg_name,
            showlegend=False),
            row=2, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Motor Efficiency'],
            name=seg_name,
            showlegend=False),
            row=3, col=1)

    fig.update_yaxes(title_text=r'Propeller Efficiency ($\eta_p$)', row=1, col=1)
    fig.update_yaxes(title_text='Figure of Merit', row=2, col=1)
    fig.update_yaxes(title_text=r'Motor Efficiency ($\eta_m$)', row=3, col=1)

    fig.update_xaxes(title_text='Time (min)', row=3, col=1)

    fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment'
    )

    fig = plot_style(fig)
    fig.show()

    if save_figure:
        save_plot(fig, save_filename, file_type)

    return
