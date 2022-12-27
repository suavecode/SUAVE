## @ingroup Visualization-Performance-Common
# plot_style.py
# 
# Created:    Nov 2022, J. Smart
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

from itertools import cycle
import plotly.express as px
import plotly

## @ingroup Visualization-Performance-Common
def plot_style(fig, *args, **kwargs):
    """Helper function for automatically setting the style of plots to the
    SUAVE standard style.

    Use immediately before showing the figure to ensure all necessary
    information is available and to avoid over-writing style when
    constructing the figure.

    Usage:  fig = plot_style(fig)
            fig.show()

    Assumptions:
    None

    Source:
    None

    Inputs:
    fig             Plotly graph object

    Outputs: 
    fig             Plotly graph object with style

    Properties Used:
    N/A	
    """

    # Setup Axes Style

    axes_style = dict(
        ticks='outside', tickwidth=2, ticklen=6,                # Major Ticks
        showline=True, linewidth=2, linecolor='black',          # Axis Boundary
        showgrid=True, gridwidth=0.5, gridcolor='grey',         # Interior Grid
        zeroline=True, zerolinewidth=1, zerolinecolor='black',  # Include Zero
        minor = dict(                                           # Minor Ticks
            ticklen=3,
            griddash='dot'
        )
    )

    # Set for both X and Y Axes, setting X-axis to time-format

    full_fig = fig.full_figure_for_development()
    x_range = full_fig.layout.xaxis.range

    if x_range[1] > 120.:
        tick_time_interval = 60.
    elif x_range[1] > 60.:
        tick_time_interval = 15.
    else:
        tick_time_interval = 5.

    fig.update_xaxes(**axes_style,
                     tick0=0.0,
                     dtick=tick_time_interval,
                     rangemode="nonnegative")
    fig.update_yaxes(**axes_style)

    # Set Colorways, Margin

    fig.update_layout(
        plot_bgcolor='white',
        margin = dict(t=80, l=80, b=80, r=80))

    # Set Line and Marker Style 
    fig.update_traces(
        marker=dict(
            size=10,
            symbol='x-thin',
            line=dict(width=0.5)
            )
        )

    # Get List of Segments in the Plot
    trace_list  = [fig.data[trace]['name'] for trace in range(len(fig.data))]
    segment_set = [i for n,i in enumerate(trace_list) if i not in trace_list[:n]]

    # Setup the colors
    NS          = len(segment_set)+1
    color_list  = px.colors.sample_colorscale("inferno", [n/(NS -1) for n in range(NS)])
    

    # Create cycle-able iterator for assigning segment colors from inferno
    colorcycler = cycle(color_list)

    # Set segment line and marker color by iterating through the colorway
    for segment in segment_set:
        segment_color = next(colorcycler)
        fig.update_traces(marker=dict(line=dict(color=segment_color)),
                          line=dict(color=segment_color),
                          selector=dict(name=segment)) 
    return fig
