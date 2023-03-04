## @ingroup Visualization-Geometry
# plot_2d_rotor.py
#
# Created:  Mar 2020, M. Clarke
# Modified: Apr 2020, M. Clarke
#           Jul 2020, M. Clarke
#           Feb 2022, M. Clarke
#           Nov 2022, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from MARC.Core import Units

from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go

## @ingroup Visualization-Geometry
def plot_rotor(prop, face_color = 'red', edge_color = 'black' ,show_figure = True, save_figure = False, save_filename = "Propeller_Geometry", file_type = ".png"):
    """This plots the geometry of a propeller or rotor

    Assumptions:
    None

    Source:
    None

    Inputs:
    MARC.Components.Energy.Converters.Propeller()

    Outputs:
    Plots

    Properties Used:
    N/A
    """
    # initalize figure
    fig = make_subplots(rows=2, cols=2)

    df1 = pd.DataFrame(dict(x=prop.radius_distribution, y=prop.twist_distribution/Units.degrees))
    df2 = pd.DataFrame(dict(x=prop.radius_distribution, y=prop.chord_distribution))
    df3 = pd.DataFrame(dict(x=prop.radius_distribution, y=prop.max_thickness_distribution))
    df4 = pd.DataFrame(dict(x=prop.radius_distribution, y=prop.mid_chord_alignment))

    fig.append_trace(go.Line(df1), row=1, col=1)
    fig.append_trace(go.Line(df2), row=1, col=2)
    fig.append_trace(go.Line(df3), row=2, col=1)
    fig.append_trace(go.Line(df4), row=2, col=2)

    fig.update_xaxes(title_text="Radial Station", row=1, col=1)
    fig.update_yaxes(title_text="Twist (Deg)", row=1, col=1)
    fig.update_xaxes(title_text="Radial Station", row=1, col=2)
    fig.update_yaxes(title_text="Chord (m)", row=1, col=2)
    fig.update_xaxes(title_text="Radial Station", row=2, col=1)
    fig.update_yaxes(title_text="Thickness (m)", row=2, col=1)
    fig.update_xaxes(title_text="Radial Station", row=2, col=2)
    fig.update_yaxes(title_text="Mid Chord Alignment (m)", row=2, col=2)

    fig.update_layout(title_text="Propeller Geometry", height=700, showlegend=False)

    if save_figure:
        fig.write_image(save_filename + '_2D' + file_type)
    
    if show_figure:
        fig.write_html( save_filename + '.html', auto_open=True)

    return
