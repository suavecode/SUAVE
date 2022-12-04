## @ingroup Plots-Performance-Weights
# eVTOL_Sunburst.py
# 
# Created:    Dec 2022, J. Smart
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

import plotly
import plotly.express as px
import plotly.graph_objects as go

from copy import deepcopy

import pandas as pd
import numpy as np

## @ingroup Plots-Performance-Weights
def eVTOL_Sunburst(vehicle, *args, **kwargs):
    """Plots a sunburst plot based on the weight breakdown of the vehicle.

    Assumptions:
    None

    Source:
    None

    Inputs:
        vehicle                 [SUAVE.Vehicle]
            .weight_breakdown   [SUAVE.Data]

    Outputs: 
    Plots

    Properties Used:
    N/A	
    """

    weights = deepcopy(vehicle.weight_breakdown)

    labels = ["TOTAL"]
    parents = [""]
    values = [weights.total]
    del weights['total']

    new_labels, new_parents, new_values = make_sunburst_lists(weights)

    labels.extend(new_labels)
    parents.extend(new_parents)
    values.extend(new_values)

    chart_df = pd.DataFrame(list(zip(labels, parents, values)), columns = ['Label', 'Parent', 'Value'])
    filter_df = chart_df[chart_df['Value']>0]

    fig = go.Figure(go.Sunburst(
        labels=filter_df['Label'].tolist(),
        parents=filter_df['Parent'].tolist(),
        values=filter_df['Value'].tolist(),
        branchvalues="total",
        marker=dict(colorscale='Inferno_r')
    ))

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    fig.show()

    return

def make_sunburst_lists(weights, parent="TOTAL"):

    new_labels = []
    new_parents = []
    new_values = []

    for key, value in weights.items():

        if key == "total":
            continue
        else:
            if isinstance(weights[key], dict):
                new_labels.append(' '.join(key.split("_")).upper())
                new_parents.append(parent)
                new_values.append(weights[key].total)
                new_parent = ' '.join(key.split("_")).upper()
                l, p, v = make_sunburst_lists(weights[key], parent=new_parent)
                new_labels.extend(l)
                new_parents.extend(p)
                new_values.extend(v)
            else:
                new_labels.append(' '.join(key.split("_")).upper())
                new_parents.append(parent)
                new_values.append(weights[key])

    return new_labels, new_parents, new_values