## @ingroup Visualization-Geometry-Common
# contour_surface_slices.py
#
# Created :Nov 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------  
import plotly.graph_objects as go   

## @ingroup Visualization-Geometry-Common
def contour_surface_slice(x,y,z,values,color_scale):
    return go.Surface(x=x,y=y,z=z,surfacecolor=values,colorscale=color_scale, showscale=False) 
