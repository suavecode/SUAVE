## @ingroup Visualization-Geometry-Three_Dimensional
# plot_3d_vehicle.py
#
# Created : Mar 2020, M. Clarke
# Modified: Apr 2020, M. Clarke
# Modified: Jul 2020, M. Clarke
# Modified: Jul 2021, E. Botero
# Modified: Oct 2021, M. Clarke
# Modified: Dec 2021, M. Clarke
# Modified: Feb 2022, R. Erhard
# Modified: Mar 2022, R. Erhard
# Modified: Sep 2022, M. Clarke
# Modified: Nov 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import numpy as np 
import plotly.graph_objects as go  
from MARC.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.generate_vortex_distribution    import generate_vortex_distribution 
from MARC.Analyses.Aerodynamics import Vortex_Lattice
from MARC.Visualization.Geometry.Three_Dimensional.plot_3d_fuselage import plot_3d_fuselage
from MARC.Visualization.Geometry.Three_Dimensional.plot_3d_wing     import plot_3d_wing 
from MARC.Visualization.Geometry.Three_Dimensional.plot_3d_nacelle  import plot_3d_nacelle
from MARC.Visualization.Geometry.Three_Dimensional.plot_3d_rotor    import plot_3d_rotor

## @ingroup Visualization-Geometry-Three_Dimensional
def plot_3d_vehicle(vehicle,
                    plot_axis = False,
                    save_figure = False,
                    alpha = 1.0,
                    plot_wing_control_points = True,
                    plot_rotor_wake_vortex_core = False,
                    save_filename = "Vehicle_Geometry",
                    x_axis_limit = None,
                    y_axis_limit = None,
                    z_axis_limit = None):
    """This plots vortex lattice panels created when Fidelity Zero  Aerodynamics
    Routine is initialized

    Assumptions:
    None

    Source:
    None

    Inputs:
    vehicle

    Outputs:
    Plots

    Properties Used:
    N/A
    """

    print("\nPlotting vehicle") 
    camera        = dict(up=dict(x=0.5, y=0.5, z=1), center=dict(x=0, y=0, z= -.75), eye=dict(x=-1.5, y=-1.5, z=.8))
    plot_data     = []
    
    # unpack vortex distribution
    try:
        VD = vehicle.vortex_distribution
    except:
        settings = Vortex_Lattice().settings
        settings.number_spanwise_vortices  = 25
        settings.number_chordwise_vortices = 5
        settings.spanwise_cosine_spacing   = False
        settings.model_fuselage            = False
        settings.model_nacelle             = False
        VD = generate_vortex_distribution(vehicle,settings)

    
    # -------------------------------------------------------------------------
    # DEFINE PLOT LIMITS 
    # -------------------------------------------------------------------------  
    if x_axis_limit == None: 
        x_min,x_max = np.minimum(0,np.min(VD.XC)*1.2), np.maximum(np.max(VD.XC)*1.2,10)
    else:
        x_min,x_max = x_axis_limit,x_axis_limit
    if y_axis_limit == None: 
        y_min,y_max = np.min(VD.YC)*1.2, np.max(VD.YC)*1.2
    else:
        y_min,y_max = y_axis_limit,y_axis_limit
    if z_axis_limit == None: 
        z_min,z_max = -1*np.max(VD.ZC), 2.5*np.max(VD.ZC)
    else:
        z_min,z_max = z_axis_limit,z_axis_limit
    
    # -------------------------------------------------------------------------
    # PLOT WING
    # ------------------------------------------------------------------------- 
    plot_data       = plot_3d_wing(plot_data,VD,color_map ='greys')
    if  plot_wing_control_points: 
        ctrl_pts = go.Scatter3d(x=VD.XC, y=VD.YC, z=VD.ZC,
                                    mode  = 'markers',
                                    marker= dict(size=6,color='red',opacity=0.8),
                                    line  = dict(color='red',width=2))
        
        plot_data.append(ctrl_pts) 
 
    # -------------------------------------------------------------------------
    # PLOT FUSELAGE
    # ------------------------------------------------------------------------- 
    for fus in vehicle.fuselages:
        plot_data = plot_3d_fuselage(plot_data,fus,color_map = 'teal')

    # -------------------------------------------------------------------------
    # PLOT NACELLE
    # ------------------------------------------------------------------------- 
    number_of_airfoil_points = 21
    tessellation             = 24
    for nacelle in vehicle.nacelles:    
        plot_data = plot_3d_nacelle(plot_data,nacelle,tessellation,number_of_airfoil_points,color_map = 'darkmint')  
        
    # -------------------------------------------------------------------------
    # PLOT ROTORS
    # ------------------------------------------------------------------------- 
    number_of_airfoil_points = 11
    for network in vehicle.networks:
        plot_data = plot_3d_energy_network(plot_data,network,number_of_airfoil_points,color_map = 'turbid' )

    fig = go.Figure(data=plot_data)
    fig.update_scenes(aspectmode   = 'cube',
                      xaxis_visible=plot_axis,
                      yaxis_visible=plot_axis,
                      zaxis_visible=plot_axis
                      )
    fig.update_layout( 
             width     = 1500,
             height    = 1500, 
             scene = dict(
                        xaxis = dict(backgroundcolor="grey", gridcolor="white", showbackground=plot_axis,
                                     zerolinecolor="white", range=[x_min,x_max]),
                        yaxis = dict(backgroundcolor="grey", gridcolor="white", showbackground=plot_axis, 
                                     zerolinecolor="white", range=[y_min,y_max]),
                        zaxis = dict(backgroundcolor="grey",gridcolor="white",showbackground=plot_axis,
                                     zerolinecolor="white", range=[z_min,z_max])),             
             scene_camera=camera) 
    fig.update_coloraxes(showscale=False) 
    fig.update_traces(opacity = alpha)
    if save_figure:
        fig.write_image(save_filename + ".png")
    fig.show()
        
    return 

## @ingroup Visualization-Geometry-Three_Dimensional
def plot_3d_energy_network(plot_data,network,number_of_airfoil_points,color_map):
    """ This plots the 3D surface of the network

    Assumptions:
    None

    Source:
    None

    Inputs:
    network            - network data structure
    network_face_color - color of panel
    network_edge_color - color of panel edge 

    Properties Used:
    N/A
    """ 
    plot_axis     = False 
    save_figure   = False 
    save_filename = 'Rotor'
    if ('rotors' in network.keys()):
        rots = network.rotors 
        for rot in rots:  
            plot_data = plot_3d_rotor(rot,save_filename,save_figure,plot_data,plot_axis,0,number_of_airfoil_points,color_map) 
 
    return plot_data