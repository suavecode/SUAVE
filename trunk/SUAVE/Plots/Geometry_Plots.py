## @ingroup Plots
# Geometry_Plots.py
# 
# Created:  Mar 2020, M. Clarke
#           Apr 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units 
import matplotlib.cm as cm
import matplotlib.pyplot as plt  
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry \
     import import_airfoil_geometry


## @ingroup Plots
# ------------------------------------------------------------------
# Vortex Lattice Method Panelization 
# ------------------------------------------------------------------
def plot_vehicle_vlm_panelization(vlm_geometry, save_figure = False, save_filename = "VLM_Panelization"):     
	"""This plots vortex lattice panels created when Fidelity Zero  Aerodynamics 
	Routine is initialized

	Assumptions:
	None

	Source:
	None

	Inputs:
	airfoil_geometry_files   <list of strings>

	Outputs: 
	Plots

	Properties Used:
	N/A	
	"""	
	face_color = [0.9,0.9,0.9] # grey        
	edge_color = [0, 0, 0]     # black
	alpha_val  = 0.5  
	fig = plt.figure(save_filename)
	axes = Axes3D(fig)    
	n_cp = vlm_geometry.n_cp 
	for i in range(n_cp): 
		X = [vlm_geometry.XA1[i],vlm_geometry.XB1[i],vlm_geometry.XB2[i],vlm_geometry.XA2[i]]
		Y = [vlm_geometry.YA1[i],vlm_geometry.YB1[i],vlm_geometry.YB2[i],vlm_geometry.YA2[i]]
		Z = [vlm_geometry.ZA1[i],vlm_geometry.ZB1[i],vlm_geometry.ZB2[i],vlm_geometry.ZA2[i]] 
		verts = [list(zip(X, Y, Z))]
		collection = Poly3DCollection(verts)
		collection.set_facecolor(face_color)
		collection.set_edgecolor(edge_color)
		collection.set_alpha(alpha_val)
		axes.add_collection3d(collection)     
		max_range = np.array([vlm_geometry.X.max()-vlm_geometry.X.min(), vlm_geometry.Y.max()-vlm_geometry.Y.min(), vlm_geometry.Z.max()-vlm_geometry.Z.min()]).max() / 2.0    
		mid_x = (vlm_geometry.X .max()+vlm_geometry.X .min()) * 0.5
		mid_y = (vlm_geometry.Y .max()+vlm_geometry.Y .min()) * 0.5
		mid_z = (vlm_geometry.Z .max()+vlm_geometry.Z .min()) * 0.5
		axes.set_xlim(mid_x - max_range, mid_x + max_range)
		axes.set_ylim(mid_y - max_range, mid_y + max_range)
		axes.set_zlim(mid_z - max_range, mid_z + max_range)          

	axes.scatter(vlm_geometry.XC,vlm_geometry.YC,vlm_geometry.ZC, c='r', marker = 'o' )
	plt.axis('off')
	plt.grid(None)
	return


# ------------------------------------------------------------------
# Plot Airfol
# ------------------------------------------------------------------
## @ingroup Plots
def plot_airfoil(airfoil_names,  line_color = 'k-', save_figure = False, save_filename = "Airfoil_Geometry", file_type = ".png"):
	"""This plots all airfoil defined in the list "airfoil_names" 

	Assumptions:
	None

	Source:
	None

	Inputs:
	airfoil_geometry_files   <list of strings>

	Outputs: 
	Plots

	Properties Used:
	N/A	
	"""
	# get airfoil coordinate geometry     
	airfoil_data = import_airfoil_geometry(airfoil_names)       

	for i in range(len(airfoil_names)):
		# separate x and y coordinates 
		airfoil_x  = airfoil_data.x_coordinates[i] 
		airfoil_y  = airfoil_data.y_coordinates[i]    

		name = save_filename + '_' + str(i)
		fig  = plt.figure(name)
		axes = fig.add_subplot(1,1,1)
		axes.set_title(airfoil_names[i])
		axes.plot(airfoil_x, airfoil_y , line_color )                  
		#axes.set_aspect('equal')
		axes.axis('equal')
		if save_figure:
			plt.savefig(name + file_type)          

	return

# ------------------------------------------------------------------
#   Propeller Geoemtry 
# ------------------------------------------------------------------
## @ingroup Plots
def plot_propeller_geometry(prop, line_color = 'bo-', save_figure = False, save_filename = "Propeller_Geometry", file_type = ".png"):
	"""This plots the geoemtry of a propeller or rotor

	Assumptions:
	None

	Source:
	None

	Inputs:
	SUAVE.Components.Energy.Converters.Propeller()

	Outputs: 
	Plots

	Properties Used:
	N/A	
	"""	
	# unpack
	Rt     = prop.tip_radius          
	Rh     = prop.hub_radius          
	num_B  = prop.number_blades       
	a_sec  = prop.airfoil_geometry          
	a_secl = prop.airfoil_polar_stations
	beta   = prop.twist_distribution         
	b      = prop.chord_distribution         
	r      = prop.radius_distribution        
	t      = prop.max_thickness_distribution

	# prepare plot parameters
	dim = len(b)
	theta = np.linspace(0,2*np.pi,num_B+1)

	fig = plt.figure(save_filename)
	fig.set_size_inches(10, 8)     
	axes = plt.axes(projection='3d') 
	axes.set_zlim3d(-1, 1)        
	axes.set_ylim3d(-1, 1)        
	axes.set_xlim3d(-1, 1)     

	chord = np.outer(np.linspace(0,1,10),b)
	if r == None:
		r = np.linspace(Rh,Rt, len(b))
	for i in range(num_B):  
		# plot propeller planfrom
		surf_x = np.cos(theta[i]) * (chord*np.cos(beta)) - np.sin(theta[i]) * (r) 
		surf_y = np.sin(theta[i]) * (chord*np.cos(beta)) + np.cos(theta[i]) * (r) 
		surf_z = chord*np.sin(beta)                                
		axes.plot_surface(surf_x ,surf_y ,surf_z, color = 'gray')

		if  a_sec != None and a_secl != None:
			# check dimension of section  
			dim_sec = len(a_secl)
			if dim_sec != dim:
				raise AssertionError("Number of sections not equal to number of stations") 

			# get airfoil coordinate geometry     
			airfoil_data = import_airfoil_geometry(a_sec)       

			#plot airfoils 
			for j in range(dim):
				airfoil_max_t = airfoil_data.thickness_to_chord[a_secl[j]]
				airfoil_xp = b[j] - airfoil_data.x_coordinates[a_secl[j]]*b[j]
				airfoil_yp = r[j]*np.ones_like(airfoil_xp)            
				airfoil_zp = airfoil_data.y_coordinates[a_secl[j]]*b[j]  * (t[j]/(airfoil_max_t*b[j]))

				transformation_1 = [[np.cos(beta[j]),0 , -np.sin(beta[j])], [0 ,  1 , 0] , [np.sin(beta[j]) , 0 , np.cos(beta[j])]]
				transformation_2 = [[np.cos(theta[i]) ,-np.sin(theta[i]), 0],[np.sin(theta[i]) , np.cos(theta[i]), 0], [0 ,0 , 1]] 
				transformation  = np.matmul(transformation_2,transformation_1)

				airfoil_x = np.zeros(len(airfoil_yp))
				airfoil_y = np.zeros(len(airfoil_yp))
				airfoil_z = np.zeros(len(airfoil_yp))     

				for k in range(len(airfoil_yp)):
					vec_1 = [[airfoil_xp[k]],[airfoil_yp[k]], [airfoil_zp[k]]]
					vec_2  = np.matmul(transformation,vec_1)
					airfoil_x[k] = vec_2[0]
					airfoil_y[k] = vec_2[1]
					airfoil_z[k] = vec_2[2]

				axes.plot3D(airfoil_x, airfoil_y, airfoil_z, color = 'gray')

	if save_figure:
		plt.savefig(save_filename + file_type)  

	return