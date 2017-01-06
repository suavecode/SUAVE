# SUAVE imports
import SUAVE
from SUAVE.Core import Data, Units

# Package imports
import numpy as np
import time
import pylab as plt
import sklearn
from sklearn import gaussian_process
from sklearn import neighbors
from sklearn import svm

data_array = np.loadtxt('square_data.txt')
xy         = data_array[:,0:2]
CL         = data_array[:,2:3]
CD         = data_array[:,3:4]

coefficients = np.hstack([CL,CD])
grid_points  = xy
CL_data   = coefficients[:,0]
CD_data   = coefficients[:,1]
xy        = grid_points 

# Gaussian Process New
regr_cl = gaussian_process.GaussianProcess()
regr_cd = gaussian_process.GaussianProcess()
cl_surrogate = regr_cl.fit(xy, CL_data)
cd_surrogate = regr_cd.fit(xy, CD_data)          

# Gaussian Process New
#regr_cl = gaussian_process.GaussianProcessRegressor()
#regr_cd = gaussian_process.GaussianProcessRegressor()
#cl_surrogate = regr_cl.fit(xy, CL_data)
#cd_surrogate = regr_cd.fit(xy, CD_data)  

# KNN
#regr_cl = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
#regr_cd = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
#cl_surrogate = regr_cl.fit(xy, CL_data)
#cd_surrogate = regr_cd.fit(xy, CD_data)  

# SVR
#regr_cl = svm.SVR(C=500.)
#regr_cd = svm.SVR()
#cl_surrogate = regr_cl.fit(xy, CL_data)
#cd_surrogate = regr_cd.fit(xy, CD_data)  
    

# Plotting points
AoA_points = np.linspace(-2,8,100)*Units.deg
mach_points = np.linspace(.25,2.1,100)      

AoA_mesh,mach_mesh = np.meshgrid(AoA_points,mach_points)

CL_sur = np.zeros(np.shape(AoA_mesh))
CD_sur = np.zeros(np.shape(AoA_mesh))        

for jj in range(len(AoA_points)):
    for ii in range(len(mach_points)):
        CL_sur[ii,jj] = cl_surrogate.predict(np.array([AoA_mesh[ii,jj],mach_mesh[ii,jj]]))
        CD_sur[ii,jj] = cd_surrogate.predict(np.array([AoA_mesh[ii,jj],mach_mesh[ii,jj]]))


fig = plt.figure('Coefficient of Lift Surrogate Plot')    
plt_handle = plt.contour(AoA_mesh/Units.deg,mach_mesh,CL_sur,levels=None)
plt.clabel(plt_handle, inline=1, fontsize=10)
cbar = plt.colorbar()
plt.scatter(xy[:,0]/Units.deg,xy[:,1])
plt.xlabel('Angle of Attack (deg)')
plt.ylabel('Mach Number')
cbar.ax.set_ylabel('Coefficient of Lift')

plt.show() 