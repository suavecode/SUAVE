#Turbofan_Network.py
#
# Created:  Anil Variyar, Feb 2016
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import scipy as sp
import datetime
import time
from SUAVE.Core import Units

import scipy
from scipy import interpolate


# python imports

import os, sys, shutil
from copy import deepcopy
from warnings import warn


from SUAVE.Core import Data
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components.Propulsors.Propulsor import Propulsor

from scipy.interpolate import griddata
import math




# ----------------------------------------------------------------------
#  Turbofan Network
# ----------------------------------------------------------------------

class Turbofan_Deck_I(Propulsor):


    def __defaults__(self):

        #setting the default values
        self.tag = 'Turbo_Fan'
        self.number_of_engines = 1.0
        self.nacelle_diameter  = 1.0
        self.engine_length     = 1.0
        self.bypass_ratio      = 1.0
        self.filename          = None
        self.model             = None

        self.power_setting_model = None
        self.gross_thrust_model  = None
        self.ram_drag_model      = None
        self.fuel_flow_model     = None
        self.sfc_model           = None

        self.model_type        = "rbf" #"2d_interpolation"
        self.engine_data       = None
        self.mach_numbers_list = None
        self.altitudes_list    = None
        self.local_mach_lib    = None
        self.scale             = 1.0


    _component_root_map = None


    def __init__(self,filename = None):

        #setting the filename if provided at initialization
        self.filename = filename


    def evaluate_thrust(self,state):
        results_o = self.evaluate_thrustt(state)

        conditions = state.conditions
        net_thrust = results_o.F*self.number_of_engines
        fuel_flow_rate = results_o.mdot*self.number_of_engines
        sfc = results_o.sfc

        F            = net_thrust*[1.0,0.0,0.0]
        mdot         = fuel_flow_rate
        F_vec        = conditions.ones_row(3) * 0.0
        F_vec[:,0]   = net_thrust[:,0] #F[:,0]
        F            = F_vec

        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot
        results.sfc                 = sfc

        #print results

        return results


    # linking the different network components
    def evaluate_thrustt(self,state):

        #Unpack
        conditions = state.conditions
        numerics   = state.numerics


        #extract mach number and altitude
        mach_number = conditions.freestream.mach_number
        altitude    = conditions.freestream.altitude/Units.ft
        throttle    = conditions.propulsion.throttle

        temp_throttle = deepcopy(throttle)
        throttle_scale = deepcopy(throttle)
        throttle_scale[:] = 1.0


        temp_throttle[throttle<0.0] = 0.0*throttle[throttle<0.0]/throttle[throttle<0.0]
        #throttle_scale[throttle<0.0] = throttle[throttle<0.0]

        temp_throttle[throttle>1.0] = 1.0*throttle[throttle>1.0]/throttle[throttle>1.0]
        throttle_scale[throttle>1.0] = throttle[throttle>1.0]

        power_setting = 21. + temp_throttle*(50.0-21.0)

        net_thrust     = np.zeros(mach_number.shape)
        fuel_flow_rate = np.zeros(mach_number.shape)
        sfc            = np.zeros(mach_number.shape)


        for iarr in xrange(len(mach_number)):


            thrust_eval = self.thrust_interp_lin(mach_number[iarr],altitude[iarr],power_setting[iarr])
            sfc_eval    = self.sfc_interp_lin(mach_number[iarr],altitude[iarr],power_setting[iarr])

            if(math.isnan(thrust_eval) or math.isnan(sfc_eval)):

                thrust_eval = self.thrust_interp_NN(mach_number[iarr],altitude[iarr],power_setting[iarr])
                sfc_eval    = self.sfc_interp_NN(mach_number[iarr],altitude[iarr],power_setting[iarr])


            net_thrust[iarr]     = self.scale*(thrust_eval)*throttle_scale[iarr]
            sfc[iarr]            = sfc_eval
            fuel_flow_rate[iarr] = (net_thrust[iarr])*sfc_eval*0.453592/3600.    #self.scale*fuel_flow*0.453592/3600.
            net_thrust[iarr]     = 4.44822*net_thrust[iarr]

            #if(thrust_eval<0):
                #net_thrust[iarr] = 0.0
                #fuel_flow_rate[iarr] = 0.0


        results = Data()
        results.F      = net_thrust
        results.mdot   = fuel_flow_rate
        results.sfc    = sfc

        return results




    def size(self,mach_number = None, altitude = None, conditions = None):


        number_of_engines         = self.number_of_engines
        sizing_thrust = self.total_design/float(number_of_engines)

        #read in the file and store the deck
        self.loadtext(self.filename)



        #call the atmospheric model to get the conditions at the specified altitude
        atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        p,T,rho,a,mu = atmosphere.compute_values(altitude,0.0)

        # setup conditions
        conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()

        # freestream conditions

        conditions.freestream.altitude           = np.atleast_1d(altitude)
        conditions.freestream.mach_number        = np.atleast_1d(mach_number)

        conditions.freestream.pressure           = np.atleast_1d(p)
        conditions.freestream.temperature        = np.atleast_1d(T)
        conditions.freestream.density            = np.atleast_1d(rho)
        conditions.freestream.dynamic_viscosity  = np.atleast_1d(mu)
        conditions.freestream.gravity            = np.atleast_1d(9.81)
        conditions.freestream.gamma              = np.atleast_1d(1.4)
        conditions.freestream.Cp                 = 1.4*287.87/(1.4-1)
        conditions.freestream.R                  = 287.87
        conditions.freestream.speed_of_sound     = np.atleast_1d(a)
        conditions.freestream.velocity           = conditions.freestream.mach_number * conditions.freestream.speed_of_sound

        # propulsion conditions
        conditions.propulsion.throttle           =  np.atleast_1d(1.0)

        sizing_state = Data()
        sizing_state.numerics = Data()
        sizing_state.conditions = conditions

        results = self.evaluate_thrustt(sizing_state)



        net_thrust = results.F

        self.scale = sizing_thrust/net_thrust




        #compute sls conditions
        #call the atmospheric model to get the conditions at the specified altitude
        atmosphere_sls = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        p,T,rho,a,mu = atmosphere_sls.compute_values(0.0,0.0)

        # setup conditions
        conditions_sls = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()

        # freestream conditions

        conditions_sls.freestream.altitude           = np.atleast_1d(0.)
        conditions_sls.freestream.mach_number        = np.atleast_1d(0.01)

        conditions_sls.freestream.pressure           = np.atleast_1d(p)
        conditions_sls.freestream.temperature        = np.atleast_1d(T)
        conditions_sls.freestream.density            = np.atleast_1d(rho)
        conditions_sls.freestream.dynamic_viscosity  = np.atleast_1d(mu)
        conditions_sls.freestream.gravity            = np.atleast_1d(9.81)
        conditions_sls.freestream.gamma              = np.atleast_1d(1.4)
        conditions_sls.freestream.Cp                 = 1.4*287.87/(1.4-1)
        conditions_sls.freestream.R                  = 287.87
        conditions_sls.freestream.speed_of_sound     = np.atleast_1d(a)
        conditions_sls.freestream.velocity           = conditions_sls.freestream.mach_number * conditions_sls.freestream.speed_of_sound

        # propulsion conditions
        conditions_sls.propulsion.throttle           =  np.atleast_1d(1.0)

        state_sls = Data()
        state_sls.numerics = Data()
        state_sls.conditions = conditions_sls
        results_sls = self.evaluate_thrustt(state_sls)
        self.sealevel_static_thrust = results_sls.F

        self.sealevel_static_mass_flow = results_sls.mdot
        self.sealevel_static_sfc = results_sls.sfc


        #self.sealevel_static_thrust = 125000.0






    def engine_out(self,state):


        temp_throttle = np.zeros(len(state.conditions.propulsion.throttle))

        for i in range(0,len(state.conditions.propulsion.throttle)):
            temp_throttle[i] = state.conditions.propulsion.throttle[i]
            state.conditions.propulsion.throttle[i] = 1.0



        results = self.evaluate_thrust(state)

        for i in range(0,len(state.conditions.propulsion.throttle)):
            state.conditions.propulsion.throttle[i] = temp_throttle[i]



        results.thrust_force_vector = results.thrust_force_vector/self.number_of_engines*(self.number_of_engines-1)
        results.vehicle_mass_rate   = results.vehicle_mass_rate/self.number_of_engines*(self.number_of_engines-1)



        return results




    def loadtext(self,filename):

        fid = open(filename,'r')
        no_of_rows = 0
        for line in fid:
            no_of_rows+=1
        fid.close()

        engine_data = np.zeros([no_of_rows,7])
        grid_points = np.zeros([no_of_rows,3])
        thrust_values = np.zeros(no_of_rows)
        sfc_values = np.zeros(no_of_rows)


        fid = open(filename,'r')

        no_of_rows = 0
        for line in fid:
            cols = line.split()
            engine_data[no_of_rows,0] = float(cols[0])
            engine_data[no_of_rows,1] = float(cols[1])
            engine_data[no_of_rows,2] = float(cols[2])
            engine_data[no_of_rows,3] = float(cols[3])
            engine_data[no_of_rows,4] = float(cols[4])
            engine_data[no_of_rows,5] = float(cols[5])
            engine_data[no_of_rows,6] = float(cols[6])

            grid_points[no_of_rows,0] = engine_data[no_of_rows,0]
            grid_points[no_of_rows,1] = engine_data[no_of_rows,1]
            grid_points[no_of_rows,2] = engine_data[no_of_rows,2]


            thrust_values[no_of_rows] = engine_data[no_of_rows,3]-engine_data[no_of_rows,4]
            sfc_values[no_of_rows] = engine_data[no_of_rows,6]


            no_of_rows+=1

            #print str(local_mach) + "_" + str(local_alt)+ "_" + str(local_power)
            #print  engine_data_dict[str(local_mach) + "_" + str(local_alt)+ "_" + str(local_power)]


        fid.close()

        thrust_interp_lin = scipy.interpolate.LinearNDInterpolator(grid_points, thrust_values)
        sfc_interp_lin = scipy.interpolate.LinearNDInterpolator(grid_points, sfc_values)

        thrust_interp_NN = scipy.interpolate.NearestNDInterpolator(grid_points, thrust_values)
        sfc_interp_NN = scipy.interpolate.NearestNDInterpolator(grid_points, sfc_values)


        self.thrust_interp_lin = thrust_interp_lin
        self.sfc_interp_lin    = sfc_interp_lin

        self.thrust_interp_NN = thrust_interp_NN
        self.sfc_interp_NN    = sfc_interp_NN



    __call__ = evaluate_thrust



