# Loads.py
#

""" SUAVE Methods for Loads Analysis
"""

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy
from SUAVE.Structure  import Data
#from SUAVE.Attributes import Constants


# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

# # TODO: declassify methods

class Loads(object):
    ''' SUAVE.Loads()
    '''
    
    # default attributes
    attribute1 = None
    attribute2 = []



    def __init__(self,AV):
        ''' class initialization
        '''
        # store pointer to AeroVehicle
        self.AV = AV
        
        # Info for V-N diagram
        self.gust_load_factor=0 # FAR part 25
        self.maneuver_load_factor=0 #FAR part 25
        self.tail_load_factor=0
        # Current load factor
        self.load_factor=0

        return
    
    def method1(self,inputs):
        ''' outputs = method1(inputs)
            more documentation
        '''
        
        #code        
        outputs = 'hello aero!'
    
    def corner_points(self,altitude,cruise_mach):
        ''' n1 v1 n2 v2 n3 v3 n4 v4 n5 v5 = corner_points(self,altitude,cruise_mach)
            (ordered clockwise)
            Computes the corner points of the V-N diagram at a specified design altitude
            and specified design cruise mach number
            Using Vd=1.25 Vc, n_max=2.5, n_min=-1.0
            Currently no gusts taken into account
            Source: FAR 25 as described in AA241 notes
        '''
        #
        n_max=2.5 #self.AV.Desc.PASS.results['nlimit'] # default value 2.5
        n_min=-1.0

        # Get necessary atmospheric values
        Atm=EarthUSStandardAtmopshere(alt=altitude,lapse='isa',oat=15.0,dt=0,unitT='C',unit='icao')
        vsnd=Atm.vsnd # basically just guessing the syntax
        density=Atm.sigma*Constants.rho0

        # Calculate design velocities
        vc=cruise_mach*vsnd # Design velocity
        vd=1.25*vc # Limit velocity

        # Get necessary properties of the vehicle 
        wt=self.AV.Desc.PASS.results['weight.maxzf'] # this may not be the correct weight to use; 
        cl_mx=self.AV.Desc.PASS.inputs['clhmax'] # max CL 
        cl_mn=self.AV.Desc.PASS.inputs['clhmax']*(-1) # min CL: don't see it in description, using -1*Clmax for now
        sref=self.AV.Desc.PASS.inputs['sref'] # surface area for calculating lift

        # some points which won't change (until gusts are calculated)
        n2=n_max; v2=vd;
        n1=n_max;
        n3=0; v3=vd; 
        n4=n_min; v4=vc;
        n5=n_min;
        # Find the velocity where load from CLmax is = nmax
        v1=(wt*n_max/(cl_mx*sref*0.5*density))**0.5
        if v1>vd: v1=vd
                        
        # Find the point where load from Clmin is < nmin
        v5=(wt*n_min/(cl_mn*sref*0.5*density))**0.5
        if v5>vc: v5=vc

        return n1,v1,n2,v2, n3, v3, n4, v4, n5, v5

    def current_load_factor(self):
        ''' n = current_load_factor(self)
        Returns the load factor at the current flight condition
        n = L/W
        '''
        lift=self.AV.getLift() # (placeholder)
        wt=self.AV.Desc.PASS.results['weight.maxzf'] # this may not be the correct weight to use; 
        # get gravity constant from Common.Constants class
        gravity=Constants.grav
        # compute maneuver+gravitational acceleration
        acceleration=self.AV.getAcceleration() + gravity # (placeholder)
        # compute load factor
        n = lift/(mass*acceleration)

    def gust_load(self,gust):
        ''' gust_load_factor = gust_load_calculation(self,gust magnitude)
        Computed from expression(s) in FAR Part 25 [given in AA241 notes] 
        Depends on gust velocity, weight, surface area, equivalent airspeed, 
        lift coefficient slope
        '''
        self.gust_load_factor=0



    def plot_VN_diagram(self):
        ''' plot_VN_diagram(self)
        Iterates over flight velocities, gusts, and maneuvers to develop the 
        V-N diagram.
        '''
        # Call maneuver_load_factor(), gust_load() at relevant flight velocites 
        # to get load factors.
        # Plots (or stores data to be used elsewhere??) V-N diagram.
        # send corner points to weights
        
        
        


