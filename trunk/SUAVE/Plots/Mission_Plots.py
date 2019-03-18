import SUAVE
from SUAVE.Core import Units 
import pylab as plt 
import matplotlib
import matplotlib.pyplot as plt  
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from SUAVE.Methods.Aerodynamics.XFOIL.compute_airfoil_polars import read_airfoil_geometry

# ------------------------------------------------------------------
#   Altitude, SFC & Weight
# ------------------------------------------------------------------
def plot_altitude_sfc_weight(results, save_figure = False, save_filename = "Altitude_SFC_Weight"):
    axis_font = {'fontname':'Arial', 'size':'14'} 
    fig = plt.figure(save_filename,figsize=(8,10))
    for segment in results.segments.values():

        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        aoa      = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        mass     = segment.conditions.weights.total_mass[:,0] / Units.lb
        altitude = segment.conditions.freestream.altitude[:,0] / Units.ft
        mdot     = segment.conditions.weights.vehicle_mass_rate[:,0]
        thrust   =  segment.conditions.frames.body.thrust_force_vector[:,0]
        sfc      = (mdot / Units.lb) / (thrust /Units.lbf) * Units.hr

        axes = fig.add_subplot(3,1,1)
        axes.plot( time , altitude , 'bo-')
        axes.set_ylabel('Altitude (ft)',axis_font)
        axes.grid(True)

        axes = fig.add_subplot(3,1,3)
        axes.plot( time , sfc , 'bo-' )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('sfc (lb/lbf-hr)',axis_font)
        axes.grid(True)

        axes = fig.add_subplot(3,1,2)
        axes.plot( time , mass , 'ro-' )
        axes.set_ylabel('Weight (lb)',axis_font)
        axes.grid(True)
        
    if save_figure:
        plt.savefig(save_filename + ".png")  
        
    return


# ------------------------------------------------------------------
#   Aircraft Velocities
# ------------------------------------------------------------------
def plot_aircraft_velocities(results, save_figure = False, save_filename = "Aircraft_Velocities"):
    axis_font = {'fontname':'Arial', 'size':'14'}  
    fig = plt.figure(save_filename,figsize=(8,10))
    for segment in results.segments.values():

        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        Lift     = -segment.conditions.frames.wind.lift_force_vector[:,2]
        Drag     = -segment.conditions.frames.wind.drag_force_vector[:,0] / Units.lbf
        Thrust   = segment.conditions.frames.body.thrust_force_vector[:,0] / Units.lb
        velocity = segment.conditions.freestream.velocity[:,0]
        pressure = segment.conditions.freestream.pressure[:,0]
        density  = segment.conditions.freestream.density[:,0]
        EAS      = velocity * np.sqrt(density/1.225)
        mach     = segment.conditions.freestream.mach_number[:,0]

        axes = fig.add_subplot(3,1,1)
        axes.plot( time , velocity / Units.kts, 'bo-')
        axes.set_ylabel('velocity (kts)',axis_font)
        axes.grid(True)

        axes = fig.add_subplot(3,1,2)
        axes.plot( time , EAS / Units.kts, 'bo-')
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Equivalent Airspeed',axis_font)
        axes.grid(True)    
        
        axes = fig.add_subplot(3,1,3)
        axes.plot( time , mach , 'bo-')
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Mach',axis_font)
        axes.grid(True)    
        
    if save_figure:
        plt.savefig(save_filename + ".png") 
        
    return

# ------------------------------------------------------------------
#   Disc and Power Loadings
# ------------------------------------------------------------------
def plot_disc_power_loading(results, save_figure = False, save_filename = "Disc_Power_Loading"):
    axis_font = {'fontname':'Arial', 'size':'14'} 
    fig = plt.figure(save_filename)
    fig.set_size_inches(10, 8) 
    for i in range(len(results.segments)):  

        time  = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        DL    = results.segments[i].conditions.propulsion.disc_loading
        PL    = results.segments[i].conditions.propulsion.power_loading   
   
        axes = fig.add_subplot(2,1,1)
        axes.plot(time, DL, 'bo-')
        axes.set_ylabel('lift disc power lb/ft2',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)
        axes.grid(True)       
  
        axes = fig.add_subplot(2,1,2)
        axes.plot(time, PL, 'bo-' )       
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('lift power loading (lb/hp)',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)     
        axes.grid(True)         

    if save_figure:
        plt.savefig(save_filename + ".png")          
        
    return


# ------------------------------------------------------------------
#   Aerodynamic Coefficients
# ------------------------------------------------------------------
def plot_aerodynamic_coefficients(results, save_figure = False, save_filename = "Aerodynamic_Coefficients"):
    axis_font = {'fontname':'Arial', 'size':'14'}  
    fig = plt.figure(save_filename)
    fig.set_size_inches(10, 8)
    for segment in results.segments.values():

        time = segment.conditions.frames.inertial.time[:,0] / Units.min
        cl   = segment.conditions.aerodynamics.lift_coefficient[:,0,None] 
        cd   = segment.conditions.aerodynamics.drag_coefficient[:,0,None] 
        aoa  = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        l_d  = cl/cd

        axes = fig.add_subplot(4,1,1)
        axes.plot( time , aoa , 'bo-' )
        axes.set_ylabel('Angle of Attack (deg)',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False) 
        axes.grid(True)

        axes = fig.add_subplot(4,1,2)
        axes.plot( time , cl, 'bo-' )
        axes.set_ylabel('CL',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False) 
        axes.grid(True)    
        
        axes = fig.add_subplot(4,1,3)
        axes.plot( time , cd, 'bo-' )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('CD',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False) 
        axes.grid(True)    
        
        axes = fig.add_subplot(4,1,4)
        axes.plot( time , l_d, 'bo-' )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('L/D',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False) 
        axes.grid(True)            
                
    if save_figure:
        plt.savefig(save_filename + ".png") 
        
    return

# ------------------------------------------------------------------
#   Aerodynamic Forces
# ------------------------------------------------------------------
def plot_aerodynamic_forces(results, save_figure = False, save_filename = "Aerodynamic_Forces"):
    axis_font = {'fontname':'Arial', 'size':'14'}  
    fig = plt.figure(save_filename,figsize=(8,6))
    for segment in results.segments.values():

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0] / Units.lbf
        eta    = segment.conditions.propulsion.throttle[:,0]

        axes = fig.add_subplot(2,1,1)
        axes.plot( time , Thrust , 'bo-')
        axes.set_ylabel('Thrust (lbf)',axis_font)
        axes.grid(True)

        axes = fig.add_subplot(2,1,2)
        axes.plot( time , eta , 'bo-' )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Throttle',axis_font)
        axes.grid(True)	 
        
    if save_figure:
        plt.savefig(save_filename + ".png") 
            
    return


# ------------------------------------------------------------------
#   Drag Components
# ------------------------------------------------------------------
def plot_drag_components(results, save_figure = False, save_filename = "Drag_Components"):
    axis_font = {'fontname':'Arial', 'size':'14'} 
    fig = plt.figure(save_filename,figsize=(8,10))
    axes = plt.gca()
    for i, segment in enumerate(results.segments.values()):

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        drag_breakdown = segment.conditions.aerodynamics.drag_breakdown
        cdp = drag_breakdown.parasite.total[:,0]
        cdi = drag_breakdown.induced.total[:,0]
        cdc = drag_breakdown.compressible.total[:,0]
        cdm = drag_breakdown.miscellaneous.total[:,0]
        cd  = drag_breakdown.total[:,0]

         
        axes.plot( time , cdp , 'ko-', label='CD parasite' )
        axes.plot( time , cdi , 'bo-', label='CD induced' )
        axes.plot( time , cdc , 'go-', label='CD compressibility' )
        axes.plot( time , cdm , 'yo-', label='CD miscellaneous' )
        axes.plot( time , cd  , 'ro-', label='CD total'   )
        if i == 0:
            axes.legend(loc='upper center')   

    axes.set_xlabel('Time (min)',axis_font)
    axes.set_ylabel('CD',axis_font)
    axes.grid(True)         
    
    if save_figure:
        plt.savefig(save_filename + ".png") 
        
    return


# ------------------------------------------------------------------
#   Electronic Conditions
# ------------------------------------------------------------------
def plot_electronic_conditions(results, save_figure = False, save_filename = "Electronic_Conditions"):
    axis_font = {'fontname':'Arial', 'size':'14'} 
    fig = plt.figure(save_filename)
    fig.set_size_inches(10, 8)
    for i in range(len(results.segments)):  
    
        time     = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        eta      = results.segments[i].conditions.propulsion.throttle[:,0]
        eta_l    = results.segments[i].conditions.propulsion.throttle[:,0]
        energy   = results.segments[i].conditions.propulsion.battery_energy[:,0] 
        volts    = results.segments[i].conditions.propulsion.voltage_under_load[:,0] 
        volts_oc = results.segments[i].conditions.propulsion.voltage_open_circuit[:,0]     
        current = results.segments[i].conditions.propulsion.current[:,0]      
        battery_amp_hr = (energy*0.000277778)/volts
        C_rating   = current/battery_amp_hr
        
        axes = fig.add_subplot(2,2,1)
        axes.plot(time, eta, 'bo-',label='Forward Motor')
        axes.plot(time, eta_l, 'ro-',label='Lift Motors')
        axes.set_ylabel('Throttle',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)
        axes.grid(True)       
    
        axes = fig.add_subplot(2,2,2)
        axes.plot(time, energy*0.000277778, 'bo-')
        axes.set_ylabel('Battery Energy (W-hr)',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)      
        axes.grid(True)   
    
        axes = fig.add_subplot(2,2,3)
        axes.plot(time, volts, 'bo-',label='Under Load')
        axes.plot(time,volts_oc, 'ro-',label='Open Circuit')
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Battery Voltage (Volts)',axis_font)  
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)     
        axes.grid(True)         
        
        axes = fig.add_subplot(2,2,4)
        axes.plot(time, C_rating, 'bo-')
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('C-Rating (C)',axis_font)  
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)     
        axes.grid(True)
 
    if save_figure:
        plt.savefig(save_filename + ".png")       
        
    return


# ------------------------------------------------------------------
#   Flight Conditions
# ------------------------------------------------------------------
def plot_flight_conditions(results, save_figure = False, save_filename = "Flight_Conditions"):
    axis_font = {'fontname':'Arial', 'size':'14'} 
    fig = plt.figure(save_filename)
    fig.set_size_inches(10, 8)
    for segment in results.segments.values():

        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        airspeed = segment.conditions.freestream.velocity[:,0] 
        theta    = segment.conditions.frames.body.inertial_rotations[:,1,None] / Units.deg
        cl       = segment.conditions.aerodynamics.lift_coefficient[:,0,None] 
        cd       = segment.conditions.aerodynamics.drag_coefficient[:,0,None] 
        aoa      = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        
        x        = segment.conditions.frames.inertial.position_vector[:,0]
        y        = segment.conditions.frames.inertial.position_vector[:,1]
        z        = segment.conditions.frames.inertial.position_vector[:,2]
        altitude = segment.conditions.freestream.altitude[:,0]
        
        axes = fig.add_subplot(2,2,1)
        axes.plot(time, altitude, 'bo-')
        axes.set_ylabel('Altitude (m)',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)         
        axes.grid(True)            

        axes = fig.add_subplot(2,2,2)
        axes.plot( time , airspeed , 'bo-' )
        axes.set_ylabel('Airspeed (m/s)',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False) 
        axes.grid(True)

        axes = fig.add_subplot(2,2,3)
        axes.plot( time , theta, 'bo-' )
        axes.set_ylabel('Pitch Angle (deg)',axis_font)
        axes.set_xlabel('Time (min)',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False) 
        axes.grid(True)    
        
        axes = fig.add_subplot(2,2,4)
        axes.plot( time , x, 'bo-', time , y, 'go-' , time , z, 'ro-')
        axes.set_ylabel('Displacement',axis_font)
        axes.set_xlabel('Time (min)',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False) 
        axes.grid(True)           
        
    if save_figure:
        plt.savefig(save_filename + ".png")
        
    return


# ------------------------------------------------------------------
#   Propulsion Conditions
# ------------------------------------------------------------------
def plot_propulsor_conditions(results, save_figure = False, save_filename = "Propulsor"):
    axis_font = {'fontname':'Arial', 'size':'14'} 
    fig = plt.figure(save_filename) 
    for i in range(len(results.segments)):  

        time   = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        rpm    = results.segments[i].conditions.propulsion.rpm  [:,0] 
        thrust = results.segments[i].conditions.frames.body.thrust_force_vector[:,2]
        torque = results.segments[i].conditions.propulsion.motor_torque 
        effp   = results.segments[i].conditions.propulsion.etap[:,0]
        effm   = results.segments[i].conditions.propulsion.etam[:,0]
        ts     = results.segments[i].conditions.propulsion.tip_speed[:,0]
        
        axes = fig.add_subplot(2,3,1)
        axes.plot(time, rpm, 'bo-')
        axes.set_ylabel('RPM',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)
        axes.grid(True)       

        axes = fig.add_subplot(2,3,2)
        axes.plot(time, -thrust, 'bo-')
        axes.set_ylabel('Thrust (N)',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)      
        axes.grid(True)   

        axes = fig.add_subplot(2,3,3)
        axes.plot(time, torque, 'bo-' )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Torque (N-m)',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)     
        axes.grid(True)   
        
        axes = fig.add_subplot(2,3,4)
        axes.plot(time, effp, 'bo-' )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Propeller Efficiency (N-m)',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)     
        axes.grid(True)           
        plt.ylim((0,1))
        
        axes = fig.add_subplot(2,3,5)
        axes.plot(time, effm, 'bo-' )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Motor Efficiency (N-m)',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)     
        axes.grid(True)
        
        axes = fig.add_subplot(2,3,6)
        axes.plot(time, ts, 'bo-' )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Tip Speed (ft/s)',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)     
        axes.grid(True)       
        
    if save_figure:
        plt.savefig(save_filename + ".png")  
            
    return


# ------------------------------------------------------------------
#   Stability Coefficients
# ------------------------------------------------------------------
def plot_stability_coefficients(results, save_figure = False, save_filename = "Stability_Coefficients"):
    axis_font = {'fontname':'Arial', 'size':'14'} 
    fig = plt.figure(save_filename)
    fig.set_size_inches(10, 8)
    for segment in results.segments.values(): 
        time = segment.conditions.frames.inertial.time[:,0] / Units.min
        cm   = segment.conditions.stability.static['CM'][:,0,None] 
        cm_alpha   = segment.conditions.stability.static['Cm_alpha'][:,0,None] 
        SM         = ((segment.conditions.stability.static['NP'][:,0,None] - 2.0144 )/ 0.9644599977664836)*100  
        aoa      = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        
        axes = fig.add_subplot(2,2,1)
        axes.plot( time , aoa, 'bo-' )
        axes.set_ylabel(r'$AoA$',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False) 
        axes.grid(True)    
         
        axes = fig.add_subplot(2,2,2)
        axes.plot( time , cm, 'bo-' )
        axes.set_ylabel(r'$C_M$',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False) 
        axes.grid(True)    
        
        axes = fig.add_subplot(2,2,3)
        axes.plot( time , cm_alpha, 'bo-' )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel(r'$C_M\alpha$',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False) 
        axes.grid(True)    
        
        axes = fig.add_subplot(2,2,4)
        axes.plot( time , SM, 'bo-' )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Static Margin (%)',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False) 
        axes.grid(True) 
    
    if save_figure:
        plt.savefig(save_filename + ".png")
        
    return


# ------------------------------------------------------------------
#   Propeller Geoemtry 
# ------------------------------------------------------------------
def plot_propeller_geometry(prop, save_figure = False, save_filename = "Propeller_Geometry"):   
    # unpack
    Rt     = prop.tip_radius          
    Rh     = prop.hub_radius          
    num_B  = prop.number_blades       
    a_sec  = prop.airfoil_sections           
    a_secl = prop.airfoil_section_location   
    beta   = prop.twist_distribution         
    b      = prop.chord_distribution         
    r      = prop.radius_distribution        
    t      = prop.max_thickness_distribution
    
    # prepare plot parameters
    dim = len(b)
    theta = np.linspace(0,2*np.pi,num_B+1)
    
    fig = plt.figure("Propeller Geoemtry")
    ax = plt.axes(projection='3d') 
    ax.set_zlim3d(-1, 1)        
    ax.set_ylim3d(-1, 1)        
    ax.set_xlim3d(-1, 1)     
    
    chord = np.outer(np.linspace(0,1,10),b)
    for i in range(num_B):  
        # plot propeller planfrom
        surf_x = np.cos(theta[i]) * (chord*np.cos(beta)) - np.sin(theta[i]) * (r) 
        surf_y = np.sin(theta[i]) * (chord*np.cos(beta)) + np.cos(theta[i]) * (r) 
        surf_z = chord*np.sin(beta)                                
        ax.plot_surface(surf_x ,surf_y ,surf_z, color = 'gray')
        
        if  a_sec != None and a_secl != None:
            # check dimension of section  
            dim_sec = len(a_secl)
            if dim_sec != dim:
                raise AssertionError("Number of sections not equal to number of stations") 
                      
            # get airfoil coordinate geometry     
            airfoil_data = read_airfoil_geometry(a_sec)       
            
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
                            
                ax.plot3D(airfoil_x, airfoil_y, airfoil_z, color = 'gray')
                
    if save_figure:
        plt.savefig(save_filename + ".png")  
                    
    return


# ------------------------------------------------------------------
#   Propeller Performance
# ------------------------------------------------------------------
def plot_propeller_performance(noise, save_figure = False, save_filename = "Propeller_Performance"): 
    axis_font = {'fontname':'Arial', 'size':'14'} 
    fig.set_size_inches(10, 8)  
    fig = plt.figure(save_filename)
    T = noise.blade_T_distribution[:][0]
    Q = noise.blade_Q_distribution[:][0]
    r = noise.radius_distribution
           
    axes = fig.add_subplot(2,1,1)
    axes.plot(r, T , 'bo-')
    axes.set_ylabel('T (N)',axis_font)
    axes.set_xlabel('r (m)',axis_font)
    axes.get_yaxis().get_major_formatter().set_scientific(False)
    axes.get_yaxis().get_major_formatter().set_useOffset(False)
    axes.grid(True)  
    
    axes = fig.add_subplot(2,1,2)
    axes.plot( r, Q, 'bo-')
    axes.set_ylabel('Q (Nm)',axis_font)
    axes.set_xlabel('r (m)',axis_font)
    axes.get_yaxis().get_major_formatter().set_scientific(False)
    axes.get_yaxis().get_major_formatter().set_useOffset(False)
    axes.grid(True)    
    
    if save_figure:
        plt.savefig(save_filename + ".png")  
        
    return