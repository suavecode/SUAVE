## @ingroup Plots-Performance
# Mission_Plots.py
#
# Created:  Mar 2020, M. Clarke
#           Apr 2020, M. Clarke
#           Sep 2020, M. Clarke
#           Apr 2021, M. Clarke
#           Dec 2021, S. Claridge

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units , Data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import plotly.graph_objects as go
import matplotlib.ticker as ticker

# ------------------------------------------------------------------
#   Altitude, SFC & Weight
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_altitude_sfc_weight(results, line_color = 'bo-', save_figure = False, save_filename = "Altitude_SFC_Weight" , file_type = ".png",
                             width=8, height=5):
    """This plots the altitude, speficic fuel comsumption and vehicle weight

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.condtions.
        freestream.altitude
        weights.total_mass
        weights.vehicle_mass_rate
        frames.body.thrust_force_vector

    Outputs:
    Plots

    Properties Used:
    N/A
    """
    axis_font = {'size':'14'}
    fig = plt.figure(save_filename)
    fig.set_size_inches(width, height)
    for segment in results.segments.values():
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        mass     = segment.conditions.weights.total_mass[:,0] / Units.lb
        altitude = segment.conditions.freestream.altitude[:,0] / Units.ft
        mdot     = segment.conditions.weights.vehicle_mass_rate[:,0]
        thrust   =  segment.conditions.frames.body.thrust_force_vector[:,0]
        sfc      = (mdot / Units.lb) / (thrust /Units.lbf) * Units.hr

        axes = plt.subplot(3,1,1)
        axes.plot( time , altitude , line_color)
        axes.set_ylabel('Altitude (ft)',axis_font)
        set_axes(axes)

        axes = plt.subplot(3,1,3)
        axes.plot( time , sfc , line_color )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('sfc (lb/lbf-hr)',axis_font)
        set_axes(axes)

        axes = plt.subplot(3,1,2)
        axes.plot( time , mass , 'ro-' )
        axes.set_ylabel('Weight (lb)',axis_font)
        set_axes(axes)

    plt.tight_layout()
    if save_figure:
        plt.savefig(save_filename + file_type)

    return

# ------------------------------------------------------------------
#   Aircraft Velocities
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_aircraft_velocities(results, line_color = 'bo-', save_figure = False, save_filename = "Aircraft_Velocities", file_type = ".png",
                             width=8, height=5):
    """This plots aircraft velocity, mach , true air speed

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.condtions.freestream.
        velocity
        density
        mach_number

    Outputs:
    Plots

    Properties Used:
    N/A
    """
    axis_font = {'size':'14'}
    fig = plt.figure(save_filename)
    fig.set_size_inches(width, height)
    for segment in results.segments.values():
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        velocity = segment.conditions.freestream.velocity[:,0]
        density  = segment.conditions.freestream.density[:,0]
        EAS      = velocity * np.sqrt(density/1.225)
        mach     = segment.conditions.freestream.mach_number[:,0]

        axes = plt.subplot(3,1,1)
        axes.plot( time , velocity / Units.kts, line_color)
        axes.set_ylabel('velocity (kts)',axis_font)
        set_axes(axes)

        axes = plt.subplot(3,1,2)
        axes.plot( time , EAS / Units.kts, line_color)
        axes.set_ylabel('Equivalent Airspeed',axis_font)
        set_axes(axes)

        axes = plt.subplot(3,1,3)
        axes.plot( time , mach , line_color)
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Mach',axis_font)
        set_axes(axes)

    plt.tight_layout()
    if save_figure:
        plt.savefig(save_filename + file_type)

    return

# ------------------------------------------------------------------
#   Disc and Power Loadings
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_disc_power_loading(results, line_color = 'bo-', save_figure = False, save_filename = "Disc_Power_Loading", file_type = ".png",
                            width=8, height=5):
    """This plots the propeller disc and power loadings

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.condtions.propulsion.
        disc_loadings
        power_loading

    Outputs:
    Plots

    Properties Used:
    N/A
    """
    axis_font = {'size':'14'}
    fig = plt.figure(save_filename)
    fig.set_size_inches(width, height)

    for i in range(len(results.segments)):
        time  = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        DL    = results.segments[i].conditions.propulsion.disc_loading
        PL    = results.segments[i].conditions.propulsion.power_loading

        axes = plt.subplot(2,1,1)
        axes.plot(time, DL, line_color)
        axes.set_ylabel('lift disc power N/m^2',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,1,2)
        axes.plot(time, PL, line_color )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('lift power loading (N/W)',axis_font)
        set_axes(axes)

    plt.tight_layout()
    if save_figure:
        plt.savefig(save_filename + file_type)

    return

# ------------------------------------------------------------------
#   Plot Fuel Use
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_fuel_use(results, line_color = 'bo-', save_figure = False, save_filename = "Aircraft_Fuel_Burnt", file_type = ".png",
                  width=8, height=5):
    """This plots aircraft fuel usage
    Assumptions:
    None
    Source:
    None
    Inputs:
    results.segments.condtions.
        frames.inertial.time
        weights.fuel_mass
        weights.additional_fuel_mass
        weights.total_mass
    Outputs:
    Plots
    Properties Used:
    N/A	"""

    axis_font = {'size':'14'}
    fig = plt.figure(save_filename)
    fig.set_size_inches(width, height)

    prev_seg_fuel       = 0
    prev_seg_extra_fuel = 0
    total_fuel          = 0

    axes = plt.subplot(1,1,1)

    for i in range(len(results.segments)):

        segment  = results.segments[i]
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min

        if "has_additional_fuel" in segment.conditions.weights and segment.conditions.weights.has_additional_fuel == True:


            fuel     = segment.conditions.weights.fuel_mass[:,0]
            alt_fuel = segment.conditions.weights.additional_fuel_mass[:,0]

            if i == 0:

                plot_fuel     = np.negative(fuel)
                plot_alt_fuel = np.negative(alt_fuel)

                axes.plot( time , plot_fuel , 'ro-' , label = 'fuel')
                axes.plot( time , plot_alt_fuel , 'bo-', label = 'additional fuel' )
                axes.plot( time , np.add(plot_fuel, plot_alt_fuel), 'go-', label = 'total fuel' )

                axes.legend(loc='center right')

            else:
                prev_seg_fuel       += results.segments[i-1].conditions.weights.fuel_mass[-1]
                prev_seg_extra_fuel += results.segments[i-1].conditions.weights.additional_fuel_mass[-1]

                current_fuel         = np.add(fuel, prev_seg_fuel)
                current_alt_fuel     = np.add(alt_fuel, prev_seg_extra_fuel)

                axes.plot( time , np.negative(current_fuel)  , 'ro-' )
                axes.plot( time , np.negative(current_alt_fuel ), 'bo-')
                axes.plot( time , np.negative(current_fuel + current_alt_fuel), 'go-')

        else:

            initial_weight  = results.segments[0].conditions.weights.total_mass[:,0][0]

            for i in range(len(results.segments) ) :
                segment     = results.segments[i]
                fuel        = segment.conditions.weights.total_mass[:,0]
                time        = segment.conditions.frames.inertial.time[:,0] / Units.min
                total_fuel  = np.negative(segment.conditions.weights.total_mass[:,0] - initial_weight )
                axes.plot( time, total_fuel, 'mo-')

    axes.set_ylabel('Fuel (kg)',axis_font)
    axes.set_xlabel('Time (min)',axis_font)

    set_axes(axes)


    plt.tight_layout()

    if save_figure:
        plt.savefig(save_filename + file_type)

    return
# ------------------------------------------------------------------
#   Aerodynamic Coefficients
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_aerodynamic_coefficients(results, line_color = 'bo-', save_figure = False, save_filename = "Aerodynamic_Coefficients", file_type = ".png",
                                  width=8, height=5):
    """This plots the aerodynamic coefficients

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.condtions.aerodynamics.
        lift_coefficient
        drag_coefficient
        angle_of_attack

    Outputs:
    Plots

    Properties Used:
    N/A
    """
    axis_font = {'size':'14'}
    fig = plt.figure(save_filename)
    fig.set_size_inches(width, height)

    for segment in results.segments.values():
        time = segment.conditions.frames.inertial.time[:,0] / Units.min
        cl   = segment.conditions.aerodynamics.lift_coefficient[:,0,None]
        cd   = segment.conditions.aerodynamics.drag_coefficient[:,0,None]
        aoa  = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        l_d  = cl/cd

        axes = plt.subplot(2,2,1)
        axes.plot( time , aoa , line_color )
        axes.set_ylabel('Angle of Attack (deg)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,2)
        axes.plot( time , cl, line_color )
        axes.set_ylabel('CL',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,3)
        axes.plot( time , cd, line_color )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('CD',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,4)
        axes.plot( time , l_d, line_color )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('L/D',axis_font)
        set_axes(axes)

    plt.tight_layout()
    if save_figure:
        plt.savefig(save_filename + file_type)

    return

# ------------------------------------------------------------------
#   Aerodynamic Forces
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_aerodynamic_forces(results, line_color = 'bo-', save_figure = False, save_filename = "Aerodynamic_Forces", file_type = ".png",
                            width=8, height=5):
    """This plots the aerodynamic forces

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.condtions.frames
         body.thrust_force_vector
         wind.lift_force_vector
         wind.drag_force_vector

    Outputs:
    Plots

    Properties Used:
    N/A
    """
    axis_font = {'size':'14'}
    fig = plt.figure(save_filename)
    fig.set_size_inches(width, height)

    for segment in results.segments.values():
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]
        Lift   = -segment.conditions.frames.wind.lift_force_vector[:,2]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        eta    = segment.conditions.propulsion.throttle[:,0]

        axes = plt.subplot(2,2,1)
        axes.plot( time , eta , line_color )
        axes.set_ylabel('Throttle',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,2)
        axes.plot( time , Lift , line_color)
        axes.set_ylabel('Lift (N)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,3)
        axes.plot( time , Thrust , line_color)
        axes.set_ylabel('Thrust (N)',axis_font)
        axes.set_xlabel('Time (min)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,4)
        axes.plot( time , Drag , line_color)
        axes.set_ylabel('Drag (N)',axis_font)
        axes.set_xlabel('Time (min)',axis_font)
        set_axes(axes)

    plt.tight_layout()
    if save_figure:
        plt.savefig(save_filename + file_type)

    return

# ------------------------------------------------------------------
#   Drag Components
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_drag_components(results, line_color = 'bo-', save_figure = False, save_filename = "Drag_Components", file_type = ".png",
                         width=8, height=5):
    """This plots the drag components of the aircraft

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.condtions.aerodynamics.drag_breakdown
          parasite.total
          induced.total
          compressible.total
          miscellaneous.total

    Outputs:
    Plots

    Properties Used:
    N/A
    """
    axis_font = {'size':'14'}
    fig    = plt.figure(save_filename)
    fig.set_size_inches(width, height)
    axes = plt.subplot(1,1,1)

    for i, segment in enumerate(results.segments.values()):
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        drag_breakdown = segment.conditions.aerodynamics.drag_breakdown
        cdp = drag_breakdown.parasite.total[:,0]
        cdi = drag_breakdown.induced.total[:,0]
        cdc = drag_breakdown.compressible.total[:,0]
        cdm = drag_breakdown.miscellaneous.total[:,0]
        cd  = drag_breakdown.total[:,0]

        if i == 0:
            axes.plot( time , cdp , 'ko-', label='CD parasite' )
            axes.plot( time , cdi , 'bo-', label='CD induced' )
            axes.plot( time , cdc , 'go-', label='CD compressibility' )
            axes.plot( time , cdm , 'yo-', label='CD miscellaneous' )
            axes.plot( time , cd  , 'ro-', label='CD total'   )
            axes.legend(loc='upper center')
        else:
            axes.plot( time , cdp , 'ko-' )
            axes.plot( time , cdi , 'bo-')
            axes.plot( time , cdc , 'go-')
            axes.plot( time , cdm , 'yo-')
            axes.plot( time , cd  , 'ro-')

    axes.set_xlabel('Time (min)',axis_font)
    axes.set_ylabel('CD',axis_font)
    axes.grid(True)

    plt.tight_layout()
    if save_figure:
        plt.savefig(save_filename + file_type)

    return


# ------------------------------------------------------------------
#   Electronic Conditions
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_battery_pack_conditions(results, line_color = 'bo-', line_color2 = 'rs--', save_figure = False, save_filename = "Battery_Pack_Conditions", file_type = ".png",
                                 width=8, height=5):
    """This plots the battery pack conditions of the network

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.conditions.propulsion
         battery_power_draw
         battery_energy
         battery_voltage_under_load
         battery_voltage_open_circuit
         current

    Outputs:
    Plots

    Properties Used:
    N/A
    """

    axis_font = {'size':'14'}

    fig = plt.figure(save_filename)
    fig.set_size_inches(width, height)
    fig.suptitle('Battery Pack Conditions')
    for i in range(len(results.segments)):
        time                = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        pack_power          = results.segments[i].conditions.propulsion.battery_power_draw[:,0]
        pack_energy         = results.segments[i].conditions.propulsion.battery_energy[:,0]
        pack_volts          = results.segments[i].conditions.propulsion.battery_voltage_under_load[:,0]
        pack_volts_oc       = results.segments[i].conditions.propulsion.battery_voltage_open_circuit[:,0]
        pack_current        = results.segments[i].conditions.propulsion.battery_current[:,0]
        pack_SOC            = results.segments[i].conditions.propulsion.battery_state_of_charge[:,0]
        pack_current        = results.segments[i].conditions.propulsion.battery_current[:,0]

        pack_battery_amp_hr = (pack_energy/ Units.Wh )/pack_volts
        pack_C_instant      = pack_current/pack_battery_amp_hr
        pack_C_nominal      = pack_current/np.max(pack_battery_amp_hr)


        axes = plt.subplot(2,3,1)
        axes.plot(time, pack_SOC , line_color)
        axes.set_ylabel('SOC',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,3,2)
        axes.plot(time, (pack_energy/Units.Wh)/1000, line_color)
        axes.set_ylabel('Energy (kW-hr)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,3,3)
        axes.plot(time, -pack_power/1000, line_color)
        axes.set_ylabel('Power (kW)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,3,4)
        axes.set_ylabel('Voltage (V)',axis_font)
        axes.set_xlabel('Time (mins)',axis_font)
        set_axes(axes)
        if i == 0:
            axes.plot(time, pack_volts, line_color,label='Under Load')
            axes.plot(time,pack_volts_oc, line_color2,label='Open Circuit')
        else:
            axes.plot(time, pack_volts, line_color)
            axes.plot(time,pack_volts_oc,line_color2)
        axes.legend(loc='best')

        axes = plt.subplot(2,3,5)
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('C-Rate (C)',axis_font)
        set_axes(axes)
        if i == 0:
            axes.plot(time, pack_C_instant, line_color,label='Instantaneous')
            axes.plot(time, pack_C_nominal, line_color2,label='Nominal')
        else:
            axes.plot(time, pack_C_instant, line_color)
            axes.plot(time, pack_C_nominal, line_color2)
        axes.legend(loc='best')

        axes = plt.subplot(2,3,6)
        axes.plot(time, pack_current, line_color)
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Current (A)',axis_font)
        set_axes(axes)


    # Set limits
    for i in range(1,7):
        ax         = plt.subplot(2,3,i)
        y_lo, y_hi = ax.get_ylim()
        if y_lo>0: y_lo = 0
        y_hi       = y_hi*1.1
        ax.set_ylim(y_lo,y_hi)


    plt.tight_layout()
    if save_figure:
        fig.savefig(save_filename + file_type)

    return

# ------------------------------------------------------------------
#   Electronic Conditions
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_battery_cell_conditions(results, line_color = 'bo-',line_color2 = 'rs--', save_figure = False, save_filename = "Battery_Cell_Conditions", file_type = ".png",
                                 width=8, height=5):
    """This plots the battery pack conditions of the network

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.conditions.propulsion
         battery_power_draw
         battery_energy
         voltage_under_load
         voltage_open_circuit
         current

    Outputs:
    Plots

    Properties Used:
    N/A
    """

    axis_font = {'size':'14'}

    fig  = plt.figure(save_filename)
    fig.set_size_inches(width, height)
    fig.suptitle('Battery Cell Conditions')
    for i in range(len(results.segments)):
        time                = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        cell_power          = results.segments[i].conditions.propulsion.battery_cell_power_draw[:,0]
        cell_energy         = results.segments[i].conditions.propulsion.battery_cell_energy[:,0]
        cell_volts          = results.segments[i].conditions.propulsion.battery_cell_voltage_under_load[:,0]
        cell_volts_oc       = results.segments[i].conditions.propulsion.battery_cell_voltage_open_circuit[:,0]
        cell_current        = results.segments[i].conditions.propulsion.battery_cell_current[:,0]
        cell_SOC            = results.segments[i].conditions.propulsion.battery_state_of_charge[:,0]
        cell_temp           = results.segments[i].conditions.propulsion.battery_cell_temperature[:,0]
        cell_charge         = results.segments[i].conditions.propulsion.battery_cell_charge_throughput[:,0]
        cell_current        = results.segments[i].conditions.propulsion.battery_cell_current[:,0]
        cell_battery_amp_hr = (cell_energy/ Units.Wh )/cell_volts

        cell_battery_amp_hr = (cell_energy/ Units.Wh )/cell_volts
        cell_C_instant      = cell_current/cell_battery_amp_hr
        cell_C_nominal      = cell_current/np.max(cell_battery_amp_hr)


        axes = plt.subplot(3,3,1)
        axes.plot(time, cell_SOC, line_color)
        axes.set_ylabel('SOC',axis_font)
        set_axes(axes)

        axes = plt.subplot(3,3,2)
        axes.plot(time, (cell_energy/Units.Wh), line_color)
        axes.set_ylabel('Energy (W-hr)',axis_font)
        set_axes(axes)


        axes = plt.subplot(3,3,3)
        axes.plot(time, -cell_power, line_color)
        axes.set_ylabel('Power (W)',axis_font)
        set_axes(axes)


        axes = plt.subplot(3,3,4)
        axes.set_ylabel('Voltage (V)',axis_font)
        set_axes(axes)
        if i == 0:
            axes.plot(time, cell_volts, line_color,label='Under Load')
            axes.plot(time,cell_volts_oc, line_color2,label='Open Circuit')
            axes.legend(loc='upper right')
        else:
            axes.plot(time, cell_volts, line_color)
            axes.plot(time,cell_volts_oc, line_color2)

        axes = plt.subplot(3,3,5)
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('C-Rate (C)',axis_font)
        set_axes(axes)
        if i == 0:
            axes.plot(time, cell_C_instant, line_color,label='Instantaneous')
            axes.plot(time, cell_C_nominal, line_color2,label='Nominal')
            axes.legend(loc='upper right')
        else:
            axes.plot(time, cell_C_instant, line_color)
            axes.plot(time, cell_C_nominal, line_color2)


        axes = plt.subplot(3,3,6)
        axes.plot(time, cell_charge, line_color)
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Current (A)',axis_font)
        set_axes(axes)

        axes = plt.subplot(3,3,7)
        axes.plot(time, cell_charge, line_color)
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Charge Throughput (Ah)',axis_font)
        set_axes(axes)

        axes = plt.subplot(3,3,8)
        axes.plot(time, cell_temp, line_color)
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Temperature (K)',axis_font)
        set_axes(axes)


    # Set limits
    for i in range(1,9):
        ax         = plt.subplot(3,3,i)
        y_lo, y_hi = ax.get_ylim()
        if y_lo>0: y_lo = 0
        y_hi       = y_hi*1.1
        ax.set_ylim(y_lo,y_hi)

    plt.tight_layout()
    if save_figure:
        fig.savefig(save_filename + file_type)

    return

# ------------------------------------------------------------------
#   Battery Degradation
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_battery_degradation(results, line_color = 'bo-',line_color2 = 'rs--', save_figure = False, save_filename = "Battery_Cell_Conditions", file_type = ".png",
                             width=8, height=5):
    """This plots the battery cell degradation

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.conditions.propulsion
        battery_cycle_day                     [unitless]
        battery_capacity_fade_factor          [-]
        battery_resistance_growth_factor      [-]
        battery_cell_charge_throughput        [Ah]

    Outputs:
    Plots

    Properties Used:
    N/A
    """

    axis_font = {'size':'14'}

    fig  = plt.figure(save_filename)
    fig.set_size_inches(width, height)
    fig.suptitle('Battery Cell Degradation')

    num_segs          = len(results.segments)
    time_hrs          = np.zeros(num_segs)
    capacity_fade     = np.zeros_like(time_hrs)
    resistance_growth = np.zeros_like(time_hrs)
    cycle_day         = np.zeros_like(time_hrs)
    charge_throughput = np.zeros_like(time_hrs)

    for i in range(num_segs):
        time_hrs[i]            = results.segments[i].conditions.frames.inertial.time[-1,0] / Units.hour
        cycle_day[i]           = results.segments[i].conditions.propulsion.battery_cycle_day
        capacity_fade[i]       = results.segments[i].conditions.propulsion.battery_capacity_fade_factor
        resistance_growth[i]   = results.segments[i].conditions.propulsion.battery_resistance_growth_factor
        charge_throughput[i]   =  results.segments[i].conditions.propulsion.battery_cell_charge_throughput[-1,0]

    axes = plt.subplot(2,2,1)
    axes.plot(charge_throughput, capacity_fade, line_color)
    axes.plot(charge_throughput, resistance_growth, line_color2)
    axes.set_ylabel('% Capacity Fade/Resistance Growth',axis_font)
    axes.set_xlabel('Time (hrs)',axis_font)
    set_axes(axes)

    axes = plt.subplot(2,2,2)
    axes.plot(time_hrs, capacity_fade, line_color)
    axes.plot(time_hrs, resistance_growth, line_color2)
    axes.set_ylabel('% Capacity Fade/Resistance Growth',axis_font)
    axes.set_xlabel('Time (hrs)',axis_font)
    set_axes(axes)

    axes = plt.subplot(2,2,3)
    axes.plot(cycle_day, capacity_fade, line_color)
    axes.plot(cycle_day, resistance_growth, line_color2)
    axes.set_ylabel('% Capacity Fade/Resistance Growth',axis_font)
    axes.set_xlabel('Time (days)',axis_font)
    set_axes(axes)

    plt.tight_layout()
    if save_figure:
        fig.savefig(save_filename + file_type)

    return


# ------------------------------------------------------------------
#   Flight Conditions
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_flight_conditions(results, line_color = 'bo-', save_figure = False, save_filename = "Flight_Conditions", file_type = ".png",
                           width=8, height=5):
    """This plots the flights the conditions

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.conditions.
         frames
             body.inertial_rotations
             inertial.position_vector
         freestream.velocity
         aerodynamics.
             lift_coefficient
             drag_coefficient
             angle_of_attack

    Outputs:
    Plots

    Properties Used:
    N/A
    """
    axis_font = {'size':'14'}
    fig = plt.figure(save_filename)
    fig.set_size_inches(width, height)

    for segment in results.segments.values():
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        airspeed = segment.conditions.freestream.velocity[:,0] /   Units['mph']
        theta    = segment.conditions.frames.body.inertial_rotations[:,1,None] / Units.deg
        Range    = segment.conditions.frames.inertial.aircraft_range[:,0]/ Units.nmi
        altitude = segment.conditions.freestream.altitude[:,0]/Units.feet

        axes = plt.subplot(2,2,1)
        axes.plot(time, altitude, line_color)
        axes.set_ylabel('Altitude (ft)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,2)
        axes.plot( time , airspeed , line_color )
        axes.set_ylabel('Airspeed (mph)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,3)
        axes.plot( time , theta, line_color )
        axes.set_ylabel('Pitch Angle (deg)',axis_font)
        axes.set_xlabel('Time (min)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,4)
        axes.plot( time , Range, 'bo-')
        axes.set_ylabel('Range (nmi)',axis_font)
        axes.set_xlabel('Time (min)',axis_font)
        set_axes(axes)

    plt.tight_layout()
    if save_figure:
        plt.savefig(save_filename + file_type)

    return

# ------------------------------------------------------------------
#  Aircraft Trajectory
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_flight_trajectory(results, line_color = 'bo-', line_color2 = 'rs--', save_figure = False, save_filename = "Flight_Trajectory", file_type = ".png",
                           width=8, height=5):
    """This plots the 3D flight trajectory of the aircraft.

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.conditions.
         frames
             body.inertial_rotations
             inertial.position_vector
         freestream.velocity
         aerodynamics.
             lift_coefficient
             drag_coefficient
             angle_of_attack

    Outputs:
    Plots

    Properties Used:
    N/A
    """
    axis_font = {'size':'14'}
    fig = plt.figure(save_filename)
    fig.set_size_inches(width, height)

    for segment in results.segments.values():
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        x        = segment.conditions.frames.inertial.position_vector[:,0]
        y        = segment.conditions.frames.inertial.position_vector[:,1]
        z        = -segment.conditions.frames.inertial.position_vector[:,2]

        axes = plt.subplot(2,2,1)
        axes.plot( time , x , line_color )
        axes.plot( time , y , line_color2 )
        axes.set_xlabel('Distance (m)',axis_font)
        axes.set_xlabel('Time (min)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,2)
        axes.plot(x, y , line_color)
        axes.set_xlabel('x (m)',axis_font)
        axes.set_ylabel('y (m)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,3)
        axes.plot( time , z, line_color )
        axes.set_ylabel('z (m)',axis_font)
        axes.set_xlabel('Time (min)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,4, projection='3d')
        axes.scatter(x, y, z, marker='o',color = 'k')
        axes.set_xlabel('x',axis_font)
        axes.set_ylabel('y',axis_font)
        axes.set_zlabel('z',axis_font)
        set_axes(axes)

    plt.tight_layout()
    if save_figure:
        plt.savefig(save_filename + file_type)

    return
# ------------------------------------------------------------------
#   Propulsion Conditions
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_propeller_conditions(results, line_color = 'bo-', save_figure = False, save_filename = "Propeller", file_type = ".png",
                              width=8, height=5):
    """This plots the propeller performance

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.conditions.
        frames.inertial.time
        propulsion.rpm
        frames.body.thrust_force_vector
        propulsion.propeller_motor_torque
        propulsion.propeller_tip_mach

    Outputs:
    Plots

    Properties Used:
    N/A
    """

    axis_font = {'size':'14'}
    fig = plt.figure(save_filename)
    fig.set_size_inches(width, height)

    for segment in results.segments.values():
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        rpm    = segment.conditions.propulsion.propeller_rpm[:,0]
        thrust = np.linalg.norm(segment.conditions.frames.body.thrust_force_vector[:,:],axis=1)
        torque = segment.conditions.propulsion.propeller_motor_torque[:,0]
        tm     = segment.conditions.propulsion.propeller_tip_mach[:,0]
        Cp     = segment.conditions.propulsion.propeller_power_coefficient[:,0]
        eta    = segment.conditions.propulsion.throttle[:,0]

        axes = plt.subplot(2,3,1)
        axes.plot(time, thrust, line_color)
        axes.set_ylabel('Thrust (N)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,3,2)
        axes.plot(time, rpm, line_color)
        axes.set_ylabel('RPM',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,3,3)
        axes.plot(time, torque, line_color )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Torque (N-m)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,3,4)
        axes.plot( time , eta , line_color )
        axes.set_ylabel('Throttle',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,3,5)
        axes.plot(time, Cp, line_color )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Power Coefficient',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,3,6)
        axes.plot(time, tm, line_color )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Tip Mach',axis_font)
        set_axes(axes)

    # Set limits
    for i in range(1,7):
        ax         = plt.subplot(2,3,i)
        y_lo, y_hi = ax.get_ylim()
        if y_lo>0: y_lo = 0
        y_hi       = y_hi*1.1
        ax.set_ylim(y_lo,y_hi)

    plt.tight_layout()
    if save_figure:
        plt.savefig(save_filename + file_type)

    return

## @ingroup Plots-Performance
def plot_tiltrotor_conditions(results,configs,line_color='bo-',save_figure=False, save_filename="Tiltrotor", file_type=".png",
                              width=8, height=5):
    """This plots the tiltrotor conditions

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.conditions.
        frames.inertial.time
        propulsion.propeller_y_axis_rotation

    Outputs:
    Plots

    Properties Used:
    N/A
    """

    axis_font = {'size':'14'}
    fig = plt.figure(save_filename)
    fig.set_size_inches(width, height)

    config = configs[list(configs.keys())[0]]
    net    = config.networks[list(config.networks.keys())[0]]
    props  = net.propellers
    D      = 2 * props[list(props.keys())[0]].tip_radius

    for s, segment in enumerate(results.segments.values()):

        Vx      = segment.state.conditions.frames.inertial.velocity_vector[:,0]
        Vz      = segment.state.conditions.frames.inertial.velocity_vector[:,2]

        body_angle = segment.state.conditions.frames.body.inertial_rotations[:,1] / Units.deg
        y_rot      = segment.conditions.propulsion.propeller_y_axis_rotation[:,0] / Units.deg
        time       = segment.conditions.frames.inertial.time[:,0] / Units.min
        Vinf       = segment.conditions.freestream.velocity[:,0]

        thrust_vector = segment.conditions.frames.body.thrust_force_vector
        Tx = thrust_vector[:,0]
        Tz = thrust_vector[:,2]
        thrust_angle  = np.arccos(Tx / np.sqrt(Tx**2 + Tz**2))
        velocity_angle = np.arctan(-Vz / Vx)

        n     = segment.conditions.propulsion.propeller_rpm[:,0] / 60
        J     = Vinf/(n*D)

        prop_incidence_angles =  thrust_angle - velocity_angle

        axes = plt.subplot(2,2,1)
        axes.plot(time, y_rot, line_color)
        axes.set_xlabel('Time (mins)', axis_font)
        axes.set_ylabel('Network Y-Axis Rotation (deg)', axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,2)
        axes.plot(time, body_angle, line_color)
        axes.set_xlabel('Time (mins)', axis_font)
        axes.set_ylabel('Aircraft Pitch', axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,3)
        axes.plot(time, J, line_color)
        axes.set_xlabel('Time (mins)', axis_font)
        axes.set_ylabel('Advance Ratio (J=V/nD)', axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,4)
        axes.plot(time, prop_incidence_angles/Units.deg, line_color)
        axes.set_xlabel('Time (mins)', axis_font)
        axes.set_ylabel('Propeller Incidence', axis_font)
        set_axes(axes)
    plt.tight_layout()
    if save_figure:
        plt.savefig(save_filename + file_type)


    fig2 = plt.figure("Rotor Operation")
    fig2.set_size_inches(width, height)
    #marks = ['bs', 'oo', 'go', 'r^', 'ms','k-','ro','gs','yo']
    for s, segment in enumerate(results.segments.values()):

        Vx      = segment.state.conditions.frames.inertial.velocity_vector[:,0]
        Vz      = segment.state.conditions.frames.inertial.velocity_vector[:,2]
        Vinf    = segment.conditions.freestream.velocity[:,0]

        thrust_vector = segment.conditions.frames.body.thrust_force_vector
        Tx = thrust_vector[:,0]
        Tz = thrust_vector[:,2]
        thrust_angle  = np.arccos(Tx / np.sqrt(Tx**2 + Tz**2))
        velocity_angle = np.arctan(-Vz / Vx)

        n     = segment.conditions.propulsion.propeller_rpm[:,0] / 60
        J     = Vinf/(n*D)

        prop_incidence_angles =  thrust_angle - velocity_angle

        axes = plt.subplot(1,1,1)
        axes.scatter(prop_incidence_angles/Units.deg, J, label=segment.tag)
        axes.set_xlabel("Propeller Incidence Angle [deg]")
        axes.set_ylabel("Advance Ratio, J=V/nD")
        set_axes(axes)

    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.tight_layout()

    return

# ------------------------------------------------------------------
#   Electric Propulsion efficiencies
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_eMotor_Prop_efficiencies(results, line_color = 'bo-', save_figure = False, save_filename = "eMotor_Prop_Propulsor", file_type = ".png",
                                  width=8, height=5):
    """This plots the electric driven network propeller efficiencies

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.conditions.propulsion.
         etap
         etam

    Outputs:
    Plots

    Properties Used:
    N/A
    """
    axis_font = {'size':'14'}
    fig = plt.figure(save_filename)
    fig.set_size_inches(width, height)
    for s,segment in enumerate(results.segments.values()):
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        effp   = segment.conditions.propulsion.propeller_efficiency[:,0]
        fom    = segment.conditions.propulsion.figure_of_merit[:,0]
        effm   = segment.conditions.propulsion.propeller_motor_efficiency[:,0]

        axes = plt.subplot(1,2,1)
        axes.plot(time, effp, line_color, label=r'$\eta_p$')
        axes.plot(time, fom, 'ro-', label='FoM')
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel(r'Propeller Efficiency ($\eta_p$)',axis_font)

        set_axes(axes)
        plt.ylim((0,1))
        if s==0:
            plt.legend()

        axes = plt.subplot(1,2,2)
        axes.plot(time, effm, line_color )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel(r'Motor Efficiency ($\eta_m$)',axis_font)
        set_axes(axes)
        plt.ylim((0,1))

    plt.tight_layout()
    if save_figure:
        plt.savefig(save_filename + file_type)

    return

# ------------------------------------------------------------------
#   Stability Coefficients
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_stability_coefficients(results, line_color = 'bo-', save_figure = False, save_filename = "Stability_Coefficients", file_type = ".png",
                                width=8, height=5):
    """This plots the static stability characteristics of an aircraft

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.conditions.stability.
       static
           CM
           Cm_alpha
           static_margin
       aerodynamics.
           angle_of_attack
    Outputs:
    Plots

    Properties Used:
    N/A
    """
    axis_font = {'size':'14'}
    fig = plt.figure(save_filename)
    fig.set_size_inches(width, height)

    for segment in results.segments.values():
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        cm       = segment.conditions.stability.static.CM[:,0]
        cm_alpha = segment.conditions.stability.static.Cm_alpha[:,0]
        SM       = segment.conditions.stability.static.static_margin[:,0]
        aoa      = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg

        axes = plt.subplot(2,2,1)
        axes.plot( time , aoa, line_color )
        axes.set_ylabel(r'$AoA$',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,2)
        axes.plot( time , cm, line_color )
        axes.set_ylabel(r'$C_M$',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,3)
        axes.plot( time , cm_alpha, line_color )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel(r'$C_M\alpha$',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,4)
        axes.plot( time , SM, line_color )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Static Margin (%)',axis_font)
        set_axes(axes)

    plt.tight_layout()
    if save_figure:
        plt.savefig(save_filename + file_type)

    return

# ------------------------------------------------------------------
#   Solar Flux
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_solar_flux(results, line_color = 'bo-', save_figure = False, save_filename = "Solar_Flux", file_type = ".png",
                    width=8, height=5):
    """This plots the solar flux and power train performance of an solar powered aircraft

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.conditions.propulsion
        solar_flux
        battery_power_draw
        battery_energy

    Outputs:
    Plots

    Properties Used:
    N/A
    """

    axis_font = {'size':'14'}
    fig       = plt.figure(save_filename)
    fig.set_size_inches(width, height)

    for segment in results.segments.values():
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        flux   = segment.conditions.propulsion.solar_flux[:,0]
        charge = segment.conditions.propulsion.battery_power_draw[:,0]
        energy = segment.conditions.propulsion.battery_energy[:,0] / Units.MJ

        axes = plt.subplot(3,1,1)
        axes.plot( time , flux , line_color )
        axes.set_ylabel('Solar Flux (W/m$^2$)',axis_font)
        set_axes(axes)

        axes = plt.subplot(3,1,2)
        axes.plot( time , charge , line_color )
        axes.set_ylabel('Charging Power (W)',axis_font)
        set_axes(axes)

        axes = plt.subplot(3,1,3)
        axes.plot( time , energy , line_color )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Battery Energy (MJ)',axis_font)
        set_axes(axes)

    plt.tight_layout()
    if save_figure:
        plt.savefig(save_filename + file_type)

    return

# ------------------------------------------------------------------
#   Lift-Cruise Network
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_lift_cruise_network(results, line_color = 'bo-',line_color2 = 'r^-', save_figure = False, save_filename = "Lift_Cruise_Network", file_type = ".png",
                             width=8, height=5):
    """This plots the electronic and propulsor performance of a vehicle with a lift cruise network

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.conditions.propulsion
         throttle
         lift_rotor_throttle
         battery_energy
         battery_specfic_power
         voltage_under_load
         voltage_open_circuit

    Outputs:
    Plots

    Properties Used:
    N/A
    """
    axis_font = {'size':'14'}
    # ------------------------------------------------------------------
    #   Electronic Conditions
    # ------------------------------------------------------------------
    fig = plt.figure("Lift_Cruise_Battery_Pack_Conditions")
    fig.set_size_inches(width, height)
    for i in range(len(results.segments)):
        time           = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        eta            = results.segments[i].conditions.propulsion.throttle[:,0]
        eta_l          = results.segments[i].conditions.propulsion.throttle_lift[:,0]
        energy         = results.segments[i].conditions.propulsion.battery_energy[:,0]/ Units.Wh
        specific_power = results.segments[i].conditions.propulsion.battery_specfic_power[:,0]
        volts          = results.segments[i].conditions.propulsion.battery_voltage_under_load[:,0]
        volts_oc       = results.segments[i].conditions.propulsion.battery_voltage_open_circuit[:,0]

        plt.title('Battery Pack Conditions')
        axes = plt.subplot(2,2,1)
        axes.set_ylabel('Throttle',axis_font)
        set_axes(axes)
        plt.ylim((0,1))
        if i == 0:
            axes.plot(time, eta, line_color,label='Propeller Motor')
            axes.plot(time, eta_l, line_color2,label='Lift Rotor Motor')
            axes.legend(loc='upper center')
        else:
            axes.plot(time, eta, line_color)
            axes.plot(time, eta_l, line_color2)

        axes = plt.subplot(2,2,2)
        axes.plot(time, energy, line_color)
        axes.set_ylabel('Battery Energy (W-hr)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,2,3)
        axes.set_ylabel('Battery Voltage (Volts)',axis_font)
        axes.set_xlabel('Time (mins)',axis_font)
        set_axes(axes)
        if i == 0:
            axes.plot(time, volts, line_color,label='Under Load')
            axes.plot(time,volts_oc, line_color2,label='Open Circuit')
            axes.legend(loc='upper center')
        else:
            axes.plot(time, volts, line_color)
            axes.plot(time,volts_oc,line_color2)

        axes = plt.subplot(2,2,4)
        axes.plot(time, specific_power, line_color)
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Specific Power',axis_font)
        set_axes(axes)

    plt.tight_layout()
    if save_figure:
        plt.savefig("Lift_Cruise_Battery_Pack_Conditions" + file_type)


    # ------------------------------------------------------------------
    #   Propulsion Conditions
    # ------------------------------------------------------------------
    fig = plt.figure("Prop-Rotor Network")
    fig.set_size_inches(width, height)
    for i in range(len(results.segments)):
        time         = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        prop_rpm     = results.segments[i].conditions.propulsion.propeller_rpm[:,0]
        prop_thrust  = results.segments[i].conditions.frames.body.thrust_force_vector[:,0]
        prop_torque  = results.segments[i].conditions.propulsion.propeller_motor_torque[:,0]
        prop_effp    = results.segments[i].conditions.propulsion.propeller_efficiency[:,0]
        prop_effm    = results.segments[i].conditions.propulsion.propeller_motor_efficiency[:,0]
        prop_Cp      = results.segments[i].conditions.propulsion.propeller_power_coefficient[:,0]
        lift_rotor_rpm    = results.segments[i].conditions.propulsion.lift_rotor_rpm[:,0]
        lift_rotor_thrust = -results.segments[i].conditions.frames.body.thrust_force_vector[:,2]
        lift_rotor_torque = results.segments[i].conditions.propulsion.lift_rotor_motor_torque[:,0]
        lift_rotor_effp   = results.segments[i].conditions.propulsion.lift_rotor_efficiency[:,0]
        lift_rotor_effm   = results.segments[i].conditions.propulsion.lift_rotor_motor_efficiency[:,0]
        lift_rotor_Cp     = results.segments[i].conditions.propulsion.lift_rotor_power_coefficient[:,0]

        # title
        plt.title("Prop-Rotor Network")

        # plots
        axes = plt.subplot(2,3,1)
        axes.plot(time, prop_rpm, line_color)
        axes.plot(time, lift_rotor_rpm, line_color2)
        axes.set_ylabel('RPM',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,3,2)
        axes.plot(time, prop_thrust,line_color)
        axes.plot(time, lift_rotor_thrust, line_color2)
        axes.set_ylabel('Thrust (N)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,3,3)
        axes.plot(time, prop_torque, line_color)
        axes.plot(time, lift_rotor_torque, line_color2)
        axes.set_ylabel('Torque (N-m)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,3,4)
        axes.plot(time, prop_effp, line_color )
        axes.plot(time, lift_rotor_effp, line_color2)
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel(r'Propeller Efficiency, $\eta_{propeller}$',axis_font)
        set_axes(axes)
        plt.ylim((0,1))

        axes = plt.subplot(2,3,5)
        axes.plot(time, prop_effm, line_color )
        axes.plot(time, lift_rotor_effm,line_color2)
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel(r'Motor Efficiency, $\eta_{motor}$',axis_font)
        set_axes(axes)
        plt.ylim((0,1))

        axes = plt.subplot(2,3,6)
        axes.plot(time, prop_Cp,line_color )
        axes.plot(time, lift_rotor_Cp, line_color2 )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Power Coefficient, $C_{P}$',axis_font)
        set_axes(axes)

    plt.tight_layout()
    if save_figure:
        plt.savefig("Propulsor_Network" + file_type)

    # ------------------------------------------------------------------
    #   Propulsion Conditions
    # ------------------------------------------------------------------
    fig = plt.figure("Lift_Rotor")
    fig.set_size_inches(width, height)
    for i in range(len(results.segments)):
        time   = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        rpm    = results.segments[i].conditions.propulsion.lift_rotor_rpm [:,0]
        thrust = results.segments[i].conditions.frames.body.thrust_force_vector[:,2]
        torque = results.segments[i].conditions.propulsion.lift_rotor_motor_torque
        effp   = results.segments[i].conditions.propulsion.lift_rotor_efficiency[:,0]
        effm   = results.segments[i].conditions.propulsion.lift_rotor_motor_efficiency[:,0]
        Cp     = results.segments[i].conditions.propulsion.lift_rotor_power_coefficient[:,0]

        # title
        plt.title("Lift Rotor")

        # plots
        axes = plt.subplot(2,3,1)
        axes.plot(time, rpm, line_color2)
        axes.set_ylabel('RPM',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,3,2)
        axes.plot(time, -thrust, line_color2)
        axes.set_ylabel('Thrust (N)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,3,3)
        axes.plot(time, torque, line_color2 )
        axes.set_ylabel('Torque (N-m)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,3,4)
        axes.plot(time, effp, line_color2,label= r'$\eta_{lift rotor}$' )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel(r'Propeller Efficiency, $\eta_{lift rotor}$',axis_font)
        set_axes(axes)
        plt.ylim((0,1))

        axes = plt.subplot(2,3,5)
        axes.plot(time, effm, line_color2 )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel(r'Motor Efficiency, $\eta_{mot}$',axis_font)
        set_axes(axes)
        plt.ylim((0,1))

        axes = plt.subplot(2,3,6)
        axes.plot(time, Cp , line_color2 )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Power Coefficient, $C_{P}$',axis_font)
        set_axes(axes)

    plt.tight_layout()
    if save_figure:
        plt.savefig("Lift_Rotor" + file_type)

    # ------------------------------------------------------------------
    #   Propulsion Conditions
    # ------------------------------------------------------------------
    fig = plt.figure("Propeller")
    fig.set_size_inches(width, height)
    for i in range(len(results.segments)):
        time   = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        rpm    = results.segments[i].conditions.propulsion.propeller_rpm [:,0]
        thrust = results.segments[i].conditions.frames.body.thrust_force_vector[:,0]
        torque = results.segments[i].conditions.propulsion.propeller_motor_torque[:,0]
        effp   = results.segments[i].conditions.propulsion.propeller_efficiency[:,0]
        effm   = results.segments[i].conditions.propulsion.propeller_motor_efficiency[:,0]
        Cp     = results.segments[i].conditions.propulsion.propeller_power_coefficient[:,0]

        # title
        plt.title("Propeller")

        # plots
        axes = plt.subplot(2,3,1)
        axes.plot(time, rpm,line_color)
        axes.set_ylabel('RPM')
        set_axes(axes)

        axes = plt.subplot(2,3,2)
        axes.plot(time, thrust,line_color)
        axes.set_ylabel('Thrust (N)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,3,3)
        axes.plot(time, torque, line_color)
        axes.set_ylabel('Torque (N-m)',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,3,4)
        axes.plot(time, effp,line_color)
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel(r'Propeller Efficiency $\eta_{propeller}$',axis_font)
        set_axes(axes)
        plt.ylim((0,1))

        axes = plt.subplot(2,3,5)
        axes.plot(time, effm,line_color )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel(r'Motor Efficiency $\eta_{motor}$',axis_font)
        set_axes(axes)

        axes = plt.subplot(2,3,6)
        axes.plot(time, Cp, line_color )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Power Coefficient',axis_font)
        set_axes(axes)

    plt.tight_layout()
    if save_figure:
        plt.savefig("Cruise_Propulsor" + file_type)

    # ------------------------------------------------------------------
    #   Propulsion Conditions
    # ------------------------------------------------------------------
    fig = plt.figure("Tip_Mach")
    for i in range(len(results.segments)):
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        rtm  = results.segments[i].conditions.propulsion.lift_rotor_tip_mach[:,0]
        ptm  = results.segments[i].conditions.propulsion.propeller_tip_mach[:,0]

        # title
        plt.title("Tip Mach Number")

        # plots
        axes = plt.subplot(1,1,1)
        axes.set_ylabel('Mach',axis_font)
        set_axes(axes)
        if i == 0:
            axes.plot(time, ptm, line_color,label='Propeller')
            axes.plot(time, rtm, line_color2,label='Lift Rotor')
            axes.legend(loc='upper center')
        else:
            axes.plot(time, ptm, line_color )
            axes.plot(time, rtm, line_color2 )

    plt.tight_layout()
    if save_figure:
        plt.savefig("Tip_Mach" + file_type)


    return

# ------------------------------------------------------------------
#   Pressure Coefficient
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_surface_pressure_contours(results,vehicle, save_figure = False, save_filename = "Surface_Pressure", file_type = ".png"):
    """This plots the surface pressure distrubtion at all control points
    on all lifting surfaces of the aircraft

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.aerodynamics.
        pressure_coefficient
    vehicle.vortex_distribution.
       n_cw
       n_sw
       n_w

    Outputs:
    Plots

    Properties Used:
    N/A
    """
    VD         = vehicle.vortex_distribution
    n_cw       = VD.n_cw
    n_cw       = VD.n_cw
    n_sw       = VD.n_sw
    n_w        = VD.n_w
    b_pts      = np.concatenate(([0],np.cumsum(VD.n_sw*VD.n_cw)))

    # Create a boolean for not plotting vertical wings
    idx        = 0
    plot_flag  = np.ones(n_w)
    for wing in vehicle.wings:
        if wing.vertical:
            plot_flag[idx] = 0
            idx += 1
        else:
            idx += 1
        if wing.vertical and wing.symmetric:
            plot_flag[idx] = 0
            idx += 1
        else:
            idx += 1

    img_idx    = 1
    seg_idx    = 1
    for segment in results.segments.values():
        num_ctrl_pts = len(segment.conditions.frames.inertial.time)
        for ti in range(num_ctrl_pts):
            CP         = segment.conditions.aerodynamics.pressure_coefficient[ti]

            fig        = plt.figure()
            axes       = plt.subplot(1, 1, 1)
            x_max      = max(VD.XC) + 2
            y_max      = max(VD.YC) + 2
            axes.set_ylim(x_max, 0)
            axes.set_xlim(-y_max, y_max)
            fig.set_size_inches(8,8)
            for i in range(n_w):
                n_pts     = (n_sw[i] + 1) * (n_cw[i]+ 1)
                xc_pts    = VD.X[i*(n_pts):(i+1)*(n_pts)]
                x_pts     = np.reshape(np.atleast_2d(VD.XC[b_pts[i]:b_pts[i+1]]).T, (n_sw[i],-1))
                y_pts     = np.reshape(np.atleast_2d(VD.YC[b_pts[i]:b_pts[i+1]]).T, (n_sw[i],-1))
                z_pts     = np.reshape(np.atleast_2d(CP[b_pts[i]:b_pts[i+1]]).T, (n_sw[i],-1))
                x_pts_p   = x_pts*((n_cw[i]+1)/n_cw[i]) - x_pts[0,0]*((n_cw[i]+1)/n_cw[i])  +  xc_pts[0]
                points    = np.linspace(0.001,1,50)
                A         = np.cumsum(np.sin(np.pi/2*points))
                levals    = -(np.concatenate([-A[::-1],A[1:]])/(2*A[-1])  + A[-1]/(2*A[-1]) )[::-1]*0.015
                color_map = plt.cm.get_cmap('jet')
                rev_cm    = color_map.reversed()
                if plot_flag[i] == 1:
                    CS  = axes.contourf(y_pts,x_pts_p, z_pts, cmap = rev_cm,levels=levals,extend='both')

            # Set Color bar
            cbar = fig.colorbar(CS, ax=axes)
            cbar.ax.set_ylabel('$C_{P}$', rotation =  0)
            plt.axis('off')
            plt.grid(None)

            if save_figure:
                plt.savefig( save_filename + '_' + str(img_idx) + file_type)
            img_idx += 1
        seg_idx +=1

    return


# ------------------------------------------------------------------
#   Sectional Lift Distribution
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_lift_distribution(results,vehicle, save_figure = False, save_filename = "Sectional_Lift", file_type = ".png"):
    """This plots the sectional lift distrubtion at all control points
    on all lifting surfaces of the aircraft

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.aerodynamics.
        inviscid_wings_sectional_lift
    vehicle.vortex_distribution.
       n_sw
       n_w

    Outputs:
    Plots

    Properties Used:
    N/A
    """
    VD         = vehicle.vortex_distribution
    n_w        = VD.n_w
    b_sw       = np.concatenate(([0],np.cumsum(VD.n_sw)))

    axis_font  = {'size':'12'}
    img_idx    = 1
    seg_idx    = 1
    for segment in results.segments.values():
        num_ctrl_pts = len(segment.conditions.frames.inertial.time)
        for ti in range(num_ctrl_pts):
            cl_y = segment.conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional[ti]
            line = ['-b','-b','-r','-r','-k']
            fig  = plt.figure()
            fig.set_size_inches(8,8)
            axes = plt.subplot(1,1,1)
            for i in range(n_w):
                y_pts = VD.Y_SW[b_sw[i]:b_sw[i+1]]
                z_pts = cl_y[b_sw[i]:b_sw[i+1]]
                axes.plot(y_pts, z_pts, line[i] )
            axes.set_xlabel("Spanwise Location (m)",axis_font)
            axes.set_title('$C_{Ly}$',axis_font)

            if save_figure:
                plt.savefig( save_filename + '_' + str(img_idx) + file_type)
            img_idx += 1
        seg_idx +=1

    return

# ------------------------------------------------------------------
#   VLM Video
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def create_video_frames(results,vehicle, save_figure = True ,flight_profile = True,  save_filename = "Flight_Mission_Frame", file_type = ".png"):
    """This creates video frames of the aerodynamic conditions of the vehicle as well as the
    surface pressure coefficient throughout a mission

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.
       aerodynamics.
          lift_coefficient
          drag_coefficient
       conditions.
           freestream.altitude
           weights.total_mass

    vehicle.vortex_distribution.
       n_cp
       n_cw
       n_sw
       n_w
       n_fus

    Outputs:
    Plots

    Properties Used:
    N/A
    """
    VD         = vehicle.vortex_distribution
    n_cw       = VD.n_cw
    n_sw       = VD.n_sw
    n_w        = VD.n_w
    n_fus      = VD.n_fus
    b_pts      = np.concatenate(([0],np.cumsum(VD.n_sw*VD.n_cw)))

    # Create a boolean for not plotting vertical wings
    idx        = 0
    plot_flag  = np.ones(n_w)
    for wing in vehicle.wings:
        if wing.vertical:
            plot_flag[idx] = 0
            idx += 1
        else:
            idx += 1
        if wing.vertical and wing.symmetric:
            plot_flag[idx] = 0
            idx += 1
        else:
            idx += 1

    axis_font  = {'size':'16'}
    img_idx    = 1
    seg_idx    = 1
    for segment in results.segments.values():
        num_ctrl_pts = len(segment.conditions.frames.inertial.time)
        for ti in range(num_ctrl_pts):
            CP         = segment.conditions.aerodynamics.pressure_coefficient[ti]
            fig        = plt.figure(constrained_layout=True)
            fig.set_size_inches(12, 6.75)
            gs         = fig.add_gridspec(4, 4)
            axes       = plt.subplot(gs[:, :-1])

            x_max = max(VD.XC) + 2
            y_max = max(VD.YC) + 2
            axes.set_ylim(x_max, -2)
            axes.set_xlim(-y_max, y_max)

            # plot wing CP distribution
            for i in range(n_w):
                n_pts     = (n_sw[i] + 1) * (n_cw[i]+ 1)
                xc_pts    = VD.X[i*(n_pts):(i+1)*(n_pts)]
                x_pts     = np.reshape(np.atleast_2d(VD.XC[b_pts[i]:b_pts[i+1]]).T, (n_sw[i],-1))
                y_pts     = np.reshape(np.atleast_2d(VD.YC[b_pts[i]:b_pts[i+1]]).T, (n_sw[i],-1))
                z_pts     = np.reshape(np.atleast_2d(CP[b_pts[i]:b_pts[i+1]]).T, (n_sw[i],-1))
                x_pts_p   = x_pts*((n_cw[i]+1)/n_cw[i]) - x_pts[0,0]*((n_cw[i]+1)/n_cw[i])  +  xc_pts[0]
                points    = np.linspace(0.001,1,50)
                A         = np.cumsum(np.sin(np.pi/2*points))
                levals    = -(np.concatenate([-A[::-1],A[1:]])/(2*A[-1])  + A[-1]/(2*A[-1]) )[::-1]*0.015
                color_map = plt.cm.get_cmap('jet')
                rev_cm    = color_map.reversed()
                if plot_flag[i] == 1:
                    CS    = axes.contourf( y_pts,x_pts_p, z_pts, cmap = rev_cm,levels=levals,extend='both')

            # Set Color bar
            sfmt = ticker.ScalarFormatter(useMathText=True)
            sfmt = ticker.FormatStrFormatter('%.3f')
            cbar = fig.colorbar(CS, ax=axes , format= sfmt )
            cbar.ax.set_ylabel('$C_{P}$', labelpad  = 20, rotation =  0, fontsize =16)

            # plot fuselage
            for i in range(n_fus):
                n_pts  = (n_sw + 1) * (n_cw + 1)
                j      = n_w + i
                x_pts  = np.reshape(np.atleast_2d(VD.X[j*(n_pts):(j+1)*(n_pts)]).T, (n_sw+1,n_cw+1))
                y_pts  = np.reshape(np.atleast_2d(VD.Y[j*(n_pts):(j+1)*(n_pts)]).T, (n_sw+1,n_cw+1))
                z_pts  = np.reshape(np.atleast_2d(VD.Z[j*(n_pts):(j+1)*(n_pts)]).T, (n_sw+1,n_cw+1))

            plt.axis('off')
            plt.grid(None)

            if flight_profile:
                time_vec      = np.empty(shape=[0,1])
                cl_vec        = np.empty(shape=[0,1])
                cd_vec        = np.empty(shape=[0,1])
                l_d_vec       = np.empty(shape=[0,1])
                altitude_vec  = np.empty(shape=[0,1])
                mass_vec      = np.empty(shape=[0,1])
                for seg_i in range(seg_idx):
                    if seg_i == seg_idx-1:
                        t_vals   = results.segments[seg_i].conditions.frames.inertial.time[0:ti+1] / Units.min
                        cl_vals  = results.segments[seg_i].conditions.aerodynamics.lift_coefficient[0:ti+1]
                        cd_vals  = results.segments[seg_i].conditions.aerodynamics.drag_coefficient[0:ti+1]
                        l_d_vals = cl_vals/cd_vals
                        alt_vals = results.segments[seg_i].conditions.freestream.altitude[0:ti+1] / Units.ft
                        m_vals   = results.segments[seg_i].conditions.weights.total_mass[0:ti+1] * 0.001

                    else:
                        t_vals   = results.segments[seg_i].conditions.frames.inertial.time / Units.min
                        cl_vals  = results.segments[seg_i].conditions.aerodynamics.lift_coefficient
                        cd_vals  = results.segments[seg_i].conditions.aerodynamics.drag_coefficient
                        l_d_vals = cl_vals/cd_vals
                        alt_vals = results.segments[seg_i].conditions.freestream.altitude / Units.ft
                        m_vals   = results.segments[seg_i].conditions.weights.total_mass * 0.001

                    time_vec      = np.append(time_vec     ,t_vals[:,0])
                    cl_vec        = np.append(cl_vec       ,cl_vals[:,0])
                    cd_vec        = np.append(cd_vec       ,cd_vals[:,0])
                    l_d_vec       = np.append(l_d_vec      , l_d_vals[:,0])
                    altitude_vec  = np.append(altitude_vec ,alt_vals[:,0])
                    mass_vec      = np.append(mass_vec     ,m_vals[:,0])

                mini_axes1 = plt.subplot(gs[0:1, -1])
                mini_axes1.plot(time_vec, altitude_vec , 'ko-')
                mini_axes1.set_ylabel('Altitude (ft)',axis_font)
                mini_axes1.set_xlim(-10,420)
                mini_axes1.set_ylim(0,36000)
                mini_axes1.grid(False)

                mini_axes2 = plt.subplot(gs[1:2, -1])
                mini_axes2.plot(time_vec, mass_vec , 'ro-' )
                mini_axes2.set_ylabel('Weight (tons)',axis_font)
                mini_axes2.grid(False)
                mini_axes2.set_xlim(-10,420)
                mini_axes2.set_ylim(60,80)

                mini_axes3 = plt.subplot(gs[2:3, -1])
                mini_axes3.plot( time_vec, cl_vec, 'bo-'  )
                mini_axes3.set_ylabel('$C_{L}$',axis_font)
                mini_axes3.set_xlim(-10,420)
                mini_axes3.set_ylim(0.3,0.9)
                mini_axes3.grid(False)

                mini_axes4 = plt.subplot(gs[3:4, -1])
                mini_axes4.plot(time_vec , l_d_vec ,'go-'  )
                mini_axes4.set_ylabel('L/D',axis_font)
                mini_axes4.set_xlabel('Time (mins)',axis_font)
                mini_axes4.set_xlim(-10,420)
                mini_axes4.set_ylim(15,20)
                mini_axes4.grid(False)

            if save_figure:
                plt.savefig(save_filename + '_' + str(img_idx) + file_type)
            img_idx += 1
        seg_idx +=1
# ------------------------------------------------------------------
#   Rotor/Propeller Acoustics
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def plot_ground_noise_levels(results, line_color = 'bo-', save_figure = False, save_filename = "Sideline Noise Levels"):
    """This plots the A-weighted Sound Pressure Level as a function of time at various aximuthal angles 
    on the ground
    
    Assumptions:
    None
    
    Source:
    None
    
    Inputs: 
    results.segments.conditions.
        frames.inertial.position_vector   - position vector of aircraft 
        noise.                            
            total_SPL_dBA                 - total SPL (dbA)
            total_microphone_locations    - microphone locations
            
    Outputs: 
    Plots
    
    Properties Used:
    N/A	
    """     
    
    noise_data = post_process_noise_data(results) 
    SPL        = noise_data.SPL_dBA_ground_mic      
    gm         = noise_data.SPL_dBA_ground_mic_loc    
    gm_x       = gm[:,:,0]
    gm_y       = gm[:,:,1]
    colors     = cm.jet(np.linspace(0, 1,noise_data.N_gm_y))   
    
    # figure parameters
    axis_font    = {'size':'14'} 
    fig          = plt.figure(save_filename)
    fig.set_size_inches(8, 8) 
    axes        = fig.add_subplot(1,1,1) 
      
    max_SPL = np.max(SPL,axis=0) 
    for k in range(noise_data.N_gm_y):
        axes.plot(gm_x[:,0], max_SPL[:,k], marker = 'o', color = colors[k], label= r'mic at y = ' + str(round(gm_y[0,k],1)) + r' m' )
    axes.set_ylabel('SPL (dBA)',axis_font)
    axes.set_xlabel('Range (m)',axis_font)
    set_axes(axes)
    axes.legend(loc='upper right')
    if save_figure:
        plt.savefig(save_filename + ".png")


    return

## @ingroup Plots-Performance 
def plot_flight_profile_noise_contours(results, line_color = 'bo-', save_figure = False, save_filename = "Noise_Contour",show_figure = True):
    """This plots two contour surface of the maximum A-weighted Sound Pressure Level in the defined computational domain. 
    The first contour is the that of radiated noise on level ground only while the second contains radiated noise on buildings
    as well as the aircraft trajectory.
    
    Assumptions:
    None
    
    Source:
    None
    
    Inputs: 
    results.segments.conditions.
        frames.inertial.position_vector   - position vector of aircraft 
        noise.                            
            total_SPL_dBA                 - total SPL (dbA)
            total_microphone_locations    - microphone locations
            
    Outputs: 
    Plots
    
    Properties Used:
    N/A	
    """    

    noise_data = post_process_noise_data(results)     

    SPL_contour_gm  = noise_data.SPL_dBA_ground_mic      
    SPL_contour_bm  = noise_data.SPL_dBA_building_mic    
    Aircraft_pos    = noise_data.aircraft_position       
    X               = noise_data.SPL_dBA_ground_mic_loc[:,:,0]  
    Y               = noise_data.SPL_dBA_ground_mic_loc[:,:,1]  
    Z               = noise_data.SPL_dBA_ground_mic_loc[:,:,2]  
     
    plot_data      = []  
    max_SPL_gm     = np.max(SPL_contour_gm,axis=0) 
    
    # ---------------------------------------------------------------------------
    # Level ground contour 
    # ---------------------------------------------------------------------------
    filename_1          = 'Level_Ground_' + save_filename
    fig                 = plt.figure(filename_1)
    fig.set_size_inches(10 ,10)
    min_SPL             = 35
    max_SPL             = 80
    levs                = np.linspace(min_SPL,max_SPL,25)
    axes                = fig.add_subplot(1,1,1) 
    CS                  = axes.contourf(X,Y,max_SPL_gm, levels  = levs, cmap=plt.cm.jet, extend='both')
    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel('SPL (dBA)', rotation =  90)
    axes.set_ylabel('Spanwise $x_{fp}$ (m)',labelpad = 15)
    axes.set_xlabel('Streamwise $x_{fp}$ (m)')

    # ---------------------------------------------------------------------------
    # Comprehensive contour including buildings
    # ---------------------------------------------------------------------------
    ground_contour   = contour_surface_slice(X,Y,Z,max_SPL_gm)
    plot_data.append(ground_contour)

    # Aircraft Trajectory
    aircraft_trajectory = go.Scatter3d(x=Aircraft_pos[:,0], y=Aircraft_pos[:,1], z=Aircraft_pos[:,2],
                                mode='markers',
                                marker=dict(size=6,color='black',opacity=0.8),
                                line=dict(color='black',width=2))
    plot_data.append(aircraft_trajectory)

    # Define Colorbar Bounds
    min_alt     = 0
    max_alt     = np.max(Aircraft_pos[:,2])

    # Adjust Plot Camera
    camera        = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=-1., y=-1., z=.25))
    building_loc  = results.segments[0].analyses.noise.settings.building_locations
    num_buildings = len( building_loc) 
    if num_buildings > 0:
        max_alt     = np.maximum(max_alt, max((np.array(building_loc))[:,2]))
        min_bm_SPL  = np.min(SPL_contour_bm)
        max_bm_SPL  = np.max(SPL_contour_bm)
        min_SPL     = np.minimum(min_bm_SPL,min_SPL)
        max_SPL     = np.maximum(max_bm_SPL,max_SPL)

        # Get SPL on Building Surfaces
        max_SPL_contour_bm       = np.max(SPL_contour_bm,axis=0)
        building_dimensions      = results.segments[0].analyses.noise.settings.building_dimensions
        N_x                      = results.segments[0].analyses.noise.settings.building_microphone_x_resolution
        bldg_mic_loc             = results.segments[0].analyses.noise.settings.building_microphone_locations
        N_y                      = results.segments[0].analyses.noise.settings.building_microphone_y_resolution
        N_z                      = results.segments[0].analyses.noise.settings.building_microphone_z_resolution
        num_mics_on_xz_surface   = N_x*N_z
        num_mics_on_yz_surface   = N_y*N_z
        num_mics_on_xy_surface   = N_x*N_y
        num_mics_per_building    = 2*(num_mics_on_xz_surface + num_mics_on_yz_surface) +  num_mics_on_xy_surface

        # get surfaces of buildings
        for bldg_idx in range(num_buildings):
                # front (y-z plane)
                side_1_start = bldg_idx*num_mics_per_building 
                side_1_end   = bldg_idx*num_mics_per_building + num_mics_on_yz_surface
                surf_1_x     = np.ones((N_y,N_z))*(building_loc[bldg_idx][0] - building_dimensions[bldg_idx][0]/2)
                surf_1_y     = bldg_mic_loc[side_1_start:side_1_end,1].reshape(N_y,N_z)
                surf_1_z     = bldg_mic_loc[side_1_start:side_1_end,2].reshape(N_y,N_z)
                SPL_vals_1   = max_SPL_contour_bm[side_1_start:side_1_end].reshape(N_y,N_z)
                bldg_surf_1  = contour_surface_slice(surf_1_x ,surf_1_y ,surf_1_z ,SPL_vals_1)
                plot_data.append(bldg_surf_1)    
    
                # right (x-z plane)
                side_2_start = side_1_end
                side_2_end   = side_2_start + num_mics_on_xz_surface
                surf_2_x     = bldg_mic_loc[side_2_start:side_2_end,0].reshape(N_x,N_z)    
                surf_2_y     = np.ones((N_x,N_z))*building_loc[bldg_idx][1] + building_dimensions[bldg_idx][1]/2
                surf_2_z     = bldg_mic_loc[side_2_start:side_2_end,2].reshape(N_x,N_z)
                SPL_vals_2   = max_SPL_contour_bm[side_2_start:side_2_end].reshape(N_x,N_z)
                bldg_surf_2  = contour_surface_slice(surf_2_x ,surf_2_y ,surf_2_z ,SPL_vals_2)
                plot_data.append(bldg_surf_2)     
                
                # back (y-z plane)
                side_3_start = side_2_end
                side_3_end   = side_3_start + num_mics_on_yz_surface
                surf_3_x     = np.ones((N_y,N_z))*(building_loc[bldg_idx][0] + building_dimensions[bldg_idx][0]/2)
                surf_3_y     = bldg_mic_loc[side_3_start:side_3_end,1].reshape(N_y,N_z)
                surf_3_z     = bldg_mic_loc[side_3_start:side_3_end,2].reshape(N_y,N_z)
                SPL_vals_3   = max_SPL_contour_bm[side_3_start:side_3_end].reshape(N_y,N_z)
                bldg_surf_3  = contour_surface_slice(surf_3_x ,surf_3_y ,surf_3_z ,SPL_vals_3)
                plot_data.append(bldg_surf_3)                          
                
                # left (x-z plane)
                side_4_start = side_3_end
                side_4_end   = side_4_start +  num_mics_on_xz_surface 
                surf_4_x     = bldg_mic_loc[side_4_start:side_4_end,0].reshape(N_x,N_z)
                surf_4_y     = np.ones((N_x,N_z))*(building_loc[bldg_idx][1] - building_dimensions[bldg_idx][1]/2)
                surf_4_z     = bldg_mic_loc[side_4_start:side_4_end,2].reshape(N_x,N_z)
                SPL_vals_4   = max_SPL_contour_bm[side_4_start:side_4_end].reshape(N_x,N_z)
                bldg_surf_4  = contour_surface_slice(surf_4_x ,surf_4_y ,surf_4_z ,SPL_vals_4)
                plot_data.append(bldg_surf_4) 
                
                # top (x-y plane)
                side_5_start = side_4_end 
                side_5_end   = (bldg_idx+1)*num_mics_per_building   
                surf_5_x     = bldg_mic_loc[side_5_start:side_5_end,0].reshape(N_x,N_y)
                surf_5_y     = bldg_mic_loc[side_5_start:side_5_end,1].reshape(N_x,N_y)
                surf_5_z     = np.ones((N_x,N_y))*(building_dimensions[bldg_idx][2])
                SPL_vals_5   = max_SPL_contour_bm[side_5_start:side_5_end].reshape(N_x,N_y)
                bldg_surf_5  = contour_surface_slice(surf_5_x ,surf_5_y ,surf_5_z ,SPL_vals_5)
                plot_data.append(bldg_surf_5)         
    
    fig = go.Figure(data=plot_data)
    fig.update_layout(
             title_text= 'Flight_Profile_' + save_filename, 
             title_x = 0.5,
             width   = 750,
             height  = 750,
             font_size=18,
             scene_zaxis_range=[min_alt,max_alt], 
             coloraxis=dict(colorscale='Jet',
                            colorbar_thickness=50,
                            colorbar_nticks=20,
                            colorbar_title_text = 'SPL (dBA)',
                            colorbar_tickfont_size=18,
                            colorbar_title_side="right",
                            colorbar_ypad=60,
                            colorbar_len= 0.75,
                            **colorax(min_SPL, max_SPL)),
             scene_camera=camera)
    if show_figure:
        fig.show()
    return


def post_process_noise_data(results): 

    # unpack 
    N_segs         = len(results.segments)
    N_ctrl_pts     = len(results.segments[0].conditions.frames.inertial.time[:,0]) 
    N_bm           = results.segments[0].conditions.noise.number_of_building_microphones 
    N_gm_x         = results.segments[0].analyses.noise.settings.microphone_x_resolution
    N_gm_y         = results.segments[0].analyses.noise.settings.microphone_y_resolution   
    dim_mat        = N_segs*N_ctrl_pts 
    SPL_contour_gm = np.ones((dim_mat,N_gm_x,N_gm_y))*30 
    SPL_contour_bm = np.ones((dim_mat,N_bm))*30
    Aircraft_pos   = np.zeros((dim_mat,3)) 
    Mic_pos_gm     = results.segments[0].conditions.noise.total_ground_microphone_locations[0].reshape(N_gm_x,N_gm_y,3) 
    
    for i in range(N_segs):  
        if  results.segments[i].battery_discharge == False:
            pass
        else:      
            S_gm_x = results.segments[i].analyses.noise.settings.microphone_x_stencil
            S_gm_y = results.segments[i].analyses.noise.settings.microphone_y_stencil
            S_locs = results.segments[i].conditions.noise.ground_microphone_stencil_locations
            for j in range(N_ctrl_pts):
                idx                    = i*N_ctrl_pts + j 
                Aircraft_pos[idx,0]    = results.segments[i].conditions.frames.inertial.position_vector[j,0] 
                Aircraft_pos[idx,1]    = results.segments[i].conditions.frames.inertial.position_vector[j,1] 
                Aircraft_pos[idx,2]    = -results.segments[i].conditions.frames.inertial.position_vector[j,2] 
                stencil_length         = S_gm_x*2 + 1
                stencil_width          = S_gm_y*2 + 1
                SPL_contour_gm[idx,int(S_locs[j,0]):int(S_locs[j,1]),int(S_locs[j,2]):int(S_locs[j,3])]  = results.segments[i].conditions.noise.total_SPL_dBA[j].reshape(stencil_length ,stencil_width )  
                if N_bm > 0:
                    SPL_contour_bm[idx,:]  = results.segments[i].conditions.noise.total_SPL_dBA[j,-N_bm:]  
    
    noise_data                        = Data()
    noise_data.SPL_dBA_ground_mic     = np.nan_to_num(SPL_contour_gm)
    noise_data.SPL_dBA_building_mic   = np.nan_to_num(SPL_contour_bm)
    noise_data.aircraft_position      = Aircraft_pos
    noise_data.SPL_dBA_ground_mic_loc = Mic_pos_gm 
    noise_data.N_gm_y                 = N_gm_y
    noise_data.N_gm_x                 = N_gm_x  
    
    return noise_data



def contour_surface_slice(x,y,z,values):
    return go.Surface(x=x,y=y,z=z,surfacecolor=values,coloraxis='coloraxis')

def colorax(vmin, vmax):
    return dict(cmin=vmin,
                cmax=vmax)

# ------------------------------------------------------------------
#   Set Axis Parameters
# ------------------------------------------------------------------
## @ingroup Plots-Performance
def set_axes(axes):
    """This sets the axis parameters for all plots

    Assumptions:
    None

    Source:
    None

    Inputs
    axes

    Outputs:
    axes

    Properties Used:
    N/A
    """

    axes.minorticks_on()
    axes.grid(which='major', linestyle='-', linewidth=0.5, color='grey')
    axes.grid(which='minor', linestyle=':', linewidth=0.5, color='grey')
    axes.grid(True)
    axes.get_yaxis().get_major_formatter().set_scientific(False)
    axes.get_yaxis().get_major_formatter().set_useOffset(False)

    return
