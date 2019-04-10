import SUAVE
from SUAVE.Core import Units
import numpy as np 
import pytest 

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt  
from matplotlib.testing.decorators import image_comparison 
from SUAVE.Plots.Mission_Plots import * 

def main():
    results = load_plt_data()
    
    # Compare Plot for  Aerodynamic Forces
    #@pytest.mark.mpl_image_compare 
    @image_comparison(baseline_images=['Aerodynamic_Forces'], extensions=['png'])
    plot_aerodynamic_forces(results)
    
    # Compare Plot for  Aerodynamic Coefficients 
    @image_comparison(baseline_images=['Aerodynamic_Coefficients'], extensions=['png'])
    plot_aerodynamic_coefficients(results)
    
    # Compare Plot for  Drag Components
    @image_comparison(baseline_images=['Drag_Components'], extensions=['png'])
    plot_drag_components(results)
    
    # Compare Plot for  Altitude, sfc, vehicle weight 
    @image_comparison(baseline_images=['Altitude_SFC_Weight'], extensions=['png'])
    plot_altitude_sfc_weight(results)
    
    # Compare Plot for Aircraft Velocities 
    @image_comparison(baseline_images=['Aircraft_Velocities'], extensions=['png'])
    plot_aircraft_velocities(results)      

    # Compare Plot for Flight Conditions 
    @image_comparison(baseline_images=['Flight_Conditions'], extensions=['png'])    
    plot_flight_conditions(results)
        
    
    # If we use --mpl, it should detect that the figure is wrong
    #code = subprocess.call('{0} -m pytest {1}'.format(sys.executable, 'SUAVE/regression/scripts/plots/baseline_images/Flight_Conditions.png'), shell=True)
    #assert (code == 0), 'Plot regression failed'   

    return 

    ## Test functions
#@pytest.mark.mpl_image_compare
#def test_plot_flight_conditions(results): 
    #fig  = plt.figure(figsize=(8,6),dpi=100)
    #ax   = fig.add_subplot(111)
    #sol  = set_up_solution()
    #item = {'frames': griddle.data.TimeSeries([sol]),
            #'axes': ax,
            #'field': 0,
            #'plot_type': 'line'}
    #line, = griddle.plot_item_frame(item,0)
    #assert type(line) == matplotlib.lines.Line2D
    #assert line.get_data()[0].shape == (100,)
    #return fig

#def test_read_data():
    #pass

#@image_comparison(baseline_images=['spines_axes_positions'], extensions=['png'])
#def test_spines_axes_positions():
    ## SF bug 2852168
    #fig = plt.figure()
    #x = np.linspace(0,2*np.pi,100)
    #y = 2*np.sin(x)
    #ax = fig.add_subplot(1,1,1)
    #ax.set_title('centered spines')
    #ax.plot(x,y)
    #ax.spines['right'].set_position(('axes',0.1))
    #ax.yaxis.set_ticks_position('right')
    #ax.spines['top'].set_position(('axes',0.25))
    #ax.xaxis.set_ticks_position('top')
    #ax.spines['left'].set_color('none')
    #ax.spines['bottom'].set_color('none') 
    
def load_plt_data():
    return SUAVE.Input_Output.SUAVE.load('../B737/plot_data_B737.res')

if __name__ == '__main__':     
    main()  
    
