## @ingroup Input_Output-Results
# plot_constraint_diagram.py 

# Created: Nov. 2021 S. Karpuk
# Updated: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
#  Plot the constraint diagram
# ----------------------------------------------------------------------
## @ingroup Input_Output-Results
def plot_constraint_diagram(constraints, plot_tag, eng_type, filename='constraint_diagram'):
    """This creates a constraint diagram_plot and prints the design point

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    constraints.
        wing_loading            [N/m^2]
        all_constraints         [N/N] or [kW/N]        thrust or power curves
        combined_curve          [N/N] or [kW/N]        a combined constraint curve
        engine.type             <string>

    filename (optional)  <string> Determines the name of the saved file

    Outputs:
    filename              Saved files with names as above

    Properties Used:
    N/A
    """

    # Unpack inputs and 
    constraint_matrix       = constraints.constraint_matrix
    combined_constraint     = constraints.combined_design_curve
    design_thrust_to_weight = constraints.des_thrust_to_weight
    wing_loading            = constraints.wing_loading / 9.81
    design_wing_loading     = constraints.des_wing_loading / 9.81
    landing_wing_loading    = constraints.landing_wing_loading / 9.81

    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1) 
    ax.set_xlabel('W/S, kg/sq m')   

    # Convert the input into commmon units
    if eng_type != ('turbofan' or 'Turbofan') and eng_type != ('turbojet' or 'Turbojet'):
        ax.set_ylabel('P/W, kW/kg')
        plt.ylim(0, 0.3)
        combined_constraint     = 9.81 * combined_constraint / 1000             # converts to kW/kg
        constraint_matrix       = 9.81 * np.asarray(constraint_matrix) / 1000               # converts to kW/kg
        design_thrust_to_weight = 9.81 * design_thrust_to_weight / 1000         # converts to kW/kg 
    else:
        plt.ylim(0, 6)
        ax.set_ylabel('T/W, N/kg') 
        combined_constraint     = 9.81 * combined_constraint                    # converts to N/kg
        constraint_matrix       = 9.81 * np.asarray(constraint_matrix)                      # converts to N/kg
        design_thrust_to_weight = 9.81 * design_thrust_to_weight                # converts to N/kg    
             

    # plot all prescribed constraints
    for i in range(len(constraints.plot_legend)):
        ax.plot(wing_loading ,constraint_matrix[i][:], label = constraints.plot_legend[i])
    ax.plot(landing_wing_loading,[0,30], label = 'Landing', color = 'k')
    ax.plot(wing_loading ,combined_constraint, label = 'Combined', color = 'r')

    ax.scatter(design_wing_loading,design_thrust_to_weight, label = 'Design point')
    ax.set_xlim(0, landing_wing_loading[0]+10)     

    
    ax.legend(loc=2,)
    ax.grid(True)
    plt.savefig(filename + str('.png'), dpi=150)

    if plot_tag  == True: 
        plt.show()        
        

    # Write an output file with the design point
    f = open(filename + str('.dat'), "w")
    f.write('Output file with the constraint analysis design point\n\n')           
    f.write("Design point :\n")
    f.write('     Wing loading = ' + str(design_wing_loading) + ' kg/sq m\n') 
    if eng_type != ('turbofan' or 'Turbofan') and eng_type != ('turbojet' or 'Turbojet'):
        f.write('     Power-to-weight ratio = ' + str(design_thrust_to_weight) + ' kW/kg\n')    
    else:
        f.write('     Thrust-to-weight ratio = ' + str(design_thrust_to_weight) + ' N/kg\n')  
    f.close()    


# ----------------------------------------------------------------------
#   Module Test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print(' Error: No test defined ! ')
