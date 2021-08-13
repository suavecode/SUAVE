# generate_airfoil_transitions.py
# 
# Created:  March 2021, R. Erhard
# Modified: 


from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.generate_interpolated_airfoils import generate_interpolated_airfoils 
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry 
import pylab as plt
import os



def main():
    ospath        = os.path.abspath(__file__)
    separator     = os.path.sep
    airfoils_path = ospath.split('airfoil_import' + separator + 'airfoil_interpolation_test.py')[0] + 'Vehicles/Airfoils' + separator
    a_labels      = ["Clark_y", "E63"]
    nairfoils     = 4   # number of total airfoils
    
    a1 = airfoils_path + a_labels[0]+ ".txt"
    a2 = airfoils_path + a_labels[1]+ ".txt"
    new_files = generate_interpolated_airfoils(a1, a2, nairfoils, save_filename="Transition")
    plt.show()
    
    # import the new airfoil geometries and compare to the regression:
    airfoil_data_1   = import_airfoil_geometry([new_files['a_1'].name])
    airfoil_data_2   = import_airfoil_geometry([new_files['a_2'].name])    
    airfoil_data_1_r = import_airfoil_geometry(["Transition1_regression.txt"])
    airfoil_data_2_r = import_airfoil_geometry(["Transition2_regression.txt"])
    
    # ensure coordinates are the same:
    
    assert( max(abs(airfoil_data_1.x_coordinates[0] - airfoil_data_1_r.x_coordinates[0])) < 1e-5)
    assert( max(abs(airfoil_data_2.x_coordinates[0] - airfoil_data_2_r.x_coordinates[0])) < 1e-5)
    assert( max(abs(airfoil_data_1.y_coordinates[0] - airfoil_data_1_r.y_coordinates[0])) < 1e-5)
    assert( max(abs(airfoil_data_2.y_coordinates[0] - airfoil_data_2_r.y_coordinates[0])) < 1e-5)

    return


if __name__ == "__main__":
    main()