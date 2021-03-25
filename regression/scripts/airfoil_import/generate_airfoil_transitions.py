# generate_airfoil_transitions.py
# 
# Created:  March 2021, R. Erhard
# Modified: 


from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.generate_airfoil_transitions import generate_airfoil_transition 
import pylab as plt
import os



def main():
    
    airfoils_path = os.path.join(os.path.dirname(__file__), "Polars/")
    a_labels  = ["Clark_y", "E63"]
    space     = 0.2 # distance for transition to occur
    nairfoils = 4   # number of total airfoils
    
    a1 = airfoils_path + a_labels[0]+".txt"
    a2 = airfoils_path + a_labels[1]+ ".txt"
    new_files = generate_airfoil_transition(a1, a2, space, nairfoils, save_file=True, save_filename="Polars/Transition")
    #plt.show()
    
    # open and check new file:
    f = open(new_files[next(iter(new_files))].name)
    data_block = f.readlines()
    
    test_entry = data_block[30]
    assert( test_entry == ' 0.514476 0.080181\n')
    
    return


if __name__ == "__main__":
    main()