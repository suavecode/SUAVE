## @ingroup Time_Accurate-Simulations
# save_prop_vtk.py
# 
# Created:    Jun 2021, R. Erhard
# Modified: 
#           


def save_prop_vtk(vehicle, filename, Results, i_prop, Gprops):
    """
    Assumptions: 
         Quad cell structures for mesh
    """
    # Generate propeller point geometry
    prop   = vehicle.propulsors.battery_propeller.propeller
    # Create file
    with open(filename, 'w') as f:
        #---------------------
        # Write header
        #---------------------
        l1 = "# vtk DataFile Version 4.0"               # File version and identifier
        l2 = "\nSUAVE Model of PROWIM Propeller "       # Title 
        l3 = "\nASCII"                                  # Data type
        l4 = "\nDATASET UNSTRUCTURED_GRID"              # Dataset structure / topology     
        
        header = [l1, l2, l3, l4]
        f.writelines(header)     
        
        # --------------------
        # Write Points
        # --------------------   
        n_r      = len(prop.chord_distribution)
        n_af     = Gprops.n_af
        n_blades = prop.number_of_blades
        
        n_vertices = n_blades*(n_r)*(n_af)    # total number of node vertices
        points_header = "\n\nPOINTS "+str(n_vertices) +" float"
        f.write(points_header)
        
        for B_idx in range(n_blades):
            # Get geometry of blade for current propeller instance
            G = Gprops[i_prop][B_idx]
            for c_idx in range(n_af):
                for r_idx in range(n_r):
                    if r_idx ==n_r-1 and c_idx ==n_af-1:
                        # rightmost TE cell, use B2
                        xp = round(G.XB2[c_idx-1,r_idx-1],4)
                        yp = round(G.YB2[c_idx-1,r_idx-1],4)
                        zp = round(G.ZB2[c_idx-1,r_idx-1],4)                    
                    elif c_idx == n_af-1:
                        # last chordwise element, use A2 of prior
                        xp = round(G.XA2[c_idx-1,r_idx],4)
                        yp = round(G.YA2[c_idx-1,r_idx],4)
                        zp = round(G.ZA2[c_idx-1,r_idx],4)
                    elif r_idx==n_r-1:
                        # last radial station, use prior B1
                        xp = round(G.XB1[c_idx,r_idx-1],4)
                        yp = round(G.YB1[c_idx,r_idx-1],4)
                        zp = round(G.ZB1[c_idx,r_idx-1],4)
                    else:
                        # print the point index (Left LE --> Left TE --> Right LE --> Right TE)
                        xp = round(G.XA1[c_idx,r_idx],4)
                        yp = round(G.YA1[c_idx,r_idx],4)
                        zp = round(G.ZA1[c_idx,r_idx],4)
                        
                    new_point = "\n"+str(xp)+" "+str(yp)+" "+str(zp)
                    node_number = r_idx + (n_r)*c_idx
                    #print("Point: ", new_point, "; Node Number: ", str(node_number))
                    f.write(new_point)  
                    print(node_number)
                    
        #---------------------    
        # Write Cells:
        #---------------------
        cells_per_blade = n_af*(n_r-1)
        n_cells         = n_blades*cells_per_blade
        v_per_cell      = 4    # quad cells
        size            = n_cells*(1+v_per_cell) # total number of integer values required to represent the list
        cell_header     = "\n\nCELLS "+str(n_cells)+" "+str(size)
        f.write(cell_header)
        
        for B_idx in range(n_blades):
            for i in range(cells_per_blade):
                
                new_cell = "\n4 "+str(i)+" "+str(i+n_af)+" "+str(i+n_af+1)+" "+str(i+1)
                f.write(new_cell)
                
        
        #---------------------        
        # Write Cell Types:
        #---------------------
        cell_type_header  = "\n\nCELL_TYPES "+str(n_cells)
        f.write(cell_type_header)        
        for i in range(n_cells):
            f.write("\n9")            
    f.close()
    
    
    return