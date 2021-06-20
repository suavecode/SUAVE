## @ingroup Time_Accurate-Simulations
# save_wing_vtk.py
# 
# Created:    Jun 2021, R. Erhard
# Modified: 
#           


def save_wing_vtk(VD, filename, Results):
    "Saves a SUAVE wing object as a VTK in legacy format."
    
    n_cw = VD.n_cw[0]
    n_sw = VD.n_sw[0]
    n_cp = VD.n_cp # number of control points and panels on wing

    # Create file
    with open(filename, 'w') as f:
    
        #---------------------
        # Write header
        #---------------------
        l1 = "# vtk DataFile Version 4.0"     # File version and identifier
        l2 = "\nSUAVE Model of PROWIM Wing "  # Title 
        l3 = "\nASCII"                        # Data type
        l4 = "\nDATASET UNSTRUCTURED_GRID"    # Dataset structure / topology

        header = [l1, l2, l3, l4]
        f.writelines(header) 
        
        #---------------------    
        # Write Points:
        #---------------------    
        n_indices = (n_cw+1)*(n_sw+1)    # total number of cell vertices
        points_header = "\n\nPOINTS "+str(n_indices) +" float"
        f.write(points_header)
        
        cw_laps =0
        for i in range(n_indices):
            
            if i == n_indices-1:
                # Last index; use B2 to get rightmost TE node
                xp = round(VD.XB2[i-cw_laps-n_cw-1],4)
                yp = round(VD.YB2[i-cw_laps-n_cw-1],4)
                zp = round(VD.ZB2[i-cw_laps-n_cw-1],4)
            elif i > (n_indices-1-(n_cw+1)):
                # Last spanwise set; use B1 to get rightmost node indices
                xp = round(VD.XB1[i-cw_laps-n_cw],4)
                yp = round(VD.YB1[i-cw_laps-n_cw],4)
                zp = round(VD.ZB1[i-cw_laps-n_cw],4)
            elif i==0:
                # first point
                xp = round(VD.XA1[i],4)
                yp = round(VD.YA1[i],4)
                zp = round(VD.ZA1[i],4)            
                
            elif i==cw_laps + n_cw*(cw_laps+1): #i%n_cw==0:
                # Last chordwise station for this spanwise location; use A2 to get left TE node
                cw_laps = cw_laps +1
                xp = round(VD.XA2[i-cw_laps],4)
                yp = round(VD.YA2[i-cw_laps],4)
                zp = round(VD.ZA2[i-cw_laps],4)  
                
            else:
                # print the point index (Left LE --> Left TE --> Right LE --> Right TE)
                xp = round(VD.XA1[i-cw_laps],4)
                yp = round(VD.YA1[i-cw_laps],4)
                zp = round(VD.ZA1[i-cw_laps],4)
            
            new_point = "\n"+str(xp)+" "+str(yp)+" "+str(zp)
            f.write(new_point)
    
        #---------------------    
        # Write Cells:
        #---------------------
        n            = n_cp # total number of cells
        v_per_cell   = 4 # quad cells
        size         = n_cp*(1+v_per_cell) # total number of integer values required to represent the list
        cell_header  = "\n\nCELLS "+str(n)+" "+str(size)
        f.write(cell_header)
        
        
        for i in range(n_cp):
            if i==0:
                node = i
            elif i%n_cw ==0:
                node = node+1
            new_cell = "\n4 "+str(node)+" "+str(node+1)+" "+str(node+n_cw+2)+" "+str(node+n_cw+1)
            f.write(new_cell)
            
            # update node:
            node = node+1
    
        #---------------------        
        # Write Cell Types:
        #---------------------
        cell_type_header  = "\n\nCELL_TYPES "+str(n)
        f.write(cell_type_header)        
        for i in range(n_cp):
            f.write("\n9")
    
        #--------------------------        
        # Write Scalar Cell Data:
        #--------------------------
        cell_data_header  = "\n\nCELL_DATA "+str(n)
        f.write(cell_data_header)    
        
        # First scalar value
        f.write("\nSCALARS cl float 1")
        f.write("\nLOOKUP_TABLE default")   
        cl = Results['cl_y_DVE'][0]
        for i in range(n_cp):
            new_cl = str(cl[int(i/n_cw)])
            f.write("\n"+new_cl)
            
        f.write("\nSCALARS cl_CL float 1")
        f.write("\nLOOKUP_TABLE default")   
        cl = Results['cl_y_DVE'][0]
        CL = Results['CL_wing_DVE'][0][0]
        for i in range(n_cp):
            new_cl_CL = str(cl[int(i/n_cw)]/CL)
            f.write("\n"+new_cl_CL)
            
        f.write("\nSCALARS cd float 1")
        f.write("\nLOOKUP_TABLE default")   
        cd = Results['cdi_y_DVE'][0]
        for i in range(n_cp):
            new_cd = str(cd[int(i/n_cw)])
            f.write("\n"+new_cd)  
            
        f.write("\nSCALARS cd_CD float 1")
        f.write("\nLOOKUP_TABLE default")   
        cd = Results['cdi_y_DVE'][0]
        CD = Results['CDi_wing_DVE'][0][0]
        for i in range(n_cp):
            new_cd_CD = str(cd[int(i/n_cw)]/CD)
            f.write("\n"+new_cd_CD)        
    
    f.close()
    
    return
