function results = extract_AVL_data(filename)

%Extract Data
File_id = fopen(filename);

while ~feof(File_id)
    data = fgetl(File_id);
    
    if length(data) > 18
        if strcmp(data(3:7),'CLtot')
            CL = str2num(data(11:19));
        end
        if strcmp(data(3:7),'CDtot')
            CD = str2num(data(11:19));
        end
        if strcmp(data(3:7),'CXtot')
            CX = str2num(data(11:19));
            Cl = str2num(data(33:41));
        end
        if strcmp(data(3:7),'CYtot')
            CY = str2num(data(11:19));
            Cm = str2num(data(33:41));
        end
        if strcmp(data(3:7),'CZtot')
            CZ = str2num(data(11:19));
            Cn = str2num(data(33:41));
        end  
    end
    if length(data) > 21
        if strcmp(data(2:21),'z'' force CL |    CLa')
            CLa = str2num(data(25:34));
            CLb = str2num(data(45:54));
        end
        if strcmp(data(2:21),'y  force CY |    CYa')
            CYa = str2num(data(25:34));
            CYb = str2num(data(45:54));
        end
        if strcmp(data(2:21),'x'' mom.  Cl''|    Cla')
            Cla = str2num(data(25:34));
            Clb = str2num(data(45:54));
        end
        if strcmp(data(2:21),'y  mom.  Cm |    Cma')
            Cma = str2num(data(25:34));
            Cmb = str2num(data(45:54));
        end
        if strcmp(data(2:21),'z'' mom.  Cn''|    Cna')
            Cna = str2num(data(25:34));
            Cnb = str2num(data(45:54));
        end
        
        
        if strcmp(data(2:21),'z'' force CL |    CLp')
            CLp = str2num(data(25:34));
            CLq = str2num(data(45:54));
            CLr = str2num(data(65:74));
        end
        if strcmp(data(2:21),'y  force CY |    CYp')
            CYp = str2num(data(25:34));
            CYq = str2num(data(45:54));
            CYr = str2num(data(65:74));
        end
        if strcmp(data(2:21),'x'' mom.  Cl''|    Clp')
            Clp = str2num(data(25:34));
            Clq = str2num(data(45:54));
            Clr = str2num(data(65:74));
        end
        if strcmp(data(2:21),'y  mom.  Cm |    Cmp')
            Cmp = str2num(data(25:34));
            Cmq = str2num(data(45:54));
            Cmr = str2num(data(65:74));
        end
        if strcmp(data(2:21),'z'' mom.  Cn''|    Cnp')
            Cnp = str2num(data(25:34));
            Cnq = str2num(data(45:54));
            Cnr = str2num(data(65:74));
        end
        
        
        if strcmp(data(2:21),'z'' force CL |   CLd1')
            CLde = str2num(data(25:34));
            CLdr = str2num(data(45:54));
        end
        if strcmp(data(2:21),'y  force CY |   CYd1')
            Cyde = str2num(data(25:34));
            CYdr = str2num(data(45:54));
        end
        if strcmp(data(2:21),'x'' mom.  Cl''|   Cld1')
            Clde = str2num(data(25:34));
            Cldr = str2num(data(45:54));
        end
        if strcmp(data(2:21),'y  mom.  Cm |   Cmd1')
            Cmde = str2num(data(25:34));
            Cmdr = str2num(data(45:54));
        end
        if strcmp(data(2:21),'z'' mom.  Cn''|   Cnd1')
            Cnde = str2num(data(25:34));
            Cndr = str2num(data(45:54));
        end
        
    end
end
fclose('all');
results = [CL,CD,CY,Cl,Cm,Cn,CX,CY,CZ]';
end