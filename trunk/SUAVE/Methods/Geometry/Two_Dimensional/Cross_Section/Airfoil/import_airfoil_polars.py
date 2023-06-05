## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# import_airfoil_polars.py
#
# Created:  Mar 2019, M. Clarke
#           Mar 2020, M. Clarke
#           Sep 2020, M. Clarke
#           May 2021, R. Erhard
#           Nov 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data, Units
import numpy as np

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def  import_airfoil_polars(airfoil_polar_files,angel_of_attack_discretization = 89):
    """This imports airfoil polars from a text file output from XFOIL or Airfoiltools.com

    Assumptions:
    Input airfoil polars file is obtained from XFOIL or from Airfoiltools.com
    Source:
    http://airfoiltools.com/
    Inputs:
    airfoil polar files   <list of strings>
    Outputs:
    data       numpy array with airfoil data
    Properties Used:
    N/A
    """

    # number of airfoils
    num_polars                   = 0
    n_p = len(airfoil_polar_files)
    if n_p < 3:
        raise AttributeError('Provide three or more airfoil polars to compute surrogate')
    num_polars            = max(num_polars, n_p)

    # create empty data structures
    airfoil_data = Data()
    AoA          = np.zeros((num_polars,angel_of_attack_discretization))
    CL           = np.zeros((num_polars,angel_of_attack_discretization))
    CD           = np.zeros((num_polars,angel_of_attack_discretization))
    Re           = np.zeros(num_polars)
    Ma           = np.zeros(num_polars)

    AoA_interp = np.linspace(-6,16,angel_of_attack_discretization)

    for j in range(len(airfoil_polar_files)):
        # Open file and read column names and data block
        f = open(airfoil_polar_files[j])
        data_block = f.readlines()
        f.close()

            # check for xfoil format
            xFoilLine = pd.read_csv(airfoil_polar_files[i][j], sep="\t", skiprows=0, nrows=1)
            if "XFOIL" in str(xFoilLine):
                xfoilPolarFormat = True
                polarData = pd.read_csv(airfoil_polar_files[i][j], skiprows=[1,2,3,4,5,6,7,8,9,11], skipinitialspace=True, delim_whitespace=True)
            elif "xflr5" in str(xFoilLine):
                xfoilPolarFormat = True
                polarData = pd.read_csv(airfoil_polar_files[i][j], skiprows=[0,1,2,3,4,5,6,7,8,10], skipinitialspace=True, delim_whitespace=True)
            else:
                xfoilPolarFormat = False

            # Read data
            if xfoilPolarFormat:
                # get data, extract Re, Ma
<<<<<<< HEAD

=======
                
>>>>>>> 72cb92b496e5352bef50a3348acc071dac763fbe
                headerLine = pd.read_csv(airfoil_polar_files[i][j], sep="\t", skiprows=7, nrows=1)
                headerString = str(headerLine.iloc[0])
                ReString = headerString.split('Re =',1)[1].split('e 6',1)[0]
                MaString = headerString.split('Mach =',1)[1].split('Re',1)[0]
            else:
                # get data, extract Re, Ma
                polarData = pd.read_csv(airfoil_polar_files[i][j], sep=" ")
                ReString = airfoil_polar_files[i][j].split('Re_',1)[1].split('e6',1)[0]
                MaString = airfoil_polar_files[i][j].split('Ma_',1)[1].split('_',1)[0]

            airfoil_aoa = polarData["alpha"]
            airfoil_cl = polarData["CL"]
            airfoil_cd = polarData["CD"]

        # Remove any extra lines at end of file:
        last_line = False
        while last_line == False:
            if data_block[-1]=='\n':
                data_block = data_block[0:-1]
            else:
                last_line = True

        data_len = len(data_block)
        airfoil_aoa= np.zeros(data_len)
        airfoil_cl = np.zeros(data_len)
        airfoil_cd = np.zeros(data_len)

        # Loop through each value: append to each column
        for line_count , line in enumerate(data_block):
            airfoil_aoa[line_count] = float(data_block[line_count][0:8].strip())
            airfoil_cl[line_count]  = float(data_block[line_count][10:17].strip())
            airfoil_cd[line_count]  = float(data_block[line_count][20:27].strip())

        AoA[j,:] = AoA_interp
        CL[j,:]  = np.interp(AoA_interp,airfoil_aoa,airfoil_cl)
        CD[j,:]  = np.interp(AoA_interp,airfoil_aoa,airfoil_cd)

    airfoil_data.aoa_from_polar               = AoA*Units.degrees
    airfoil_data.re_from_polar                = Re
    airfoil_data.mach_number                  = Ma
    airfoil_data.lift_coefficients            = CL
    airfoil_data.drag_coefficients            = CD

    return airfoil_data
