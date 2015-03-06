# AVL_Callable.py
#
# Created:  Tim Momose, Dec 2014


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import os
import numpy as np

# SUAVE imports
from SUAVE.Core import Data
import SUAVE.Plugins.VyPy.tools.redirect as redirect
from SUAVE.Analyses.Missions.Segments.Conditions.Aerodynamics import Aerodynamics
from SUAVE.Analyses.Missions.Segments.Conditions.Conditions   import Conditions
from SUAVE.Methods.Aerodynamics.AVL.purge_files   import purge_files
from SUAVE.Methods.Aerodynamics.AVL.Data.Results  import Results
from SUAVE.Methods.Aerodynamics.AVL.Data.Settings import Settings
from SUAVE.Methods.Aerodynamics.AVL.Data.Cases import Run_Case


# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class AVL_Callable(Data):
    """ SUAVE.Methods.Aerodynamics.AVL.AVL_Callable
        aerodynamic model that performs a vortex lattice analysis using AVL
        (Athena Vortex Lattice, by Mark Drela of MIT).

        this class is callable, see self.__call__

    """

    def __defaults__(self):
        self.tag        = 'avl'
        self.keep_files = True

        self.settings = Settings()

        self.analysis_temps = Data()
        self.analysis_temps.current_batch_index = 0
        self.analysis_temps.current_batch_file  = None
        self.analysis_temps.current_cases       = None


    def initialize(self,vehicle):

        self.features = vehicle
        self.tag      = 'avl_analysis_of_{}'.format(vehicle.tag)
        self.settings.filenames.run_folder = \
            os.path.abspath(self.settings.filenames.run_folder)
        if not os.path.exists(self.settings.filenames.run_folder):
            os.mkdir(self.settings.filenames.run_folder)

        return

    def translate_conditions_to_cases(self,conditions):
        """ Takes SUAVE Conditions() data structure and translates to a Container of
            avl Run_Case()s.
        """
        # set up aerodynamic Conditions object
        cases = Run_Case.Container()

        for i in range(conditions._size):
            case = Run_Case()
            case.tag  = self.settings.filenames.case_template.format(self.analysis_temps.current_batch_index,i+1)
            case.mass = conditions.weights.total_mass[i][0]
            case.conditions.freestream.mach     = conditions.freestream.mach_number[i][0]
            case.conditions.freestream.velocity = conditions.freestream.velocity[i][0]
            case.conditions.freestream.density  = conditions.freestream.density[i][0]
            case.conditions.freestream.gravitational_acceleration = conditions.freestream.gravity[i][0]
            case.conditions.aerodynamics.angle_of_attack = conditions.aerodynamics.angle_of_attack[i][0]
            case.conditions.aerodynamics.side_slip_angle = conditions.aerodynamics.side_slip_angle[i][0]
            case.stability_and_control.control_deflections = np.array([[]]) # TODO How to do this from the SUAVE side?
            cases.append_case(case)

        return cases

    def translate_results_to_conditions(self,cases,results):
        """ Takes avl results structure containing the results of each run case stored
            each in its own Data() object. Translates into the Conditions() data structure.
        """
        # set up aerodynamic Conditions object
        res = Aerodynamics()
        ones_1col = res.ones_row(1)
        # add missing entries
        res.aerodynamics.roll_moment_coefficient  = ones_1col * 0
        res.aerodynamics.pitch_moment_coefficient = ones_1col * 0
        res.aerodynamics.yaw_moment_coefficient   = ones_1col * 0
        res.aerodynamics.drag_breakdown.induced   = Conditions()
        res.aerodynamics.drag_breakdown.induced.total = ones_1col * 0
        res.aerodynamics.drag_breakdown.induced.efficiency_factor = ones_1col * 0
        res.aerodynamics.cz_alpha                 = ones_1col * 0
        res.aerodynamics.cy_alpha                 = ones_1col * 0
        res.aerodynamics.cl_alpha                 = ones_1col * 0
        res.aerodynamics.cm_alpha                 = ones_1col * 0
        res.aerodynamics.cn_alpha                 = ones_1col * 0
        res.aerodynamics.cz_beta                  = ones_1col * 0
        res.aerodynamics.cy_beta                  = ones_1col * 0
        res.aerodynamics.cl_beta                  = ones_1col * 0
        res.aerodynamics.cm_beta                  = ones_1col * 0
        res.aerodynamics.cn_beta                  = ones_1col * 0
        res.aerodynamics.neutral_point            = ones_1col * 0

        res.expand_rows(len(cases))

        # Move results data to the Conditions data structure       
        for i,case_res in enumerate(results):
            res.freestream.velocity[i][0] = cases[i].conditions.freestream.velocity
            res.freestream.mach_number[i][0] = cases[i].conditions.freestream.mach
            res.freestream.gravity[i][0] = cases[i].conditions.freestream.gravitational_acceleration
            res.freestream.density[i][0] = cases[i].conditions.freestream.density

            res.weights.total_mass[i][0] = cases[i].mass

            res.aerodynamics.roll_moment_coefficient[i][0] = case_res.aerodynamics.roll_moment_coefficient
            res.aerodynamics.pitch_moment_coefficient[i][0] = case_res.aerodynamics.pitch_moment_coefficient
            res.aerodynamics.yaw_moment_coefficient[i][0] = case_res.aerodynamics.yaw_moment_coefficient
            res.aerodynamics.lift_coefficient[i][0] = case_res.aerodynamics.total_lift_coefficient
            res.aerodynamics.drag_breakdown.induced.total[i][0] = case_res.aerodynamics.induced_drag_coefficient
            res.aerodynamics.drag_breakdown.induced.efficiency_factor[i][0] = case_res.aerodynamics.span_efficiency_factor
            res.aerodynamics.cz_alpha[i][0] = -case_res.stability.alpha_derivatives.lift_curve_slope
            res.aerodynamics.cy_alpha[i][0] = case_res.stability.alpha_derivatives.side_force_derivative
            res.aerodynamics.cl_alpha[i][0] = case_res.stability.alpha_derivatives.roll_moment_derivative
            res.aerodynamics.cm_alpha[i][0] = case_res.stability.alpha_derivatives.pitch_moment_derivative
            res.aerodynamics.cn_alpha[i][0] = case_res.stability.alpha_derivatives.yaw_moment_derivative
            res.aerodynamics.cz_beta[i][0] = -case_res.stability.beta_derivatives.lift_coefficient_derivative
            res.aerodynamics.cy_beta[i][0] = case_res.stability.beta_derivatives.side_force_derivative
            res.aerodynamics.cl_beta[i][0] = case_res.stability.beta_derivatives.roll_moment_derivative
            res.aerodynamics.cm_beta[i][0] = case_res.stability.beta_derivatives.pitch_moment_derivative
            res.aerodynamics.cn_beta[i][0] = case_res.stability.beta_derivatives.yaw_moment_derivative
            res.aerodynamics.neutral_point[i][0] = case_res.stability.neutral_point

        return res


    def evaluate(self,run_conditions):
        """ process vehicle to setup geometry, condititon and configuration

            Inputs:
                run_conditions - DataDict() of aerodynamic conditions; until input
                method is finalized, will just assume mass_properties are always as 
                defined in self.features

            Outputs:
                results - a DataDict() of type 
                SUAVE.Analyses.Missions.Segments.Conditions.Aerodynamics(), augmented with
                case data on moment coefficients and control derivatives

            Assumptions:

        """
        #assert cases is not None and len(cases) , 'run_case container is empty or None'
        self.analysis_temps.current_batch_index  += 1
        self.analysis_temps.current_batch_file = self.settings.filenames.batch_template.format(self.analysis_temps.current_batch_index)
        cases = self.translate_conditions_to_cases(run_conditions)
        self.analysis_temps.current_cases = cases        

        for case in cases:
            case.result_filename = self.settings.filenames.output_template.format(case.tag)

        with redirect.folder(self.settings.filenames.run_folder,[],[],False):
            write_geometry(self)
            write_run_cases(self)
            write_input_deck(self)

            results_avl = run_analysis(self)

        results = self.translate_results_to_conditions(cases,results_avl)

        if not self.keep_files:
            from shutil import rmtree
            rmtree(os.path.abspath(self.settings.filenames.run_folder))

        return results

    def __call__(self,*args,**kwarg):
        return self.evaluate(*args,**kwarg)



##############################
## Methods used in __call__ ##
##############################

def write_geometry(self):
    # imports
    from SUAVE.Methods.Aerodynamics.AVL.write_geometry import make_header_text,translate_avl_wing,make_surface_text,translate_avl_body,make_body_text

    # unpack inputs
    aircraft      = self.features
    geometry_file = self.settings.filenames.features

    # Open the geometry file after purging if it already exists
    purge_files([geometry_file])

    with open(geometry_file,'w') as geometry:

        header_text = make_header_text(self)
        geometry.write(header_text)
        for w in aircraft.wings:
            avl_wing = translate_avl_wing(w)
            wing_text = make_surface_text(avl_wing)
            geometry.write(wing_text)
        for b in aircraft.fuselages:
            avl_body = translate_avl_body(b)
            body_text = make_body_text(avl_body)
            geometry.write(body_text)

    return


def write_run_cases(self):

    # imports
    from SUAVE.Methods.Aerodynamics.AVL.write_run_cases import make_controls_case_text

    # unpack avl_inputs
    batch_filename = self.analysis_temps.current_batch_file
    aircraft       = self.features

    base_case_text = \
'''

 ---------------------------------------------
 Run case  {0}:   {1}

 alpha        ->  alpha       =   {2}
 beta         ->  beta        =   {3}
 pb/2V        ->  pb/2V       =   0.00000
 qc/2V        ->  qc/2V       =   0.00000
 rb/2V        ->  rb/2V       =   0.00000
{4}

 alpha     =   0.00000     deg
 beta      =   0.00000     deg
 pb/2V     =   0.00000
 qc/2V     =   0.00000
 rb/2V     =   0.00000
 CL        =   0.00000
 CDo       =   0.00000
 bank      =   0.00000     deg
 elevation =   0.00000     deg
 heading   =   0.00000     deg
 Mach      =   {5}
 velocity  =   {6}     m/s
 density   =   {7}     kg/m^3
 grav.acc. =   {8}     m/s^2
 turn_rad. =   0.00000     m
 load_fac. =   0.00000
 X_cg      =   {9}     m
 Y_cg      =   {10}     m
 Z_cg      =   {11}     m
 mass      =   {12}     kg
 Ixx       =   {13}     kg-m^2
 Iyy       =   {14}     kg-m^2
 Izz       =   {15}     kg-m^2
 Ixy       =   {16}     kg-m^2
 Iyz       =   {17}     kg-m^2
 Izx       =   {18}     kg-m^2
 visc CL_a =   0.00000
 visc CL_u =   0.00000
 visc CM_a =   0.00000
 visc CM_u =   0.00000

'''#{4} is a set of control surface inputs that will vary depending on the control surface configuration


    # Open the geometry file after purging if it already exists
    purge_files([batch_filename])
    with open(batch_filename,'w') as runcases:

        x_cg = self.features.mass_properties.center_of_gravity[0]
        y_cg = self.features.mass_properties.center_of_gravity[1]
        z_cg = self.features.mass_properties.center_of_gravity[2]
        mass = 0 #self.default_case.mass TODO: FIGURE OUT WHAT TO DEFAULT MASS TO, AND WHERE TO STORE IT BEFORE ANALYSIS.
        moments_of_inertia = aircraft.mass_properties.moments_of_inertia.tensor
        Ixx  = moments_of_inertia[0][0]
        Iyy  = moments_of_inertia[1][1]
        Izz  = moments_of_inertia[2][2]
        Ixy  = moments_of_inertia[0][1]
        Iyz  = moments_of_inertia[1][2]
        Izx  = moments_of_inertia[2][0]

        for case in self.analysis_temps.current_cases:
            index = case.index
            name  = case.tag
            alpha = case.conditions.aerodynamics.angle_of_attack
            beta  = case.conditions.aerodynamics.side_slip_angle
            mach  = case.conditions.freestream.mach
            v     = case.conditions.freestream.velocity
            rho   = case.conditions.freestream.density
            g     = case.conditions.freestream.gravitational_acceleration

            # form controls text
            controls = []
            if case.stability_and_control.control_deflections:
                for cs in case.stability_and_control.control_deflections:
                    cs_text = make_controls_case_text(cs)
                    controls.append(cs_text)
            controls_text = ''.join(controls)
            case_text = base_case_text.format(index,name,alpha,beta,controls_text,
                                              mach,v,rho,g,x_cg,y_cg,z_cg,mass,
                                              Ixx,Iyy,Izz,Ixy,Iyz,Izx)
            runcases.write(case_text)

    return


def write_input_deck(self):

    open_runs = \
'''CASE {}
'''
    base_input = \
'''OPER
'''
    # unpack
    files_path    = self.settings.filenames.run_folder
    batch         = self.analysis_temps.current_batch_file
    deck_filename = self.settings.filenames.input_deck

    # purge old versions and write the new input deck
    purge_files([deck_filename])
    with open(deck_filename,'w') as input_deck:
        input_deck.write(open_runs.format(batch))
        input_deck.write(base_input)
        for case in self.analysis_temps.current_cases:
            case_command = make_case_command(self,case)
            input_deck.write(case_command)
        input_deck.write('\n\nQUIT\n')

    return


def make_case_command(self,case):

    base_case_command = \
'''{0}
x
{1}
{2}
'''
    directory = self.settings.filenames.run_folder
    index = case.index
    case_tag = case.tag
    res_type = 'st' # This needs to change to multiple ouputs if you want to add the ability to read other types of results
    results_file = case.result_filename
    purge_files([results_file],directory)
    case_command = base_case_command.format(index,res_type,results_file)

    return case_command


def run_analysis(self):

    #avl_bin_path      = self.settings.filenames.avl_bin_name
    #files_path        = self.settings.filenames.run_folder
    #geometry_filename = self.settings.filenames.features
    #deck_filename     = self.settings.filenames.input_deck

    call_avl(self)

    results = read_results(self)

    return results


def call_avl(self):

    import sys
    import time
    import subprocess
    import SUAVE.Plugins.VyPy.tools.redirect as redirect

    log_file = self.settings.filenames.log_filename
    err_file = self.settings.filenames.err_filename
    if isinstance(log_file,str):
        purge_files(log_file)
    if isinstance(err_file,str):
        purge_files(err_file)
    avl_call = self.settings.filenames.avl_bin_name
    geometry = self.settings.filenames.features
    in_deck  = self.settings.filenames.input_deck
    #batch    = self.analysis_temps.current_batch_file

    with redirect.output(log_file,err_file):

        ctime = time.ctime() # Current date and time stamp
        sys.stdout.write("Log File of System stdout from AVL Run \n{}\n\n".format(ctime))
        sys.stderr.write("Log File of System stderr from AVL Run \n{}\n\n".format(ctime))

        with open(in_deck,'r') as commands:
            avl_run = subprocess.Popen([avl_call,geometry],stdout=sys.stdout,stderr=sys.stderr,stdin=subprocess.PIPE)
            for line in commands:
                avl_run.stdin.write(line)
        avl_run.wait()

        exit_status = avl_run.returncode
        ctime = time.ctime()
        sys.stdout.write("\nProcess finished: {0}\nExit status: {1}\n".format(ctime,exit_status))
        sys.stderr.write("\nProcess finished: {0}\nExit status: {1}\n".format(ctime,exit_status))        

    return exit_status


def read_results(self):

    results = Data()

    for case in self.analysis_temps.current_cases:
        num_ctrl = len(case.stability_and_control.control_deflections)
        with open(case.result_filename,'r') as res_file:
            case_res = Results()
            case_res.tag = case.tag
            lines   = res_file.readlines()
            case_res.aerodynamics.roll_moment_coefficient  = float(lines[19][32:42].strip())
            case_res.aerodynamics.pitch_moment_coefficient = float(lines[20][32:42].strip())
            case_res.aerodynamics.yaw_moment_coefficient   = float(lines[21][32:42].strip())
            case_res.aerodynamics.total_lift_coefficient   = float(lines[23][10:20].strip())
            #case_res.aerodynamics.total_drag_coefficient   = float(lines[24][10:20].strip())
            case_res.aerodynamics.induced_drag_coefficient = float(lines[25][32:42].strip())
            case_res.aerodynamics.span_efficiency_factor   = float(lines[27][32:42].strip())

            case_res.stability.alpha_derivatives.lift_curve_slope           = float(lines[36+num_ctrl][25:35].strip())
            case_res.stability.alpha_derivatives.side_force_derivative      = float(lines[37+num_ctrl][25:35].strip())
            case_res.stability.alpha_derivatives.roll_moment_derivative     = float(lines[38+num_ctrl][25:35].strip())
            case_res.stability.alpha_derivatives.pitch_moment_derivative    = float(lines[39+num_ctrl][25:35].strip())
            case_res.stability.alpha_derivatives.yaw_moment_derivative      = float(lines[40+num_ctrl][25:35].strip())
            case_res.stability.beta_derivatives.lift_coefficient_derivative = float(lines[36+num_ctrl][44:55].strip())
            case_res.stability.beta_derivatives.side_force_derivative       = float(lines[37+num_ctrl][44:55].strip())
            case_res.stability.beta_derivatives.roll_moment_derivative      = float(lines[38+num_ctrl][44:55].strip())
            case_res.stability.beta_derivatives.pitch_moment_derivative     = float(lines[39+num_ctrl][44:55].strip())
            case_res.stability.beta_derivatives.yaw_moment_derivative       = float(lines[40+num_ctrl][44:55].strip())
            case_res.stability.neutral_point                                = float(lines[50+13*(num_ctrl>0)][22:33].strip())

            results.append(case_res)


    return results