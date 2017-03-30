'''
Define the program flow class.

Rick Fenrich 2/14/17
'''

class Program_Flow():
    def __init__(self):
        self.name = 'none'
        
        self.fidelity_levels = 0
        self.evaluation_order = []
        self.mutual_setup_step = 0
        self.function_dependency = []
        self.n_cores = 0
        self.function_evals_in_unique_directory = 0
        self.link_files = []
        self.setup_link_files = []
        self.history_tags = []

