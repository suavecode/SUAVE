# Tim Momose, October 2014


from SUAVE.Structure import Data, Data_Exception, Data_Warning

def read_results(results_directory,results_filename):
	
	results_path = results_directory + results_filename
	results = open(results_path)
	
	try:
		lines   = results.readlines()
		CL_line = lines[23]
		CD_line = lines[24]
		CLtot   = CL_line[10:20].strip()
		CDtot   = CD_line[10:20].strip()
		
		res = Data()
		res.tag = ('Results: {}'.format(results_filename))
		res.CL_total = CLtot
		res.CD_total = CDtot
		
	finally:
		results.close()
	
	return res
