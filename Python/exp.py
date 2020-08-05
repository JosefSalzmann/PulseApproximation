import numpy as np

def sigmoid(t, args):
	return 1/(1+np.exp(-args[0]*(t*10**10-args[1])))

input_initial = 	[1, 11,-1, 12] #[-1, 11,1, 12]
input_lower_bound = [0, 0,-100, 0] #[-100, 0,0, 0] 
input_upper_bound = [100, 25,0, 25] #[0, 25,100, 25]


output_initial = 	 [-1, 11,1, 12] #[1, 11,-1, 12]
output_lower_bound = [-100, 0,0, 0] #[0, 0,-100, 0]
output_upper_bound = [0, 25,100, 25] #[100, 25,0, 25]

parameter_names = ['left_steepness', 'left_shift', 'right_steepness', 'right_shift']

steepness_index = 0
shift_index = 1


input_shift = 1
input_length = 3
output_shift = 1 #delay
output_length = 3

trace_shift = 1 # all parameters with this index will be deteremined automatically
trace_initial = [1, 0]
trace_falling_lower_bound = [-2.0, 0] 
trace_falling_upper_bound = [-0.1, 0] 
trace_rising_lower_bound = [0.1, 0] 
trace_rising_upper_bound = [2, 0] 
trace_steepness = 0 # index of the steepness parameter
trace_paramters_names = ['Steepness', 'Shift']
