import numpy as np

def sigmoid(t, args):
	return 1/(1+np.exp(-args[0]*(t*10**10-args[1])))

input_initial = 	[1, 10,-1, 10]
input_lower_bound = [0, 0,-100, 0]
input_upper_bound = [100, 25,0, 25]


output_initial = 	 input_initial
output_lower_bound = input_lower_bound
output_upper_bound = input_upper_bound

parameter_names = ['left_steepness', 'left_shift', 'right_steepness', 'right_shift'];

input_shift = 1
input_length = 3
output_shift = 3
output_length = 1