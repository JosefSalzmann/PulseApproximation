import numpy as np

exp_taylor_coeffs = np.array([	-0.000000400000000, 0, -0.000000533333333, 0, -0.000001280000000,
						0, -0.000003657142857, 0, -0.000011377777778, 0, -0.000037236363636,
						0, -0.000126030769231, 0, -0.000436906666667, 0, -0.001542023529412,
						0, -0.005518821052632, 0, -0.019972876190476, 0, -0.072944417391304,
						0, -0.268435456000000, 0, -0.994205392592593, 0, -3.702558013793104])
exp_taylor_coeffs = exp_taylor_coeffs*(10**7)
						
def taylor_approx(x,a):
	sum = 0
	for i in range(0,len(exp_taylor_coeffs)):
		sum = sum + (exp_taylor_coeffs[i]*x**(i+1))*(1/(1+np.exp(2*(i-a))))  #2.5
	return sum	

def sigmoid(t, args):
	return (1/(1+np.exp(taylor_approx(args[0]*(t*10**10-args[1]), args[2]))))*(1/(1+np.exp(1000*(args[0]*(t*10**10-args[1]))))) + (1/(1+np.exp(taylor_approx(args[0]*(t*10**10-args[1]), args[3]))))*(1/(1+np.exp(-1000*(args[0]*(t*10**10-args[1])))))
	

input_initial = 	[0.8,  10, 3, 3,  -0.3, 12, 5, 5]
input_lower_bound = [0,  0,  0, 0, -5, 0, 0, 0]
input_upper_bound = [5,25, 10,10,  0,  25,10,10]

parameter_names = ['left_steepness', 'left_shift', 'left_curvature_left', 'left_curvature_right', 'right_steepness', 'right_shift', 'right_curvature_left', 'right_curvature_right']

output_initial = 	 input_initial
output_lower_bound = input_lower_bound
output_upper_bound = input_upper_bound

input_shift = 1
input_length = 5
output_shift = 5
output_length = 1


