import numpy as np

exp_taylor_coeffs = np.array([	-0.000000400000000, -0.000000533333333, -0.000001280000000,
						-0.000003657142857, -0.000011377777778, -0.000037236363636,
						-0.000126030769231, -0.000436906666667, -0.001542023529412])
exp_taylor_coeffs = exp_taylor_coeffs*(10**7)
						
def taylor_approx(x,a):
	sum = x * exp_taylor_coeffs[0]
	if a < len(exp_taylor_coeffs):
		for i in range(1,len(exp_taylor_coeffs)):
			sum = sum + (exp_taylor_coeffs[i]*x**(2*i+1))*(1/(1+np.exp(taylor_approx(a-i,50))))     #1/(1+np.exp(1.8*(i-a)))    # 
	else:
		for i in range(1,len(exp_taylor_coeffs)):
			sum = sum + exp_taylor_coeffs[i]*x**(2*i+1)
	return sum	

def sigmoid(t, args):
	return (1/(1+np.exp(taylor_approx(args[0]*(t*10**10-args[1]), args[2]))))*(1/(1+np.exp(1000*(args[0]*(t*10**10-args[1]))))) + (1/(1+np.exp(taylor_approx(args[0]*(t*10**10-args[1]), args[3]))))*(1/(1+np.exp(-1000*(args[0]*(t*10**10-args[1])))))
	

input_initial = 	[0.4,  10, 0.6, 0.6,  -0.3, 12, 0.6, 0.6]
input_lower_bound = [0.2,  0,  0.5, 0.3, -1, 0, 0.3, 0.3]
input_upper_bound = [1,   25, 3,  4,  -0.2,  25, 4, 4]

parameter_names = ['left_steepness', 'left_shift', 'left_curvature_left', 'left_curvature_right', 'right_steepness', 'right_shift', 'right_curvature_left', 'right_curvature_right']

output_initial = 	 input_initial
output_lower_bound = input_lower_bound
output_upper_bound = input_upper_bound

input_shift = 1
input_length = 5
output_shift = 5
output_length = 1


