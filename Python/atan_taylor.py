import numpy as np

tan_taylor_coeffs = [	1.000000000000000, 0, 0.333333333333333, 0, 0.133333333333333, 
						0, 0.053968253968254, 0, 0.021869488536155, 0, 0.008863235529902,
						0, 0.003592128036572, 0, 0.001455834387051, 0, 0.000590027440946,
						0, 0.000239129114244, 0, 0.000096915379569, 0, 0.000039278323883,
						0, 0.000015918905069, 0, 0.000006451689216, 0, 0.000002614771151,
						0, 0.000001059726832, 0, 0.000000429491108 ]
						
def taylor_approx(x,a):
	sum = x
	if a < 50:
		for i in range(1,len(tan_taylor_coeffs)):
			sum = sum + (tan_taylor_coeffs[i]*x**(i+1))*((np.arctan(taylor_approx(a-i,50))/np.pi)+0.5)     #1/(1+np.exp(1.8*(i-a)))    #  
	else:
		for i in range(1,len(tan_taylor_coeffs)):
			sum = sum + tan_taylor_coeffs[i]*x**(i+1)
	return sum	

def sigmoid(t, args):
	return (0.5*np.arctan(taylor_approx(args[0]*(t*10**10-args[1]), args[2]))/(np.pi/2)+0.5)*((np.arctan(1000*args[0]*(t*10**10-args[1]))/np.pi)+0.5)+(0.5*np.arctan(taylor_approx(args[0]*(t*10**10-args[1]), args[3]))/(np.pi/2)+0.5)*((np.arctan(-1000*args[0]*(t*10**10-args[1]))/np.pi)+0.5)		  
	

input_initial = 	[0.8,  10, 3, 3,  -0.8, 12, 3, 3]
input_lower_bound = [0.1,  0,  0, 0, -5, 0, 0, 0]
input_upper_bound = [5,25, 6,6,  -0.1,  25,6,6]

parameter_names = ['left_steepness', 'left_shift', 'left_curvature_left', 'left_curvature_right', 'right_steepness', 'right_shift', 'right_curvature_left', 'right_curvature_right']

output_initial = 	 input_initial
output_lower_bound = input_lower_bound
output_upper_bound = input_upper_bound

input_shift = 1
input_length = 5
output_shift = 5
output_length = 1
