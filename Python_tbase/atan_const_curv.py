import numpy as np

tan_taylor_coeffs = [	1.000000000000000, 0, 0.333333333333333, 0, 0.133333333333333, 
						0, 0.053968253968254, 0, 0.021869488536155, 0, 0.008863235529902,
						0, 0.003592128036572, 0, 0.001455834387051, 0, 0.000590027440946 ]
						
tan_taylor_coeffs_odd = [	1.000000000000000, 0.333333333333333, 0.133333333333333, 
						0.053968253968254, 0.021869488536155, 0.008863235529902,
						0.003592128036572, 0.001455834387051, 0.000590027440946 ]
					
def taylor_approx(x,a):
	sum = x
	if a < len(tan_taylor_coeffs_odd):
		for i in range(1,len(tan_taylor_coeffs_odd)):
			sum = sum + (tan_taylor_coeffs_odd[i]*x**(2*i+1))*((np.arctan(taylor_approx(a-i,50))/np.pi)+0.5)     #1/(1+np.exp(1.8*(i-a)))    #  
	else:
		for i in range(1,len(tan_taylor_coeffs_odd)):
			sum = sum + tan_taylor_coeffs_odd[i]*x**(2*i+1)
	return sum
	


def sigmoid(t, args):
	return (0.5*np.arctan(taylor_approx(args[0]*(t*10**10-args[1]), args[2]))/(np.pi/2)+0.5)*((np.arctan(1000*args[0]*(t*10**10-args[1]))/np.pi)+0.5)+(0.5*np.arctan(taylor_approx(args[0]*(t*10**10-args[1]), 3))/(np.pi/2)+0.5)*((np.arctan(-1000*args[0]*(t*10**10-args[1]))/np.pi)+0.5)		  	
	

input_initial = 	[0.7,  11.3, 1,  -0.7, 11.7, 1] #t4_u
input_lower_bound = [0.2, 0, 0, -2, 0, 0]
input_upper_bound = [2,27, 10,  -0.2,  27, 10]

output_initial = 	 input_initial
output_lower_bound = input_lower_bound
output_upper_bound = input_upper_bound

parameter_names = ['left_steepness', 'left_shift', 'left_curv', 'right_steepness', 'right_shift', 'right_curv'];

input_shift = 1
input_length = 4
output_shift = 4
output_length = 1
