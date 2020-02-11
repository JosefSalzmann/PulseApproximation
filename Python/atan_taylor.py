import numpy as np
from scipy.special import gamma, factorial

tan_taylor_coeffs = [	1.000000000000000, 0, 0.333333333333333, 0, 0.133333333333333, 
						0, 0.053968253968254, 0, 0.021869488536155, 0, 0.008863235529902,
						0, 0.003592128036572, 0, 0.001455834387051, 0, 0.000590027440946 ]
						
tan_taylor_coeffs_odd = [	1.000000000000000, 0.333333333333333, 0.133333333333333, 
						0.053968253968254, 0.021869488536155, 0.008863235529902,
						0.003592128036572, 0.001455834387051, 0.000590027440946 ]

def taylor_approx_a(x,a):
	sum = x	
	if a < len(tan_taylor_coeffs_odd):
		for i in range(1,len(tan_taylor_coeffs_odd)):
			sum = sum + (tan_taylor_coeffs_odd[i]*x**(2*i+1))*((np.arctan(taylor_approx(1*(a-i),50))/np.pi)+0.5)     
	else:
		for i in range(1,len(tan_taylor_coeffs_odd)):
			sum = sum + tan_taylor_coeffs_odd[i]*x**(2*i+1)
	print(str(sum));
	return sum	

	

def taylor_approx(t,a):
	sum = t
	for i in range(1,len(tan_taylor_coeffs_odd)):		
		if np.floor(a) > i:
			sum = sum + (tan_taylor_coeffs_odd[i]*t**(2*i+1));
		elif np.floor(a) == i:
			x = np.absolute(t);
			a = 1-(a - i);
			diff_derivate = gamma(2*i+2)/gamma(2*i+2-a)*x**(2*i+1-a);
			#print("diff_derivate: " + str(diff_derivate));
			new_part = (diff_derivate-a*(2*i+1)*x**(2*i))*tan_taylor_coeffs_odd[i];
			sum = sum + new_part*(np.heaviside(t,0)*2-1);
			#print("t: " + str(t) + ", " + str(new_part*(np.heaviside(t,0)*2-1)))#str((np.heaviside(t,0.5)*2-1)));
	return sum
	
def sigmoid(t, args):
	return (0.5*np.arctan(taylor_approx(args[0]*(t*10**10-args[1]), args[2]))/(np.pi/2)+0.5)*((np.arctan(1000*args[0]*(t*10**10-args[1]))/np.pi)+0.5)+(0.5*np.arctan(taylor_approx(args[0]*(t*10**10-args[1]), args[3]))/(np.pi/2)+0.5)*((np.arctan(-1000*args[0]*(t*10**10-args[1]))/np.pi)+0.5)		  	
	
input_initial = 	[0.7,  11.3, 1.5, 1.5,  -0.7, 11.7, 1.5, 1.5] #t4_u
input_lower_bound = [0.3,  0,  1, 1, -2, 0, 1, 1]
input_upper_bound = [3,27, 6,6,  -0.3,  27, 6, 6]


parameter_names = ['left_steepness', 'left_shift', 'left_curvature_left', 'left_curvature_right', 'right_steepness', 'right_shift', 'right_curvature_left', 'right_curvature_right']

output_initial = 	 input_initial
output_lower_bound = input_lower_bound
output_upper_bound = [1.5,27, 6,6,  -0.6,  27, 6, 6]

input_shift = 1
input_length = 5
output_shift = 5
output_length = 1
