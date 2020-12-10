import auxiliary as aux
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from os import walk
from mpl_toolkits import mplot3d
from timeit import default_timer as timer
import sys
from scipy.special import gamma, factorial
import exp as exp
import atan_taylor as atan_taylor
import importlib
import os
import math
from matplotlib.patches import Ellipse

dir_path = os.path.dirname(os.path.realpath(__file__))

sig = importlib.import_module("atan_taylor", package=None)

tan_taylor_coeffs_odd = [	1.000000000000000, 0.333333333333333, 0.133333333333333, 
						0.053968253968254, 0.021869488536155, 0.008863235529902,
						0.003592128036572, 0.001455834387051, 0.000590027440946,
						0.000239129114244, 0.000096915379569, 0.000039278323883,
						0.000015918905069, 0.000006451689216, 0.000002614771151,
						0.000001059726832, 0.000000429491108, 0.000000174066190,
						0.000000070546369, 0.000000028591367, 0.000000011587644 ]
                    





def S(x):
	if x <= -1:
		return -1
	elif x > -1 and x <= 1:
		return x
	else:
		return 1 

def numerical_derivative(x,y):
	dy = [0.0]*(len(y)-1)
	for i in range(len(y)-1):
		dy[i] = (y[i+1]-y[i])/(x[i+1]-x[i])
	return dy

def taylor_approx_simple(x,a):
	sum = x
	for i in range(1,min(a+1,len(tan_taylor_coeffs_odd))):
		sum = sum + (tan_taylor_coeffs_odd[i]*x**(2*i+1))
	return sum	

def sigmoid_simple(t, args):
	return np.arctan(taylor_approx_simple(args[0]*(t*10**10-args[1]), args[2]))
	
	
def m(x):
	if isinstance(x, float) or isinstance(x, int):
		if x < 0:
			return 0
		elif x > 1:
			return 1
		else:
			return x
	for i in range(len(x)):
		if x[i] < 0:
			x[i] = 0
		elif x[i] > 1:
			x[i] = 1
	return x

def taylor_approx(x,a):
	sum = x	
	if a < len(tan_taylor_coeffs_odd):
		for i in range(1,len(tan_taylor_coeffs_odd)):
			sum = sum + (tan_taylor_coeffs_odd[i]*x**(2*i+1))*m(a+1-i)#((np.arctan(taylor_approx(1*(a-i),50))/np.pi)+0.5)     
	else:
		for i in range(1,len(tan_taylor_coeffs_odd)):
			sum = sum + tan_taylor_coeffs_odd[i]*x**(2*i+1)	
	return sum	

def sigmoid(t, args):
	return np.arctan(taylor_approx(args[0]*(t*10**10-args[1]), args[2]))
	
def taylor_approx_straight(x,a): #implementation with sharp edges (easier but results in worse fittings)
	sum = 0
	for i in range(0,len(tan_taylor_coeffs_odd)):
		mult = 0
		if np.floor(a+1) > i:
			mult = 1
		elif np.floor(a+1) == i:
			mult = a+1-i
		
		#print("i: " + str(i))
		#print("a+1: " + str(a+1))
		#print("mult: " + str(mult) + "\n") 
		
		sum = sum + (tan_taylor_coeffs_odd[i]*x**(2*i+1))*mult  

	return sum

def sigmoid_straight(t, args):
	return np.arctan(taylor_approx(args[0]*(t*10**10-args[1]), args[2]))
	
	
def taylor_approx_der(t,a):
	sum = t
	for i in range(1,len(tan_taylor_coeffs_odd)):		
		if np.floor(a) > i:
			sum = sum + (tan_taylor_coeffs_odd[i]*t**(2*i+1))
		elif np.floor(a) == i:
			x = np.absolute(t)
			a = 1-(a - i)
			diff_derivate = gamma(2*i+2)/gamma(2*i+2-a)*x**(2*i+1-a)
			#print("diff_derivate: " + str(diff_derivate))
			new_part = (diff_derivate-a*(2*i+1)*x**(2*i))*tan_taylor_coeffs_odd[i]
			sum = sum + new_part*(np.heaviside(t,0)*2-1)
			#print("t: " + str(t) + ", " + str(new_part*(np.heaviside(t,0)*2-1)))#str((np.heaviside(t,0.5)*2-1)))
	return sum

def sigmoid_frac_der(t, args):
	return (0.5*np.arctan(taylor_approx_der(args[0]*(t*10**10-args[1]), args[2]))/(np.pi/2)+0.5)*((np.arctan(10000*args[0]*(t*10**10-args[1]))/np.pi)+0.5)+(0.5*np.arctan(taylor_approx_der(args[0]*(t*10**10-args[1]), args[3]))/(np.pi/2)+0.5)*((np.arctan(-10000*args[0]*(t*10**10-args[1]))/np.pi)+0.5)		  	

	
params_per_sig = 2
def trace_simple_exp(t, args):
	sum = 0
	for i in range(0,len(args)//2):
		sum = sum + 1/(1+np.exp(-args[2*i]*10**10*(t-args[2*i+1])))
		
	correction = 0
	if args[0] < 0 and (len(args)/params_per_sig)%2 == 0: # first edge is falling
		correction = 1
	return 1.2*(sum-(len(args)/params_per_sig)/2+correction)
	
def trace_simple_exp_wr(t, *args):
	return trace_simple_exp(t, args)

def compensation_terms(args): # Calculates the number compensation terms needed (multiples of Vdd) such that the resulting trace is always between 0 and Vdd.
	correction = 0
	if args[0] < 0 and (len(args)/exp.num_args)%2 == 0: # first edge is falling
		correction = 1
	return (len(args)/exp.num_args)//2-correction
	
	
def trace_sigmoids_exp(t, args): # Function of the whole trace. The number of sigmoids depends on the length of args. After evaluation the resulting value is compensated by the number of terms needed.
	sum = 0
	for i in range(0,len(args)//(exp.num_args)):
		sum = sum + exp.sigmoid(t,args[exp.num_args*i:exp.num_args*(i+1)])
	return Voltage*(sum-compensation_terms(args))


def compensation_terms_atan_taylor(args): # Calculates the number compensation terms needed (multiples of Vdd) such that the resulting trace is always between 0 and Vdd.
	correction = 0
	if args[0] < 0 and (len(args)/atan_taylor.num_args)%2 == 0: # first edge is falling
		correction = 1
	return (len(args)/atan_taylor.num_args)//2-correction

def trace_sigmoids_atan_taylor(t, args): # Function of the whole trace. The number of sigmoids depends on the length of args. After evaluation the resulting value is compensated by the number of terms needed.
	sum = 0
	for i in range(0,len(args)//(atan_taylor.num_args)):
		sum = sum + atan_taylor.sigmoid(t,args[atan_taylor.num_args*i:atan_taylor.num_args*(i+1)])
	return Voltage*(sum-compensation_terms_atan_taylor(args))

Voltage = 1.2
max_x = 10
length = 1000

x = [0.0]*length

for i in range(0,length):
	x[i] = -max_x + i*(2.0*max_x/length)
	


	
	
if int(sys.argv[1]) == 0:
	atan_zero = [0.0]*length
	atan_one = [0.0]*length
	atan_two = [0.0]*length
	atan_five = [0.0]*length
	atan_ten = [0.0]*length
	atan_twenty = [0.0]*length

	for i in range(0,length):
		atan_zero[i] = sigmoid_simple(x[i]/10**10, [1,0,0])	
		atan_one[i] = sigmoid_simple(x[i]/10**10, [1,0,1])
		atan_two[i] = sigmoid_simple(x[i]/10**10, [1,0,2])
		atan_five[i] = sigmoid_simple(x[i]/10**10, [1,0,5])
		atan_ten[i] = sigmoid_simple(x[i]/10**10, [1,0,10])
		atan_twenty[i] = sigmoid_simple(x[i]/10**10, [1,0,20])

	plt.cla()
	plt.clf()
	fig = plt.gcf()
	fig.set_size_inches(8, 6)

		
	linew = 1.5
		
	
	plt.plot(x,atan_zero, linewidth=linew)
	plt.plot(x,atan_one, linewidth=linew)
	plt.plot(x,atan_two, linewidth=linew)
	plt.plot(x,atan_five, linewidth=linew)
	plt.plot(x,atan_ten, linewidth=linew)
	plt.plot(x,atan_twenty, linewidth=linew)

	plt.title('')
	plt.legend(["c = 0 (equal to atan(x))","c = 1", "c = 2", "c = 5", "c = 10", "c = 20"], loc = 'center left')
	#plt.text(-10, 1, "text")
	plt.show()
elif int(sys.argv[1]) == 1:

	x_max = 10
	x = np.linspace(0,x_max, length)

	atan_zero = [0.0]*length
	atan_1 = [0.0]*length
	atan_2 = [0.0]*length
	atan_3 = [0.0]*length
	atan_4 = [0.0]*length
	atan_5 = [0.0]*length

	for i in range(0,length):
		atan_zero[i] = sigmoid(x[i]/10**10, [1,0,0])	
		atan_1[i] = sigmoid(x[i]/10**10, [1,0,0.1])
		atan_2[i] = sigmoid(x[i]/10**10,  [1,0,0.5])
		atan_3[i] = sigmoid(x[i]/10**10,  [1,0,1])
		# atan_4[i] = sigmoid(x[i]/10**10,  [1,0,0.3])
		#atan_5[i] = sigmoid(x[i]/10**10, [1,0,1])

	plt.cla()
	plt.clf()
	fig = plt.gcf()
	fig.set_size_inches(8, 6)

		
	linew = 1.5
		
	plt.plot(x,atan_zero, linewidth=linew)
	plt.plot(x,atan_1, linewidth=linew)
	plt.plot(x,atan_2, linewidth=linew)
	plt.plot(x,atan_3, linewidth=linew)
	# plt.plot(x,atan_4, linewidth=linew)
	#plt.plot(x,atan_5,'m-', linewidth=linew)

	plt.title('')
	plt.legend(["c = 0","c = 0.1", "c = 0.5", "c = 1"], loc = 'center right')
	#plt.text(-10, 1, "text")
	plt.show() 

elif int(sys.argv[1]) == 2:
	atan_zero_zero = [0.0]*length
	atan_pfive_five = [0.0]*length


	for i in range(0,length):
		atan_zero_zero[i] = np.pi*sig.sigmoid(x[i]/10**10, [1,0,0, 0])-np.pi/2
		atan_pfive_five[i] = np.pi*sig.sigmoid(x[i]/10**10, [1,0,0.1, 5])-np.pi/2

	plt.cla()
	plt.clf()
	fig = plt.gcf()
	fig.set_size_inches(8, 6)

		
	linew = 1.5
		
	plt.plot(x,atan_zero_zero, linewidth=linew)
	plt.plot(x,atan_pfive_five, linewidth=linew)
	#plt.plot(x,atan_two,'b-', linewidth=linew)
	#plt.plot(x,atan_ten,'y-', linewidth=linew)

	plt.title('')
	plt.legend(["c = 0, d = 0","c = 5, d = 0.1"], loc = 'center left')
	#plt.text(-10, 1, "text")
	plt.show()
elif int(sys.argv[1]) == 3:
	atan = [0.0]*length
	straight = [0.0]*length


	for i in range(0,length):
		atan[i] = np.arctan(x[i])
		if(x[i] < -np.pi/2):
			straight[i] = -np.pi/2
		elif(x[i] > np.pi/2):
			straight[i] = np.pi/2
		else:
			straight[i] = x[i]

	plt.cla()
	plt.clf()
	fig = plt.gcf()
	fig.set_size_inches(8, 6)

		
	linew = 1
		
	plt.plot(x,atan,'r-', linewidth=linew)
	plt.plot(x,straight,'g-', linewidth=linew)

	plt.title('')
	plt.legend(["atan(x)","s(x)"], loc = 'center left')
	#plt.text(-10, 1, "text")
	plt.show()
elif int(sys.argv[1]) == 4:

	path = dir_path + "/../WaveformData/t4_traces/inv_t4_invSim_Traces.dat"
	data = aux.read_file(path, 100000)
	
	first_der = [[0.0]*len(data[0]) for i in range(3)]
	mult = 2000
	for i in range(1,len(first_der[0])):
		first_der[1][i] = (data[1][i]-data[1][i-1])/(data[0][i]-data[0][i-1])*10**-9
		first_der[2][i] = (data[2][i]-data[2][i-1])/(data[0][i]-data[0][i-1])*10**-9
	
	filter_size = 10
	percent_in_range = 0.001
	filter = [0.0]*filter_size
	next_edge_rising = True
	switching_points = []
	min_height = 0.05
	for i in range(0, len(data[0])):
		filter[i%filter_size] = first_der[1][i]
		if (np.sum(filter) > min_height*filter_size and next_edge_rising) or (np.sum(filter) < -1*min_height*filter_size and not next_edge_rising):
			all_same_sign = True			
			for j in range(1,filter_size):
				if filter[j]*filter[0] < 0: # not the same sign
					all_same_sign = False
					break
			if all_same_sign:
				all_in_range = True
				abs_filter = np.absolute(filter)
				for j in range(1,filter_size):
					if abs_filter[j] < abs_filter[0]*(1-percent_in_range) or abs_filter[j] > abs_filter[0]*(1+percent_in_range):
						all_in_range = False
						break
				if all_in_range:
					next_edge_rising = not next_edge_rising
					switching_points.append(i+filter_size)
	#print(switching_points)		

	approx = [0.0]*len(data[0])
	args = [0.0]*(2*len(switching_points))
	for i in range(0,len(switching_points)):
		args[2*i] = 0.7*(-1)**(i+0)
		args[2*i+1] = data[0][switching_points[i]]
		
	input_init = args
	lower_bound = [0.0]*len(args)
	upper_bound = [0.0]*len(args)
	lower_bound[1] = 0
	upper_bound[len(args)-1] = data[0][len(data[0])-1]
	for i in range(0,len(switching_points)):
		lower_bound[2*i] = args[2*i]-2.69
		upper_bound[2*i] = args[2*i]+2.69
	for i in range(0,len(switching_points)-1):
		lower_bound[2*i+1+2] = (args[2*i+1]+args[2*i+3])/2
		upper_bound[2*i+1] = (args[2*i+1]+args[2*i+3])/2
	for i in range(0,len(first_der[0])):
		approx[i] = trace_simple_exp(data[0][i], args)	
	
	data_reduced = aux.read_file(path, 10000)
	#input_params = optimize.curve_fit(trace_simple_exp_wr, data_reduced[0], data_reduced[1], input_init, bounds = (lower_bound, upper_bound), maxfev=5000)
	
	#print(input_params)
	
	#for i in range(0,len(first_der[0])):
		#approx[i] = trace_simple_exp(data[0][i], input_params[0])
	
	plt.cla()
	plt.clf()
	fig = plt.gcf()
	fig.set_size_inches(16, 4)

		
	linew = 1.5
		
	plt.plot(data[0],data[1],'r-', linewidth=linew)
	#plt.plot(data[0],data[2],'g-', linewidth=linew)
	plt.plot(data[0],approx,'b-', linewidth=linew)
	#plt.plot(data[0],first_der[2],'g--', linewidth=linew)

	plt.ylabel("Voltage [V]")
	plt.xlabel("Time [s]")
	plt.title('')
	plt.legend(["Physical Trace","Initial Guess"], loc = 'center left')
	plt.show()

elif int(sys.argv[1]) == 5:
	first_approx = [0.0]*length
	second_approx = [0.0]*length
	third_approx = [0.0]*length
	fourth_approx = [0.0]*length
	tan = [0.0]*length


	for i in range(0,length):
		x[i] = x[i]/7
		first_approx[i] = x[i]
		second_approx[i] = first_approx[i] + (x[i]**3)/3
		third_approx[i] = second_approx[i] + (x[i]**5)*0.133333333333333
		fourth_approx[i] = third_approx[i] + (x[i]**7)*0.053968253968254
		tan[i] = np.tan(x[i])

	plt.cla()
	plt.clf()
	fig = plt.gcf()
	fig.set_size_inches(8, 6)

	linew = 1
	#fig, ax = plt.subplots()
	#ax.axhline(y=0, color='k')
	#ax.axvline(x=0, color='k')
	
		
	plt.plot(x,tan,'k-', linewidth=linew)
	plt.plot(x,first_approx,'r-', linewidth=linew)
	plt.plot(x,second_approx,'g-', linewidth=linew)
	plt.plot(x,third_approx,'b-', linewidth=linew)
	plt.plot(x,fourth_approx,'y-', linewidth=linew)
	

	plt.title('Taylor approximation of tan(x)')
	plt.legend(["tan(x)", "Approx. of 1st order", "Approx. of 3rd order", "Approx. of 5th order", "Approx. of 7th order"], loc = 'lower center')
	#plt.text(-10, 1, "text")
	plt.show()
elif int(sys.argv[1]) == 6:

	path = dir_path + "/../WaveformData/t4_d/inv_t4_d000740000Traces.dat"
	data = aux.read_file(path, 10000)

	plt.cla()
	plt.clf()

	fig = plt.gcf()
	fig.set_size_inches(8, 6)

	output_first_edge_index = 0
	input_second_edge_index = 0
	output_second_edge_index = 0

	for i in range(0, len(data[0])):
		data[0][i]*=10**9
		if output_first_edge_index == 0 and data[2][i] >= 0.6:
			output_first_edge_index = i
		if output_first_edge_index != 0 and input_second_edge_index == 0 and data[1][i] >= 0.6:
			input_second_edge_index = i
		if output_first_edge_index != 0 and output_second_edge_index == 0 and data[2][i] <= 0.6:
			output_second_edge_index = i

	linew = 1.5
	
	input_steepness = (data[1][input_second_edge_index+1]-data[1][input_second_edge_index-1]) / (data[0][input_second_edge_index+1]-data[0][input_second_edge_index-1])

	input_steepness_plt = [0.0]*1372
	for i in range(0, len(input_steepness_plt)):
		input_steepness_plt[i] = data[0][i]*input_steepness


	output_steepness = (data[2][output_first_edge_index+1]-data[2][output_first_edge_index-1]) / (data[0][output_first_edge_index+1]-data[0][output_first_edge_index-1])
	output_steepness_plt = [0.0]*1372
	for i in range(0, len(output_steepness_plt)):
		output_steepness_plt[i] = data[0][i]*output_steepness
	
	plt.plot(data[0],data[1],'r-', linewidth=linew)
	plt.plot(data[0],data[2],'g-', linewidth=linew)
	plt.plot(data[0][int(output_first_edge_index-len(output_steepness_plt)/2):int(output_first_edge_index+len(output_steepness_plt)/2)],output_steepness_plt,'k--', linewidth=1.5)
	plt.plot(data[0][int(input_second_edge_index-len(input_steepness_plt)/2):int(input_second_edge_index+len(input_steepness_plt)/2)],input_steepness_plt,'k--', linewidth=1.5)
	plt.plot([data[0][output_first_edge_index]]*2, [0.35,0.6],'k-', linewidth=1.2)
	plt.plot([data[0][input_second_edge_index]]*2, [0.35,0.6],'k-', linewidth=1.2)
	
	plt.arrow(data[0][output_first_edge_index+400], 0.4, data[0][input_second_edge_index-150]-data[0][output_first_edge_index+400], 0, width=0.0005, head_width=data[0][150]/2, head_length=data[0][150], fc='k', ec='k', linestyle=('-'), capstyle='round')
	plt.arrow(data[0][input_second_edge_index-400], 0.4, -(data[0][input_second_edge_index-150]-data[0][output_first_edge_index+400]), 0, width=0.0005, head_width=data[0][150]/2, head_length=data[0][150], fc='k', ec='k', linestyle=('-'), capstyle='round')

	plt.text(data[0][(input_second_edge_index+output_first_edge_index)//2-50],0.41,r'$T$')

	plt.text(data[0][output_first_edge_index-800],0.6,r'$so_{n-1}$')
	plt.text(data[0][input_second_edge_index-470],0.6,r'$si_{n}$')

	plt.ylabel("Voltage [V]")
	plt.xlabel("Time [ns]")
	plt.title('')
	plt.legend(["Input", "Output"], loc = 'center left')
	#plt.text(-10, 1, "text")
	plt.show()

elif int(sys.argv[1]) == 7:

	path = dir_path + "/../WaveformData/t4_d/inv_t4_d000740000Traces.dat"
	data = aux.read_file(path, 10000)

	plt.cla()
	plt.clf()

	fig = plt.gcf()
	fig.set_size_inches(8, 6)

	output_first_edge_index = 0
	input_second_edge_index = 0
	output_second_edge_index = 0

	for i in range(0, len(data[0])):
		data[0][i]*=10**9
		if output_first_edge_index == 0 and data[2][i] >= 0.6:
			output_first_edge_index = i
		if output_first_edge_index != 0 and input_second_edge_index == 0 and data[1][i] >= 0.6:
			input_second_edge_index = i
		if output_first_edge_index != 0 and output_second_edge_index == 0 and data[2][i] <= 0.6:
			output_second_edge_index = i

	linew = 1.5
	
	output_steepness = (data[2][output_second_edge_index+1]-data[2][output_second_edge_index-1]) / (data[0][output_second_edge_index+1]-data[0][output_second_edge_index-1])
	output_steepness_plt = [0.0]*1100
	for i in range(0, len(output_steepness_plt)):
		output_steepness_plt[i] = 1.2+data[0][i]*output_steepness
	
	plt.plot(data[0],data[1],'r-', linewidth=linew)
	plt.plot(data[0],data[2],'g-', linewidth=linew)
	plt.plot(data[0][int(output_second_edge_index-len(output_steepness_plt)/2):int(output_second_edge_index+len(output_steepness_plt)/2)],output_steepness_plt,'k--', linewidth=1.5)
	plt.plot([data[0][input_second_edge_index]]*2, [0.35,0.6],'k-', linewidth=1.2)
	plt.plot([data[0][output_second_edge_index]]*2, [0.35,0.6],'k-', linewidth=1.2)
	
	plt.arrow(data[0][input_second_edge_index+300], 0.4, data[0][output_second_edge_index-150]-data[0][input_second_edge_index+300], 0, width=0.0005, head_width=data[0][150]/2, head_length=data[0][150], fc='k', ec='k', linestyle=('-'), capstyle='round')
	plt.arrow(data[0][output_second_edge_index-300], 0.4, -(data[0][output_second_edge_index-150]-data[0][input_second_edge_index+300]), 0, width=0.0005, head_width=data[0][150]/2, head_length=data[0][150], fc='k', ec='k', linestyle=('-'), capstyle='round')

	plt.text(data[0][(input_second_edge_index+output_second_edge_index)//2-50],0.41,r'$D$')
	plt.text(data[0][output_second_edge_index+50],0.6,r'$so_{n}$')

	plt.ylabel("Voltage [V]")
	plt.xlabel("Time [ns]")
	plt.title('')
	plt.legend(["Input", "Output"], loc = 'center left')
	#plt.text(-10, 1, "text")
	plt.show()
elif int(sys.argv[1]) == 8:

	path = dir_path + "/../WaveformData/traces_for_pictures/inv_t4_u000740000_001220000_000730000.dat"
	data = aux.read_file(path, 10000)

	plt.cla()
	plt.clf()


	input_edges = [0 for i in range(4)]
	edge_cnt = 0
	for i in range(len(data[1])):
		# print(i)
		# print(data[1][i])
		data[0][i]*=10**9
		if data[1][i]>0.59 and data[1][i] < 0.61:
			if (edge_cnt > 0 and input_edges[edge_cnt-1]+20 < i) or edge_cnt == 0:
				input_edges[edge_cnt] = i
				edge_cnt+=1

	fig = plt.gcf()
	fig.set_size_inches(10, 3)
	linew = 1.5
	plt.plot(data[0],data[1],'r-', linewidth=linew)
	plt.plot(data[0],data[2],'g-', linewidth=linew)

	arrow_size = 20
	plt.arrow(data[0][input_edges[0]], 0.6, data[0][input_edges[1]]-data[0][input_edges[0]]-data[0][arrow_size], 0, width=0.0005, head_width=data[0][arrow_size]/2, head_length=data[0][arrow_size], fc='k', ec='k', linestyle=('-'), capstyle='round')
	plt.arrow(data[0][input_edges[1]], 0.6, -(data[0][input_edges[1]]-data[0][input_edges[0]]-data[0][arrow_size]), 0, width=0.0005, head_width=data[0][arrow_size]/2, head_length=data[0][arrow_size], fc='k', ec='k', linestyle=('-'), capstyle='round')

	plt.arrow(data[0][input_edges[1]], 0.6, data[0][input_edges[2]]-data[0][input_edges[1]]-data[0][arrow_size], 0, width=0.0005, head_width=data[0][arrow_size]/2, head_length=data[0][arrow_size], fc='k', ec='k', linestyle=('-'), capstyle='round')
	plt.arrow(data[0][input_edges[2]], 0.6, -(data[0][input_edges[2]]-data[0][input_edges[1]]-data[0][arrow_size]), 0, width=0.0005, head_width=data[0][arrow_size]/2, head_length=data[0][arrow_size], fc='k', ec='k', linestyle=('-'), capstyle='round')
	
	plt.arrow(data[0][input_edges[2]], 0.6, data[0][input_edges[3]]-data[0][input_edges[2]]-data[0][arrow_size], 0, width=0.0005, head_width=data[0][arrow_size]/2, head_length=data[0][arrow_size], fc='k', ec='k', linestyle=('-'), capstyle='round')
	plt.arrow(data[0][input_edges[3]], 0.6, -(data[0][input_edges[3]]-data[0][input_edges[2]]-data[0][arrow_size]), 0, width=0.0005, head_width=data[0][arrow_size]/2, head_length=data[0][arrow_size], fc='k', ec='k', linestyle=('-'), capstyle='round')
	

	plt.text(data[0][(input_edges[0]+input_edges[1])//2-10],0.63,r'$T_{A}$',fontsize=12)
	plt.text(data[0][(input_edges[1]+input_edges[2])//2-10],0.63,r'$T_{B}$',fontsize=12)
	plt.text(data[0][(input_edges[2]+input_edges[3])//2-10],0.63,r'$T_{C}$',fontsize=12)

	plt.ylabel("Voltage [V]")
	plt.xlabel("Time [ns]")
	plt.title('')
	plt.legend(["Input", "Output"], loc = 'center left')
	plt.show()

elif int(sys.argv[1]) == 9:

	path = dir_path + "/../WaveformData/traces_for_pictures/inv_t4_u000460000_001220000_000730000.dat"
	data = aux.read_file(path, 10000)

	plt.cla()
	plt.clf()


	input_edges = [0 for i in range(4)]
	edge_cnt = 0
	for i in range(len(data[1])):
		# print(i)
		# print(data[1][i])
		data[0][i]*=10**9
		if data[1][i]>0.59 and data[1][i] < 0.61:
			if (edge_cnt > 0 and input_edges[edge_cnt-1]+20 < i) or edge_cnt == 0:
				input_edges[edge_cnt] = i
				edge_cnt+=1

	fig = plt.gcf()
	fig.set_size_inches(10, 3)
	linew = 1.5
	plt.plot(data[0],data[1],'r-', linewidth=linew)
	plt.plot(data[0],data[2],'g-', linewidth=linew)

	arrow_size = 20
	plt.arrow(data[0][input_edges[0]], 0.6, data[0][input_edges[1]]-data[0][input_edges[0]]-data[0][arrow_size], 0, width=0.0005, head_width=data[0][arrow_size]/2, head_length=data[0][arrow_size], fc='k', ec='k', linestyle=('-'), capstyle='round')
	plt.arrow(data[0][input_edges[1]], 0.6, -(data[0][input_edges[1]]-data[0][input_edges[0]]-data[0][arrow_size]), 0, width=0.0005, head_width=data[0][arrow_size]/2, head_length=data[0][arrow_size], fc='k', ec='k', linestyle=('-'), capstyle='round')

	plt.arrow(data[0][input_edges[1]], 0.6, data[0][input_edges[2]]-data[0][input_edges[1]]-data[0][arrow_size], 0, width=0.0005, head_width=data[0][arrow_size]/2, head_length=data[0][arrow_size], fc='k', ec='k', linestyle=('-'), capstyle='round')
	plt.arrow(data[0][input_edges[2]], 0.6, -(data[0][input_edges[2]]-data[0][input_edges[1]]-data[0][arrow_size]), 0, width=0.0005, head_width=data[0][arrow_size]/2, head_length=data[0][arrow_size], fc='k', ec='k', linestyle=('-'), capstyle='round')
	
	plt.arrow(data[0][input_edges[2]], 0.6, data[0][input_edges[3]]-data[0][input_edges[2]]-data[0][arrow_size], 0, width=0.0005, head_width=data[0][arrow_size]/2, head_length=data[0][arrow_size], fc='k', ec='k', linestyle=('-'), capstyle='round')
	plt.arrow(data[0][input_edges[3]], 0.6, -(data[0][input_edges[3]]-data[0][input_edges[2]]-data[0][arrow_size]), 0, width=0.0005, head_width=data[0][arrow_size]/2, head_length=data[0][arrow_size], fc='k', ec='k', linestyle=('-'), capstyle='round')
	

	plt.text(data[0][(input_edges[0]+input_edges[1])//2-35],0.63,r'$T_{A}$',fontsize=12)
	plt.text(data[0][(input_edges[1]+input_edges[2])//2-10],0.63,r'$T_{B}$',fontsize=12)
	plt.text(data[0][(input_edges[2]+input_edges[3])//2-10],0.63,r'$T_{C}$',fontsize=12)

	plt.ylabel("Voltage [V]")
	plt.xlabel("Time [ns]")
	plt.title('')
	plt.legend(["Input", "Output"], loc = 'center left')
	plt.show()
elif int(sys.argv[1]) == 10:

	path = dir_path + "/../WaveformData/t4_d/inv_t4_d000500000Traces.dat"
	data = aux.read_file(path, 10000)

	plt.cla()
	plt.clf()

	first_params = [1.16094, 13.21017]
	second_params = [-1.28307, 14.98119]
	all_params = [first_params[0], first_params[1], second_params[0], second_params[1]]
	first_edge = [trace_sigmoids_exp(data[0][i], first_params) for i in range(len(data[0]))]
	second_edge = [trace_sigmoids_exp(data[0][i], second_params) for i in range(len(data[0]))]
	both_edges = [trace_sigmoids_exp(data[0][i], all_params) for i in range(len(data[0]))]

	fig = plt.gcf()
	fig.set_size_inches(10, 3)
	linew = 1.5
	plt.plot(data[0],data[2],'g-', linewidth=linew)
	plt.plot(data[0],first_edge,'r-', linewidth=linew)
	plt.plot(data[0],second_edge,'b-', linewidth=linew)
	plt.plot(data[0],both_edges,'g--', linewidth=linew)


	plt.ylabel("Voltage [V]")
	plt.xlabel("Time [s]")
	plt.title('')
	plt.legend(["Physical Trace", "Rising Edge Approximation", "Falling Edge Approximation", "Sum of Edges"], loc = 'center left')
	plt.show()

elif int(sys.argv[1]) == 11:


	# Input parameters
	# steepness,shift
	# 1.17713,10.59539
	# -1.28775,12.25776
	# Input compensations terms,1.0
	# Output parameters
	# steepness,shift
	# ..\\WaveformData\\inv_t4_u000406000Traces.dat

	path = dir_path + "/../WaveformData/inv_t4_u000406000Traces.dat"
	data = aux.read_file(path, 10000)

	plt.cla()
	plt.clf()

	first_params = [1.17713, 10.59539]
	second_params = [-1.28775, 12.25776]
	all_params = [first_params[0], first_params[1], second_params[0], second_params[1]]
	first_edge = [trace_sigmoids_exp(data[0][i], first_params) for i in range(len(data[0]))]
	second_edge = [trace_sigmoids_exp(data[0][i], second_params) for i in range(len(data[0]))]
	both_edges = [trace_sigmoids_exp(data[0][i], all_params) for i in range(len(data[0]))]

	output_fitting_err = aux.calc_rms_error_func(trace_sigmoids_exp, all_params, data[0], data[1])/Voltage

	fig = plt.gcf()
	fig.set_size_inches(10, 3)
	linew = 1.5
	plt.plot(data[0],data[1],'g-', linewidth=linew)
	plt.plot(data[0],first_edge,'r-', linewidth=linew)
	plt.plot(data[0],second_edge,'b-', linewidth=linew)
	plt.plot(data[0],both_edges,'g--', linewidth=linew)
	plt.text(0, Voltage/8, "Fitting Error: " + str(round(output_fitting_err,5)),family="monospace")

	plt.ylabel("Voltage [V]")
	plt.xlabel("Time [s]")
	plt.title('')
	plt.legend(["Physical Trace", "Rising Edge Approximation", "Falling Edge Approximation", "Sum of Edges"], loc = 'center left')
	plt.show()
elif int(sys.argv[1]) == 12:

	path = dir_path + "/../WaveformData/inv_t4_u000406000Traces.dat"
	data = aux.read_file(path, 10000)

	plt.cla()
	plt.clf()



	fig = plt.gcf()
	fig.set_size_inches(10, 3)
	linew = 1.5
	plt.plot(data[0],data[1],'g-', linewidth=linew)


	plt.ylabel("Voltage [V]")
	plt.xlabel("Time [s]")
	plt.title('')
	plt.legend(["Physical Trace"], loc = 'center left')
	plt.show()
elif int(sys.argv[1]) == 13:

	path = dir_path + "/../WaveformData/t4_traces/inv_t4_invSim_Traces.dat"
	data = aux.read_file(path, 10000)


	first_der = [0.0]*len(data[0]) 
	for i in range(1,len(first_der)): # Calculate the first derivative of the given trace in order to find the turning points of the trace so that the number of sigmoids and their initial position can be determined.
		if (data[0][i]-data[0][i-1]) != 0:
			first_der[i] = (data[1][i]-data[1][i-1])/(data[0][i]-data[0][i-1])*10**-9*0.2
		else: # dat-files sometimes contain the same time stamp twice...
			first_der[i] = first_der[i-1] # in that case the derivative is just interpolated using the previous value.

	filter_size = 10
	percent_in_range = 0.01
	filter = [0.0]*filter_size
	starts_at_Vdd = 0
	next_edge_rising = True
	if data[1][0] > Voltage/2: # Identify if the trace starts with a falling or a rising edge.
		starts_at_Vdd = 1
		next_edge_rising = False
	switching_points = []
	min_height = 0.05
	for i in range(0, len(data[1])-filter_size): # Iterate through the derivative array and identify small portions (size of the filter) where the gradient is constant and not zero.
		filter = first_der[i:i+filter_size]
		if (np.sum(filter) > min_height*filter_size and next_edge_rising) or (np.sum(filter) < -1*min_height*filter_size and not next_edge_rising): # Check if the observed portion is not zero.
			all_same_sign = True			
			for j in range(1,filter_size): # Check if all items in the observed part of the array have the same sign.
				if filter[j]*filter[0] < 0:
					all_same_sign = False
					break
			if all_same_sign:
				all_in_range = True
				abs_filter = np.absolute(filter)
				for j in range(1,filter_size):
					if abs_filter[j] < abs_filter[0]*(1-percent_in_range) or abs_filter[j] > abs_filter[0]*(1+percent_in_range): # Check if all items are close enough to each other.
						all_in_range = False
						break
				if all_in_range: # A turning point is found.
					next_edge_rising = not next_edge_rising # The next turning point has to have the opposite direction. 
					switching_points.append(i+filter_size) # The current position is noted and will be used as inital guess for the fitting algorithm.

	plt.cla()
	plt.clf()



	fig = plt.gcf()
	fig.set_size_inches(16, 4)

	linew = 1.5
	plt.plot(data[0],data[1],'r-', linewidth=linew)

	for i in range(len(switching_points)):
		point = switching_points[i]+filter_size//2
		# point2 = switching_points[i+1]+filter_size//2
		d = 0.15
		plt.plot([data[0][point],data[0][point]],[data[1][point]-d, data[1][point]+d],'k-', linewidth=linew)
		# print(round((data[0][point]+data[0][point2])*0.5*10**10,3))



	plt.ylabel("Voltage [V]")
	plt.xlabel("Time [s]")
	plt.title('')
	plt.legend(["Physical Trace"], loc = 'center left')
	plt.show()
elif int(sys.argv[1]) == 14:

	path = dir_path + "/../WaveformData/t4_traces/inv_t4_invSim_Traces.dat"
	data = aux.read_file(path, 10000)

	params = [1.31162, 17.67323, -1.44738, 23.27652, 1.21693, 27.88467, 
			 -1.28902, 33.93985, 1.1308, 35.06268, -1.65124, 42.98049, 
			  1.28576, 48.70933, -1.47807, 53.81109, 1.25413, 58.91224, 
			  -1.31241, 63.79841, 1.18278, 68.09467, -1.63224, 75.03277]

	
	trace_fitting = [trace_sigmoids_exp(data[0][i], params) for i in range(len(data[0]))]
	error = [trace_fitting[i]-data[1][i] for i in range(len(data[0]))]

	output_fitting_err = aux.calc_rms_error_func(trace_sigmoids_exp, params, data[0], data[1])/Voltage


	plt.cla()
	plt.clf()
	fig = plt.gcf()
	fig.set_size_inches(16, 4)

	linew = 1.5
	plt.plot(data[0],data[1],'r-', linewidth=linew)
	plt.plot(data[0],trace_fitting,'r--', linewidth=linew)
	plt.plot(data[0],error,'k-', linewidth=linew)

	plt.ylabel("Voltage [V]")
	plt.xlabel("Time [s]")
	plt.title('')
	plt.text(0, Voltage/8, "Fitting  RMSE: " + str(round(0.0109022,5)),family="monospace")
	plt.legend(["Physical Trace", "Fitting", "Error"], loc = 'center left')
	plt.show()

elif int(sys.argv[1]) == 15:

	x_max = 3.5

	n = 2000

	x = np.linspace(-x_max, x_max, n)

	atan = [(2/np.pi)*np.arctan((np.pi/2)*x[i]) for i in range(n)]
	erf = [math.erf((np.sqrt(np.pi)/2)*x[i]) for i in range(n)]
	xsqrt = [x[i]/np.sqrt(1+x[i]**2) for i in range(n)]
	tanh = [np.tanh(x[i]) for i in range(n)]
	xabs = [x[i]/(1+np.abs(x[i])) for i in range(n)]
	sx = [S(x[i]) for i in range(n)]

	plt.cla()
	plt.clf()
	fig = plt.gcf()
	fig.set_size_inches(8, 4)

	linew = 1.5
	plt.plot(x,atan, linewidth=linew, label=r'$\frac{2}{\pi} arctan(\frac{\pi}{2}x)$')
	plt.plot(x,erf, linewidth=linew, label=r'$erf(\frac{\sqrt{\pi}}{2} x)$')
	plt.plot(x,xsqrt, linewidth=linew, label=r'$\frac{x}{\sqrt{1+x^2}}$')
	plt.plot(x,tanh, linewidth=linew, label=r'$tanh(x)$')
	plt.plot(x,xabs, linewidth=linew, label=r'$\frac{x}{1+|x|}$')
	plt.plot(x,sx, linewidth=linew, label=r'$S(x)$')
	# plt.plot([-x_max,x_max],[1,1], 'k-', linewidth=linew/2)
	# plt.plot([-x_max,x_max],[-1,-1], 'k-', linewidth=linew/2)
	# plt.plot([-x_max,x_max],[0,0], 'k-', linewidth=linew/2)
	# plt.plot([0,0],[1,-1], 'k-', linewidth=linew/2)

	plt.title('')
	# plt.text(0, Voltage/8, "Fitting  RMSE: " + str(round(0.0109022,5)),family="monospace")
	plt.legend(loc = 'upper left')
	plt.show()

elif int(sys.argv[1]) == 16:

	path = dir_path + "/../WaveformData/t4_u/inv_t4_u000439300Traces.dat"
	data = aux.read_file(path, 10000)
	first_params = [1.17298, 10.3973]
	second_params = [-1.29483, 13.58285]
	all_params = [first_params[0], first_params[1], second_params[0], second_params[1]]
	both_edges = [trace_sigmoids_exp(data[0][i], all_params) for i in range(len(data[0]))]
	
	error = [both_edges[i]-data[1][i] for i in range(len(data[0]))]

	plt.figure()
	fig = plt.gcf()
	ax = plt.gca()
	fig.set_size_inches(10, 3)


	ax = plt.gca()

	ellipse1 = Ellipse(xy=(8*10**-10, 0.038), width=10**-9/6.5, height=1/5, 
							edgecolor='r', fc='None', lw=0.6)
	ellipse2 = Ellipse(xy=(1.195*10**-9, 0.884), width=10**-9/6.5, height=1/5, 
							edgecolor='r', fc='None', lw=0.6)
	ax.add_patch(ellipse1)
	ax.add_patch(ellipse2)
	
	linew = 1.5

	plt.plot(data[0],data[1],'g-', linewidth=linew)
	plt.plot(data[0],both_edges,'g--', linewidth=linew)
	plt.plot(data[0],error,'k-', linewidth=linew)

	plt.ylabel("Voltage [V]")
	plt.xlabel("Time [s]")
	plt.title('')
	plt.legend(["Physical Trace", "Fitting", "Error"], loc = 'center left')
	plt.show()
elif int(sys.argv[1]) == 17:
	x_max = 3
	length = 10000
	x = np.linspace(-x_max, x_max, length)
	atan_zero = [0.0]*length
	atan_one = [0.0]*length
	atan_two = [0.0]*length
	atan_five = [0.0]*length
	atan_ten = [0.0]*length
	atan_twenty = [0.0]*length

	for i in range(0,length):
		atan_zero[i] = sigmoid_simple(x[i]/10**10, [1,0,0])	
		atan_one[i] = sigmoid_simple(x[i]/10**10, [1,0,1])
		atan_two[i] = sigmoid_simple(x[i]/10**10, [1,0,2])
		atan_five[i] = sigmoid_simple(x[i]/10**10, [1,0,5])
		atan_ten[i] = sigmoid_simple(x[i]/10**10, [1,0,10])
		atan_twenty[i] = sigmoid_simple(x[i]/10**10, [1,0,20])
	
	atan_zero_first_der = numerical_derivative(x,atan_zero)
	atan_zero_second_der = numerical_derivative(x,atan_zero_first_der)

	atan_one_first_der = numerical_derivative(x,atan_one)
	atan_one_second_der = numerical_derivative(x,atan_one_first_der)

	atan_two_first_der = numerical_derivative(x,atan_two)
	atan_two_second_der = numerical_derivative(x,atan_two_first_der)

	atan_five_first_der = numerical_derivative(x,atan_five)
	atan_five_second_der = numerical_derivative(x,atan_five_first_der)

	atan_ten_first_der = numerical_derivative(x,atan_ten)
	atan_ten_second_der = numerical_derivative(x,atan_ten_first_der)

	atan_twenty_first_der = numerical_derivative(x,atan_twenty)
	atan_twenty_second_der = numerical_derivative(x,atan_twenty_first_der)

	plt.cla()
	plt.clf()
	fig = plt.gcf()
	fig.set_size_inches(8, 6)
	
	linew = 1.5
	
	plt.plot(x[0:len(x)-2],atan_zero_second_der, linewidth=linew)
	plt.plot(x[0:len(x)-2],atan_one_second_der, linewidth=linew)
	plt.plot(x[0:len(x)-2],atan_two_second_der, linewidth=linew)
	plt.plot(x[0:len(x)-2],atan_five_second_der, linewidth=linew)
	plt.plot(x[0:len(x)-2],atan_ten_second_der, linewidth=linew)
	plt.plot(x[0:len(x)-2],atan_twenty_second_der, linewidth=linew)

	plt.title('')
	plt.legend(["c = 0", "c = 1", "c = 2", "c = 5", "c = 10", "c = 20"], loc = 'upper left')
	#plt.text(-10, 1, "text")
	plt.show()
elif int(sys.argv[1]) == 18:

	path = dir_path + "/../WaveformData/t4_u_4t/inv_t4_u000460000_000500000_000670000.dat"
	data = aux.read_file(path, 10000)

	params = [0.8762, 9.69088, 0.44981, 3.62494,
			 -0.99391, 13.80773, 0.91518, 1.09903,
			 0.87097, 18.83031, 1.09763, 3.0,
			-1.0901, 26.03614, 2.5833, 1.71901]	

	
	trace_fitting = [trace_sigmoids_atan_taylor(data[0][i], params) for i in range(len(data[0]))]
	error = [trace_fitting[i]-data[1][i] for i in range(len(data[0]))]

	output_fitting_err = aux.calc_rms_error_func(trace_sigmoids_atan_taylor, params, data[0], data[1])/Voltage


	plt.cla()
	plt.clf()
	fig = plt.gcf()
	fig.set_size_inches(16, 4)

	linew = 1.5
	plt.plot(data[0][0:int((len(data[0])*(3.5/5)))],data[1][0:int((len(data[0])*(3.5/5)))],'r-', linewidth=linew)
	plt.plot(data[0][0:int((len(data[0])*(3.5/5)))],trace_fitting[0:int((len(data[0])*(3.5/5)))],'r--', linewidth=linew)
	plt.plot(data[0][0:int((len(data[0])*(3.5/5)))],error[0:int((len(data[0])*(3.5/5)))],'k-', linewidth=linew)

	plt.ylabel("Voltage [V]")
	plt.xlabel("Time [s]")
	plt.title('')
	plt.text(0, Voltage/8, "Fitting  RMSE: " + str(round(output_fitting_err,5)),family="monospace")
	plt.legend(["Physical Trace", "Fitting", "Error"], loc = 'center left')
	plt.show()