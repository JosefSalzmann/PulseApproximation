import auxiliary as aux
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from os import walk
from mpl_toolkits import mplot3d
from timeit import default_timer as timer
import sys
from scipy.special import gamma, factorial


import importlib
sig = importlib.import_module("atan_taylor", package=None)

tan_taylor_coeffs_odd = [	1.000000000000000, 0.333333333333333, 0.133333333333333, 
						0.053968253968254, 0.021869488536155, 0.008863235529902,
						0.003592128036572, 0.001455834387051, 0.000590027440946 ]
                       
def taylor_approx_simple(x,a):
	sum = x
	for i in range(1,min(a+1,len(tan_taylor_coeffs_odd))):
		sum = sum + (tan_taylor_coeffs_odd[i]*x**(2*i+1))
	return sum	

def sigmoid_simple(t, args):
	return np.arctan(taylor_approx_simple(args[0]*(t*10**10-args[1]), args[2]))
	
	
def taylor_approx(x,a):
	sum = x
	if a < len(tan_taylor_coeffs_odd):
		for i in range(1,len(tan_taylor_coeffs_odd)):
			sum = sum + (tan_taylor_coeffs_odd[i]*x**(2*i+1))*((np.arctan(taylor_approx(a-i,50))/np.pi)+0.5)
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
	return np.arctan(taylor_approx_straight(args[0]*(t*10**10-args[1]), args[2]))
	
	
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

max_x = 10
length = 1000

x = [0.0]*length

for i in range(0,length):
	x[i] = -max_x + i*(2.0*max_x/length)
	


	
	
if int(sys.argv[1]) == 0:
	straight = [0.0]*length
	atan_zero = [0.0]*length
	atan_one = [0.0]*length
	atan_two = [0.0]*length
	atan_ten = [0.0]*length


	for i in range(0,length):
		if(x[i] < -np.pi/2):
			straight[i] = -np.pi/2
		elif(x[i] > np.pi/2):
			straight[i] = np.pi/2
		else:
			straight[i] = x[i]
		atan_zero[i] = sigmoid_simple(x[i]/10**10, [1,0,0])	
		atan_one[i] = sigmoid_simple(x[i]/10**10, [1,0,1])
		atan_two[i] = sigmoid_simple(x[i]/10**10, [1,0,2])
		atan_ten[i] = sigmoid_simple(x[i]/10**10, [1,0,5])

	plt.cla()
	plt.clf()
	fig = plt.gcf()
	fig.set_size_inches(8, 6)

		
	linew = 1
		
	
	plt.plot(x,atan_zero,'r-', linewidth=linew)
	plt.plot(x,atan_one,'g-', linewidth=linew)
	plt.plot(x,atan_two,'b-', linewidth=linew)
	plt.plot(x,atan_ten,'y-', linewidth=linew)
	plt.plot(x,straight,'c-', linewidth=linew)

	plt.title('')
	plt.legend(["c = 0 (equal to atan(x))","c = 1", "c = 2", "c = 5", "s(x)"], loc = 'center left')
	#plt.text(-10, 1, "text")
	plt.show()
elif int(sys.argv[1]) == 1:

	#for i in range(0,length):
	#	x[i] = i*(1.0*max_x/length)


	atan_zero = [0.0]*length
	atan_1 = [0.0]*length
	atan_2 = [0.0]*length
	atan_3 = [0.0]*length
	atan_4 = [0.0]*length
	atan_5 = [0.0]*length

	for i in range(0,length):
		atan_zero[i] = sigmoid_frac_der(x[i]/10**10, [1,0,1,1])	
		atan_1[i] = sigmoid_frac_der(x[i]/10**10, [1,0,1.1,2])
		atan_2[i] = sigmoid_frac_der(x[i]/10**10,  [1,0,1.2,3])
		#atan_3[i] = sigmoid(x[i]/10**10, [1,0,1])
		#atan_4[i] = sigmoid(x[i]/10**10, [1,0,0.8])
		#atan_5[i] = sigmoid(x[i]/10**10, [1,0,1])

	plt.cla()
	plt.clf()
	fig = plt.gcf()
	fig.set_size_inches(8, 6)

		
	linew = 1
		
	plt.plot(x,atan_zero,'r-', linewidth=linew)
	plt.plot(x,atan_1,'g-', linewidth=linew)
	plt.plot(x,atan_2,'b-', linewidth=linew)
	#plt.plot(x,atan_3,'y-', linewidth=linew)
	#plt.plot(x,atan_4,'c-', linewidth=linew)
	#plt.plot(x,atan_5,'m-', linewidth=linew)

	plt.title('')
	plt.legend(["first","second", "third", "c = 1"], loc = 'center right')
	#plt.text(-10, 1, "text")
	plt.show() 

elif int(sys.argv[1]) == 2:
	atan_zero_zero = [0.0]*length
	atan_pfive_five = [0.0]*length


	for i in range(0,length):
		atan_zero_zero[i] = np.pi*sig.sigmoid(x[i]/10**10, [1,0,0, 0])-np.pi/2
		atan_pfive_five[i] = np.pi*sig.sigmoid(x[i]/10**10, [1,0,0.5, 5])-np.pi/2

	plt.cla()
	plt.clf()
	fig = plt.gcf()
	fig.set_size_inches(8, 6)

		
	linew = 1
		
	plt.plot(x,atan_zero_zero,'r-', linewidth=linew)
	plt.plot(x,atan_pfive_five,'g-', linewidth=linew)
	#plt.plot(x,atan_two,'b-', linewidth=linew)
	#plt.plot(x,atan_ten,'y-', linewidth=linew)

	plt.title('')
	plt.legend(["c = 0, d = 0","c = 5, d = 0.5"], loc = 'center left')
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

	path = "../WaveformData/t4_traces/inv_t4_invSim_Traces.dat"
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
	plt.legend(["Input","Initial Guess"], loc = 'center left')
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