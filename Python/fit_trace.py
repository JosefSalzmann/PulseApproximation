import auxiliary as aux	
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import sys
import importlib

# First parameter: path to the file that should be fitted.
# Second parameter: name of the model function.
# Third parameter: Vdd.

sig_name = sys.argv[2]
sig = importlib.import_module(sig_name, package=None)
Voltage = float(sys.argv[3]) # 1.2
params_per_sig = len(sig.trace_initial)

def compensation_terms(args): # Calculates the number compensation terms needed (multiples of Vdd) such that the resulting trace is always between 0 and Vdd.
	correction = 0
	if args[sig.trace_steepness] < 0 and (len(args)/params_per_sig)%2 == 0: # first edge is falling
		correction = 1
	return (len(args)/params_per_sig)/2-correction
	
	
def trace_sigmoids(t, args): # Function of the whole trace. The number of sigmoids depends on the length of args. After evaluation the resulting value is compensated by the number of terms needed.
	sum = 0
	for i in range(0,len(args)//len(sig.trace_initial)):
		sum = sum + sig.sigmoid(t,args[len(sig.trace_initial)*i:len(sig.trace_initial)*(i+1)])
	return Voltage*(sum-compensation_terms(args))
	
def trace_sigmoids_wr(t, *args): # Wrapper function which the fitting algorithm can access.
	return trace_sigmoids(t, args)
	
def fit_trace(time, trace, lowres_time, lowres_trace): # Given a trace, this function calculates the number of sigmoids and their parameters to fit the trace.
	first_der = [0.0]*len(time) 
	for i in range(1,len(first_der)): # Calculate the first derivative of the given trace in order to find the turning points of the trace so that the number of sigmoids and their initial position can be determined.
		if (time[i]-time[i-1]) != 0:
			first_der[i] = (trace[i]-trace[i-1])/(time[i]-time[i-1])*10**-9
		else: # dat-files sometimes contain the same time stamp twice...
			first_der[i] = first_der[i-1] # in that case the derivative is just interpolated using the previous value.

	filter_size = 10
	percent_in_range = 0.001
	filter = [0.0]*filter_size
	starts_at_Vdd = 0
	next_edge_rising = True
	if trace[0] > Voltage/2: # Identify if the trace starts with a falling or a rising edge.
		starts_at_Vdd = 1
		next_edge_rising = False
	switching_points = []
	min_height = 0.02
	for i in range(0, len(trace)-filter_size): # Iterate through the derivative array and identify small portions (size of the filter) where the gradient is constant and not zero.
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

	trace_init = [0.0]*(len(sig.trace_initial)*len(switching_points)) # Array with the initial guess of the parameters of the single simgoids.
	for i in range(0,len(switching_points)):
		for j in range(0,len(sig.trace_initial)):
			trace_init[i*len(sig.trace_initial)+j] = sig.trace_initial[j]
			if j == sig.trace_steepness:
				trace_init[i*len(sig.trace_initial)+j] = trace_init[i*len(sig.trace_initial)+j] * (-1)**(i+starts_at_Vdd) # Set the steepness of the single sigmoids so that they are rising/falling.
		
		trace_init[len(sig.trace_initial)*i+sig.trace_shift] = time[switching_points[i]]*10**10 # The inital guess for the shift will be the identified turning points.
		
	lower_bound = [0.0]*len(trace_init)
	upper_bound = [0.0]*len(trace_init)
	for i in range(0,len(switching_points)): # Set the bound arrays to the values for a falling/rising sigmoid.
		for j in range(0,len(sig.trace_initial)):
			if i%2 != starts_at_Vdd:
				lower_bound[i*len(sig.trace_initial)+j] = sig.trace_falling_lower_bound[j]
				upper_bound[i*len(sig.trace_initial)+j] = sig.trace_falling_upper_bound[j]
			else:
				lower_bound[i*len(sig.trace_initial)+j] = sig.trace_rising_lower_bound[j]
				upper_bound[i*len(sig.trace_initial)+j] = sig.trace_rising_upper_bound[j]
				
	for i in range(0,len(switching_points)-1): # The bounds of the shift parameters will depend on the identified turning points and are set in between the middle of two turning points.
		lower_bound[len(sig.trace_initial)*(i+1)+sig.trace_shift] = (trace_init[len(sig.trace_initial)*i+sig.trace_shift]+trace_init[len(sig.trace_initial)*(i+1)+sig.trace_shift])/2
		upper_bound[len(sig.trace_initial)*i+sig.trace_shift] = (trace_init[len(sig.trace_initial)*i+sig.trace_shift]+trace_init[len(sig.trace_initial)*(i+1)+sig.trace_shift])/2
	lower_bound[sig.trace_shift] = 0
	upper_bound[len(trace_init)-len(sig.trace_initial)+sig.trace_shift] = time[len(time)-1]*10**10
		
	fitting_params = optimize.curve_fit(trace_sigmoids_wr, lowres_time, lowres_trace, trace_init, bounds = (lower_bound, upper_bound), maxfev=5000) # The fitting process will be performed with the lowres trace since we don't need all the details of the curve for an acceptable fitting.
	
	return fitting_params
	
	
	

path = sys.argv[1]
data = aux.read_file(path, sys.maxsize) # Read the given file in its full length.
data_reduced = aux.read_file(path, len(data[0])//10) # Read every tenth item of the given file.

input_fitting = fit_trace(data[0], data[1], data_reduced[0], data_reduced[1])
output_fitting = fit_trace(data[0], data[2], data_reduced[0], data_reduced[2])

approx_in = [0.0]*len(data_reduced[0])
approx_out = [0.0]*len(data_reduced[0])
input_fitting_error = [0.0]*len(data_reduced[0])
output_fitting_error = [0.0]*len(data_reduced[0])
for i in range(0,len(data_reduced[0])): # Calculate the fitted curves.
	approx_in[i] = trace_sigmoids(data_reduced[0][i], input_fitting[0])
	approx_out[i] = trace_sigmoids(data_reduced[0][i], output_fitting[0])
	input_fitting_error[i] = approx_in[i] - data_reduced[1][i]
	output_fitting_error[i] = approx_out[i] - data_reduced[2][i]
# Draw a picture of the original traces and the fitted curves.	
plt.cla()
plt.clf()
fig = plt.gcf()
fig.set_size_inches(16, 4)
linew = 1.5	
plt.plot(data[0],data[1],'r-', linewidth=linew)
plt.plot(data_reduced[0],approx_in,'r--', linewidth=linew)
plt.plot(data_reduced[0],input_fitting_error,'r-.', linewidth=linew)
plt.plot(data[0],data[2],'g-', linewidth=linew)
plt.plot(data_reduced[0],approx_out,'g--', linewidth=linew)
plt.plot(data_reduced[0],output_fitting_error,'g-.', linewidth=linew)

input_fitting_err = aux.calc_rms_error_func(trace_sigmoids, input_fitting[0], data_reduced[0], data_reduced[1])/Voltage
output_fitting_err = aux.calc_rms_error_func(trace_sigmoids, output_fitting[0], data_reduced[0], data_reduced[2])/Voltage

plt.text(0, Voltage/8, "In  RMSE: " + str(round(input_fitting_err,5)) + "\nOut RMSE: " + str(round(output_fitting_err,5)),family="monospace")
file_name = path.split("/")[len(path.split("/"))-1]
title = file_name[:len(file_name)-4]
plt.title(title)
plt.legend(["Input","Input fitting","Input fitting error", "Output","Output fitting","Output fitting error"], loc = 'center left')
imgpath = path[:len(path)-4]
plt.ylabel("Voltage [V]")
plt.xlabel("Time [s]")
plt.savefig(imgpath + "_fitting.svg")
#plt.show()




# Write the calculated parameters into a csv-file.
folderpath = path[0:len(path)-4]
fw = open(folderpath + "_fitting.csv", "w+")

parameter_string = []
for i in range(0,len(sig.trace_initial)-1):
	parameter_string.append(sig.trace_paramters_names[i] + ",")
parameter_string.append(sig.trace_paramters_names[len(sig.trace_initial)-1] + "\n")

write_str = []
write_str.append("Input parameters\n")
write_str.append(''.join(parameter_string))
for i in range(0, len(input_fitting[0])//len(sig.trace_initial)):
	for j in range(0,len(sig.trace_initial)-1):
		write_str.append(str(round(input_fitting[0][i*len(sig.trace_initial)+j], 5)) + ",")	
	write_str.append(str(round(input_fitting[0][i*len(sig.trace_initial) + len(sig.trace_initial)-1],5)))	
	write_str.append("\n")
write_str.append("Input compensation terms," + str(compensation_terms(input_fitting[0])) + "\n")	
	
write_str.append("Output parameters\n")
write_str.append(''.join(parameter_string))
for i in range(0, len(output_fitting[0])//len(sig.trace_initial)):	
	for j in range(0,len(sig.trace_initial)-1):
		write_str.append(str(round(output_fitting[0][i*len(sig.trace_initial)+j], 5)) + ",")	
	write_str.append(str(round(output_fitting[0][i*len(sig.trace_initial) + len(sig.trace_initial)-1],5)))	
	write_str.append("\n")
write_str.append("Output compensations term," + str(compensation_terms(output_fitting[0])) + "\n")		
write_str.append("Input RMSE," + str(round(input_fitting_err,7)) + "\n")		
write_str.append("Output RMSE," + str(round(output_fitting_err,7)) + "\n")		
	
fw.write(''.join(write_str))
fw.close()

print(''.join(write_str))