import auxiliary as aux	
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from os import walk
from mpl_toolkits import mplot3d
from timeit import default_timer as timer
import sys
import importlib



sig = importlib.import_module("exp", package=None)
Voltage = 1.2
params_per_sig = len(sig.trace_initial);

def compensation_terms(args):
	correction = 0;
	if args[sig.trace_steepness] < 0 and (len(args)/params_per_sig)%2 == 0: # first edge is falling
		correction = 1;
	return (len(args)/params_per_sig)/2-correction;
	
	
def trace_simple_exp(t, args):
	sum = 0;
	for i in range(0,len(args)/len(sig.trace_initial)):
		sum = sum + sig.sigmoid(t,args[len(sig.trace_initial)*i:len(sig.trace_initial)*(i+1)]);
	return Voltage*(sum-compensation_terms(args));
	
def trace_simple_exp_wr(t, *args):
	return trace_simple_exp(t, args);
	
def fit_trace(time, trace, lowres_time, lowres_trace):
	first_der = [0.0]*len(time);
	mult = 2000;
	for i in range(1,len(first_der)):
		first_der[i] = (trace[i]-trace[i-1])*mult;

	filter_size = 10;
	percent_in_range = 0.001;
	filter = [0.0]*filter_size;
	starts_at_Vdd = 0;
	next_edge_rising = True;
	if trace[0] > Voltage/2:
		starts_at_Vdd = 1;
		next_edge_rising = False;
	switching_points = [];
	min_height = 0.02;
	for i in range(0, len(trace)):
		filter[i%filter_size] = first_der[i];
		if (np.sum(filter) > min_height*filter_size and next_edge_rising) or (np.sum(filter) < -1*min_height*filter_size and not next_edge_rising):
			all_same_sign = True;			
			for j in range(1,filter_size):
				if filter[j]*filter[0] < 0: # not the same sign
					all_same_sign = False;
					break;
			if all_same_sign:
				all_in_range = True;
				abs_filter = np.absolute(filter);
				for j in range(1,filter_size):
					if abs_filter[j] < abs_filter[0]*(1-percent_in_range) or abs_filter[j] > abs_filter[0]*(1+percent_in_range):
						all_in_range = False;
						break;
				if all_in_range:
					next_edge_rising = not next_edge_rising;
					switching_points.append(i+filter_size);

	approx = [0.0]*len(time);
	trace_init = [0.0]*(len(sig.trace_initial)*len(switching_points));
	for i in range(0,len(switching_points)):
		for j in range(0,len(sig.trace_initial)):
			trace_init[i*len(sig.trace_initial)+j] = sig.trace_initial[j]
			if j == sig.trace_steepness:
				trace_init[i*len(sig.trace_initial)+j] = trace_init[i*len(sig.trace_initial)+j] * (-1)**(i+starts_at_Vdd);
		
		trace_init[len(sig.trace_initial)*i+sig.trace_shift] = time[switching_points[i]]*10**10;
		
	lower_bound = [0.0]*len(trace_init);
	upper_bound = [0.0]*len(trace_init);
	for i in range(0,len(switching_points)):
		for j in range(0,len(sig.trace_initial)):
			if i%2 != starts_at_Vdd:
				lower_bound[i*len(sig.trace_initial)+j] = sig.trace_falling_lower_bound[j];
				upper_bound[i*len(sig.trace_initial)+j] = sig.trace_falling_upper_bound[j];
			else:
				lower_bound[i*len(sig.trace_initial)+j] = sig.trace_rising_lower_bound[j];
				upper_bound[i*len(sig.trace_initial)+j] = sig.trace_rising_upper_bound[j];
	lower_bound[sig.trace_shift] = 0;
	upper_bound[len(trace_init)-len(sig.trace_initial)+sig.trace_shift] = time[len(time)-1]*10**10;
	for i in range(0,len(switching_points)-1):	
		lower_bound[len(sig.trace_initial)*(i+1)+sig.trace_shift] = (trace_init[len(sig.trace_initial)*i+sig.trace_shift]+trace_init[len(sig.trace_initial)*(i+1)+sig.trace_shift])/2;
		upper_bound[len(sig.trace_initial)*i+sig.trace_shift] = (trace_init[len(sig.trace_initial)*i+sig.trace_shift]+trace_init[len(sig.trace_initial)*(i+1)+sig.trace_shift])/2;

		
	fitting_params = optimize.curve_fit(trace_simple_exp_wr, lowres_time, lowres_trace, trace_init, bounds = (lower_bound, upper_bound), maxfev=5000)
	
	return fitting_params;
	
	
	

path = "../WaveformData/t4_traces/inv_t4_invSim_820_200Traces.dat"
data = aux.read_file(path, 100000)
data_reduced = aux.read_file(path, 10000)

input_fitting = fit_trace(data[0], data[1], data_reduced[0], data_reduced[1]);
output_fitting = fit_trace(data[0], data[2], data_reduced[0], data_reduced[2]);

approx_in = [0.0]*len(data_reduced[0]);
approx_out = [0.0]*len(data_reduced[0]);
for i in range(0,len(data_reduced[0])):
	approx_in[i] = trace_simple_exp(data_reduced[0][i], input_fitting[0]);
	approx_out[i] = trace_simple_exp(data_reduced[0][i], output_fitting[0]);
	
	
plt.cla()
plt.clf()
fig = plt.gcf()
fig.set_size_inches(12, 6)	
linew = 1.5	
plt.plot(data[0],data[1],'r-', linewidth=linew);
plt.plot(data[0],data[2],'g-', linewidth=linew);
plt.plot(data_reduced[0],approx_in,'r--', linewidth=linew);
plt.plot(data_reduced[0],approx_out,'g--', linewidth=linew);

input_fitting_err = aux.calc_rms_error_func(trace_simple_exp, input_fitting[0], data_reduced[0], data_reduced[1])/Voltage;
output_fitting_err = aux.calc_rms_error_func(trace_simple_exp, output_fitting[0], data_reduced[0], data_reduced[2])/Voltage;


plt.text(0, Voltage/4, "In  RMSE: " + str(round(input_fitting_err,5)) + "\nOut RMSE: " + str(round(output_fitting_err,5)),family="monospace")
file_name = path.split("/")[len(path.split("/"))-1]
title = file_name[:len(file_name)-4]
plt.title(title)
plt.legend(["Input","Output", "Input-fitting", "Output-fitting"], loc = 'center left')
imgpath = path[:len(path)-4]
plt.ylabel("Voltage [V]");
plt.xlabel("Time [s]");
plt.savefig(imgpath + "_fitting.svg")	
#plt.show()




folderpath = path[0:len(path)-4];
fw = open(folderpath + "_fitting.csv", "w+");


parameter_string = [];
for i in range(0,len(sig.trace_initial)-1):
	parameter_string.append(sig.trace_paramters_names[i] + ",");
parameter_string.append(sig.trace_paramters_names[len(sig.trace_initial)-1] + "\n");

write_str = [];
write_str.append("Input parameters\n");
write_str.append(''.join(parameter_string));
for i in range(0, len(input_fitting[0])/len(sig.trace_initial)):
	for j in range(0,len(sig.trace_initial)-1):
		write_str.append(str(round(input_fitting[0][i*len(sig.trace_initial)+j], 5)) + ",");	
	write_str.append(str(round(input_fitting[0][i*len(sig.trace_initial) + len(sig.trace_initial)-1],5)));	
	write_str.append("\n");
write_str.append("Input compensations terms," + str(compensation_terms(input_fitting[0])) + "\n");	
	
write_str.append("Output parameters\n");
write_str.append(''.join(parameter_string));
for i in range(0, len(output_fitting[0])/len(sig.trace_initial)):	
	for j in range(0,len(sig.trace_initial)-1):
		write_str.append(str(round(output_fitting[0][i*len(sig.trace_initial)+j], 5)) + ",");	
	write_str.append(str(round(output_fitting[0][i*len(sig.trace_initial) + len(sig.trace_initial)-1],5)));	
	write_str.append("\n");
write_str.append("Output compensations terms," + str(compensation_terms(output_fitting[0])) + "\n");		
write_str.append("Input RMSE," + str(round(input_fitting_err,7)) + "\n");		
write_str.append("Output RMSE," + str(round(output_fitting_err,7)) + "\n");		
	
	
#print(''.join(write_str));	
fw.write(''.join(write_str))
fw.close();











