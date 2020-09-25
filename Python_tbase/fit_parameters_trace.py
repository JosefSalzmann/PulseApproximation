import auxiliary as aux
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from os import walk
from mpl_toolkits import mplot3d
import classes as cl
import importlib
import sys
import argparse
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='Calculate the parameters of an arbitrary long trace using the transfer functions obtained from the single pulse fittings')
parser.add_argument('-t', help="path to trace file that should be fitted using the transfer functions")
parser.add_argument('-f', help="path to fitting function")
parser.add_argument('-v', help="Vdd")
parser.add_argument('-u', help="path to transfer function with upside pulse as input and lowside pulse as output")
parser.add_argument('-l', help="path to transfer function with lowside pulse as input and upside pulse as output")

args = parser.parse_args()

filepath = args.t[1:len(args.t)] # remove blank from obtained string 
filepath = dir_path + "/" + filepath
sig_name = args.f[1:len(args.f)]
sig = importlib.import_module(sig_name, package=None)
Voltage = float(args.v[1:len(args.v)]) # 1.2
input_upside_pulse_meta_func = args.u[1:len(args.u)] # input pulse starts and ends at Vdd
input_lowside_pulse_meta_func = args.l[1:len(args.l)] # input pulse starts and ends at 0V


#########
Lin_approx_order = 2
#########


def compensation_terms(args): # Calculates the number compensation terms needed (multiples of Vdd) such that the resulting trace is always between 0 and Vdd.
	correction = 0
	if args[0] < 0 and (len(args)/sig.num_args)%2 == 0: # first edge is falling
		correction = 1
	return (len(args)/sig.num_args)/2-correction
	
	
def trace_sigmoids(t, args): # Function of the whole trace. The number of sigmoids depends on the length of args. After evaluation the resulting value is compensated by the number of terms needed.
	sum = 0
	for i in range(0,len(args)//(sig.num_args)):
		sum = sum + sig.sigmoid(t,args[sig.num_args*i:sig.num_args*(i+1)])
	return Voltage*(sum-compensation_terms(args))
	
def trace_sigmoids_wr(t, *args): # Wrapper function which the fitting algorithm can access.
	return trace_sigmoids(t, args)

def meta_func(X,args): # function in the form of f(x,X_0,...,X_n) = X_0 + x*X_1 + x*X_1^2 + ... + x*X_1^j + x*X_2 + ... + x*X_n^j
	ret_val = args[0]
	arg_count = 1
	for i in range(0, 2*sig.num_args-1):
		for j in range(1, Lin_approx_order+1):
			ret_val = ret_val + args[arg_count]*(X[i]**j)
			arg_count+=1
	return ret_val

input_upside_tr_fnc_params = aux.read_transfer_fnc_parameters(dir_path + "/" + input_upside_pulse_meta_func)
input_lowside_tr_fnc_params = aux.read_transfer_fnc_parameters(dir_path + "/" + input_lowside_pulse_meta_func)


# read trace file parameters
f = open(filepath, "r")
fr = f.read()
line = fr.split("\n")
endindex = len(line)
for i in range(0, len(line)):
	if line[i].split(",")[0] == "Input compensations terms":
		endindex = i
		break

trace_parameters = [[0.0 for i in range(sig.num_args)] for j in range(endindex-2)]
for i in range(0, endindex-2):
	params = line[i+2].split(",")
	for j in range(0, sig.num_args):
		trace_parameters[i][j] = float(params[j])

starts_with_rising_edge = True
if trace_parameters[0][0] > 0:
	num_lowside_pulses = len(trace_parameters)//2
	num_upside_pulses = (len(trace_parameters)-1)//2
else:
	num_lowside_pulses = (len(trace_parameters)-1)//2
	num_upside_pulses = len(trace_parameters)//2
	starts_with_rising_edge = False

current_edge_is_rising = not starts_with_rising_edge
first_output_transition_str = line[endindex+3].split(",")


# upside_output_params = [[0.0 for i in range(sig.num_args)] for j in range(num_lowside_pulses*2)]
# lowside_output_params = [[0.0 for i in range(sig.num_args)] for j in range(num_upside_pulses*2)]

final_output_params = [[0.0 for i in range(sig.num_args)]]

for i in range(0, sig.num_args):
	final_output_params[0][i] = float(first_output_transition_str[i])

for i in range(1, len(trace_parameters)):
	T = trace_parameters[i][1] - final_output_params[i-1][1]
	input_parameters = [0.0 for i in range(2*sig.num_args-1)]
	argc = 0
	for j in range(0, sig.num_args):
		if j != 1:
			input_parameters[argc] = trace_parameters[i][j]
		else:
			argc+=1
		input_parameters[sig.num_args+j-1] = final_output_params[i-1][j]
	input_parameters[sig.num_args] = T
	output_parameters = [0.0 for i in range(sig.num_args)]

	if not current_edge_is_rising:
		for j in range(0, sig.num_args):
			output_parameters[j] = meta_func(input_parameters, input_lowside_tr_fnc_params[j])
	else:
		for j in range(0, sig.num_args):
			output_parameters[j] = meta_func(input_parameters, input_upside_tr_fnc_params[j])

	output_parameters[1]+=final_output_params[i-1][1]
	final_output_params.append(output_parameters)
	current_edge_is_rising = not current_edge_is_rising

output_param_array = [0.0 for i in range(len(final_output_params)*2)]
for i in range(0, len(final_output_params)):
	output_param_array[i*2] = final_output_params[i][0]
	output_param_array[i*2+1] = final_output_params[i][1]
















path = dir_path + "/../WaveformData/t4_traces/inv_t4_invSim_520_200Traces.dat"
data = aux.read_file(path, sys.maxsize) # Read the given file in its full length.
data_reduced = aux.read_file(path, len(data[0])//10) # Read every tenth item of the given file.

approx_out = [0.0]*len(data_reduced[0])

output_fitting_error = [0.0]*len(data_reduced[0])
for i in range(0,len(data_reduced[0])): # Calculate the fitted curves.
	approx_out[i] = trace_sigmoids(data_reduced[0][i], output_param_array)
	output_fitting_error[i] = approx_out[i] - data_reduced[2][i]
# Draw a picture of the original traces and the fitted curves.	
plt.cla()
plt.clf()
fig = plt.gcf()
fig.set_size_inches(16, 4)
linew = 1.5	
plt.plot(data[0],data[1],'r-', linewidth=linew)
#plt.plot(data_reduced[0],approx_in,'r--', linewidth=linew)
#plt.plot(data_reduced[0],input_fitting_error,'r-.', linewidth=linew)
plt.plot(data[0],data[2],'g-', linewidth=linew)
plt.plot(data_reduced[0],approx_out,'g--', linewidth=linew)
plt.plot(data_reduced[0],output_fitting_error,'g-.', linewidth=linew)

input_fitting_err = 0.0 #aux.calc_rms_error_func(trace_sigmoids, input_fitting[0], data_reduced[0], data_reduced[1])/Voltage
output_fitting_err = aux.calc_rms_error_func(trace_sigmoids, output_param_array, data_reduced[0], data_reduced[2])/Voltage

plt.text(0, Voltage/8, "In  RMSE: " + str(round(input_fitting_err,5)) + "\nOut RMSE: " + str(round(output_fitting_err,5)),family="monospace")
file_name = path.split("/")[len(path.split("/"))-1]
title = file_name[:len(file_name)-4]
plt.title(title + " fitting using transfer functions")
plt.legend(["Input","Output","Output fitting from transfer functions", "fitting error"], loc = 'center left')
imgpath = path[:len(path)-4]
plt.ylabel("Voltage [V]")
plt.xlabel("Time [s]")
plt.savefig(imgpath + "_fitting.svg")
plt.show()