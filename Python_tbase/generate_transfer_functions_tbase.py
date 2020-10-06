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


parser = argparse.ArgumentParser(description='Generate transfer functions (on pulse base) using the fitting the values of fitting.py')
parser.add_argument('-t', help="path to folder of which the transfer functions should be generated")
parser.add_argument('-f', help="path to fitting function")
parser.add_argument('-v', help="Vdd")
parser.add_argument('-a', help="approximation order (3 is usually enough)")
parser.add_argument('-p', '--p', help="generate and save picture of plot", action='store_true')

args = parser.parse_args()


dir_path = os.path.dirname(os.path.realpath(__file__))

folderpath = args.t # remove blank from obtained string
folderpath = dir_path + "/" + folderpath
sig_name = args.f
sig = importlib.import_module(sig_name, package=None)
Voltage = float(args.v) # 1.2
Lin_approx_order = int(args.a) # 3



def pulse(t,args):
	args_left = args[0:sig.num_args]
	args_right = args[sig.num_args:2*sig.num_args]
	Correction = 0
	if(args_left[0] > 0 and args_right[0] < 0):
		Correction = 1
	return Voltage*(sig.sigmoid(t, args_left) + sig.sigmoid(t, args_right) - Correction)
	
f = []
for (dirpath, dirnames, filenames) in walk(folderpath):
    f = filenames
    break

dat_file_content = []
dat_file_fittings = []

for file in filenames:
	if file.split(".")[1] == "dat":
		dat_file_content.append(aux.read_file_with_name(folderpath + "/" + file, sys.maxsize))		


fo = open(folderpath + "/fitting_parameters.txt", "r")
fr = fo.read()
line = fr.split("\n")
line_cnt = 0
file_names = []
for i in range(0,len(line)):
	line_sp = line[i].split(",")
	if len(line_sp)>2:		
		file_names.append(line_sp[0])
		line[i] = ",".join(line_sp[1:len(line_sp)-2])
		line_cnt = line_cnt + 1
	else:
		break
line = line[0:line_cnt]

line_len = len(line[0].split(','))
params = [[0.0 for i in range(line_cnt)] for j in range(line_len)]
for i in range(0,line_cnt):
	line_split = line[i].split(',')
	line_split_float = np.array(line_split).astype(np.float)
	dat_file_fittings.append(cl.Parameter_Set(file_names[i], line_split_float))
	for j in range(0,line_len):
		params[j][i] = dat_file_fittings[i].parameters[j]	


function_input_params = [[0.0 for i in range(line_cnt)] for j in range(2*sig.num_args-1)]
function_output_params = [[0.0 for i in range(line_cnt)] for j in range(sig.num_args)]

for i in range(0,line_cnt):
	argc = 0
	for j in range(0, sig.num_args):
		if j != 1:
			function_input_params[argc][i] = params[j+sig.num_args][i]
			function_input_params[sig.num_args+j-1][i] = params[j+2*sig.num_args][i]
			function_output_params[j][i] = params[j+3*sig.num_args][i]
		else:
			argc = argc + 1
			function_input_params[sig.num_args+j-1][i] = params[j+sig.num_args][i]-params[j+2*sig.num_args][i]
			function_output_params[j][i] = params[j+3*sig.num_args][i]-params[j+1*sig.num_args][i]

def meta_func(X,args): # function in the form of f(x,X_0,...,X_n) = X_0 + x*X_1 + x*X_1^2 + ... + x*X_1^j + x*X_2 + ... + x*X_n^j
	ret_val = args[0]
	arg_count = 1
	for i in range(0, 2*sig.num_args-1):
		for j in range(1, Lin_approx_order+1):
			ret_val = ret_val + args[arg_count]*(X[i]**j)
			arg_count+=1
	return ret_val

def meta_func_wr(X, *args):
	return meta_func(X,args)
	
meta_params = [[10 for i in range(1+(2*sig.num_args-1)*Lin_approx_order)] for j in range(sig.num_args)]

for i in range(0, sig.num_args):
	meta_params[i] = optimize.curve_fit(meta_func_wr, function_input_params[0:2*sig.num_args-1], function_output_params[i], meta_params[i], maxfev = 10000)[0]

	
input_names = ['']*(2*sig.num_args-1)
output_names = ['']*(sig.num_args)

argc = 0
for i in range(0, sig.num_args):
	if i != 1:
		input_names[argc] = "input_" + str(sig.parameter_names[i])
	else:
		argc = argc + 1
	input_names[sig.num_args-1+i] = "previous_output_" + str(sig.parameter_names[i])
	output_names[i] = "output_" + str(sig.parameter_names[i])
input_names[sig.num_args] = "T"


fw = open(folderpath + "/transferFunctions.txt", "w+")
for i in range(0, sig.num_args):
	print(output_names[i] + "=")
	fw.write(output_names[i] + "\n")
	print(str(meta_params[i][0])+ " +")
	fw.write("offset," + str(meta_params[i][0]) + "\n")
	arg_count = 1
	for j in range(0, 2*sig.num_args-1):
		for k in range(1, Lin_approx_order+1):
			print(str(meta_params[i][arg_count]) + "*" + input_names[j] + "^" + str(k) + "+")
			fw.write(input_names[j] + "," + str(k) +  "," + str(meta_params[i][arg_count]) + "\n")
			arg_count+=1			
	print("")
	fw.write("\n")


	
count = len(params[0])
meta_fittings = [[0 for i in range(count)] for j in range(sig.num_args)]
input_params_fitting = [[0 for i in range(2*sig.num_args-1)] for j in range(count)]

for i in range(0,count):
	for j in range(0,2*sig.num_args-1):
		input_params_fitting[i][j] = function_input_params[j][i]	

for i in range(0, sig.num_args):
	for j in range(0,count):
		meta_fittings[i][j] = meta_func(input_params_fitting[j], meta_params[i])		

for i in range(0,count):
	meta_fittings[1][i]+=params[1+1*sig.num_args][i]
		
fitting_sq_errors = [[0 for i in range(count)] for i in range(2*sig.num_args)]
for i in range(0, sig.num_args):
	fitting_sq_errors[i] = np.subtract(params[3*sig.num_args+i],meta_fittings[i])**2

fw.write("\n\n")
rms_errors = [0 for i in range(sig.num_args)]
for i in range(0, sig.num_args):
	rms_errors[i] = np.sqrt(sum(fitting_sq_errors[i])/len(fitting_sq_errors[i]))/Voltage
	print(output_names[i] + " RMSE: " + str(rms_errors[i]))
	fw.write(output_names[i] + " RMSE," + str(rms_errors[i]) + "\n")
fw.close()
	
meta_output_params = [[0.0 for i in range(2*sig.num_args)] for j in range(count)]
for i in range(0,count):
	for j in range(0,sig.num_args):
		meta_output_params[i][j] = params[j+2*sig.num_args][i]
for i in range(0,count):
	for j in range(0,sig.num_args):
		meta_output_params[i][j+sig.num_args] = meta_fittings[j][i] 
	
if args.p:
	for i in range(0, len(dat_file_content)):
		meta_output_fitting = [0]*len(dat_file_content[i].time)
		for j in range(0, len(dat_file_content[i].time)):
			meta_output_fitting[j] = pulse(dat_file_content[i].time[j], meta_output_params[i])
		plt.cla()
		plt.clf()
		fig = plt.gcf()
		fig.set_size_inches(8, 6)
		
		linew = 1
		plt.plot(dat_file_content[i].time, dat_file_content[i].input_pulse,'r-', linewidth=linew)
		plt.plot(dat_file_content[i].time, dat_file_content[i].output_pulse,'g-', linewidth=linew)		
		plt.plot(dat_file_content[i].time,meta_output_fitting,'b--', linewidth=linew)	
		plt.text(0, Voltage/4, "Out RMSE: " + str(round(aux.calc_rms_error_data(dat_file_content[i].output_pulse,meta_output_fitting),5)),family="monospace")
		plt.title(dat_file_content[i].file_name + " transfer function fitting")
		plt.legend(["Input","Output", "Meta Output fitting"], loc = 'center left')
		#plt.show()
		path = folderpath + "/" + dat_file_content[i].file_name
		imgpath = path[:len(path)-4]
		plt.savefig(imgpath + "_transferFunctions_fitting.png")

