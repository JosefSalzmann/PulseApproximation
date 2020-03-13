import auxiliary as aux
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from os import walk
from mpl_toolkits import mplot3d
import classes as cl
import importlib
import sys

# First parameter: path to the folder that should be fitted.
# Second parameter: name of the model function.
# Third parameter: Vdd.
# Fourth parameter: Approximation order.


folderpath = sys.argv[1];
sig_name = sys.argv[2];
sig = importlib.import_module(sig_name, package=None);
Voltage = float(sys.argv[3]); # 1.2;
Lin_approx_order = int(sys.argv[4]); # 2;



def pulse(t,args):
	args_left = args[0:len(sig.input_initial)/2];
	args_right = args[(len(sig.input_initial)/2):len(sig.input_initial)];
	Correction = 0;
	if(args_left[sig.steepness_index] > 0 and args_right[sig.steepness_index] < 0):
		Correction = 1;
	return Voltage*(sig.sigmoid(t, args_left) + sig.sigmoid(t, args_right) - Correction);
	
f = []
for (dirpath, dirnames, filenames) in walk(folderpath):
    f = filenames
    break

dat_file_content = []
dat_file_fittings = []

for file in filenames:
	if file.split(".")[1] == "dat":
		dat_file_content.append(aux.read_file_with_name(folderpath + "/" + file, sys.maxint))		


fo = open(folderpath + "/fitting_parameters.txt", "r")
fr = fo.read()
line = fr.split("\n")
line_cnt = 0
file_names = []
for i in range(0,len(line)):
	line_sp = line[i].split(",");
	if len(line_sp)>2:		
		file_names.append(line_sp[0])
		line[i] = ",".join(line_sp[1:len(line_sp)-2]);
		line_cnt = line_cnt + 1
	else:
		break
line = line[0:line_cnt];

line_len = len(line[0].split(','))
params = [[0.0 for i in range(line_cnt)] for j in range(line_len-1)]
input_shifts = [0.0] * line_cnt;
for i in range(0,line_cnt):
	line_split = line[i].split(',');
	dat_file_fittings.append(cl.Parameter_Set(file_names[i], np.array(map(float, line_split))))
	cnt = 0
	for j in range(0,line_len):
		if j != sig.input_shift:
			params[cnt][i] = dat_file_fittings[i].parameters[j]
			cnt+=1;		
		else:
			input_shifts[i] = dat_file_fittings[i].parameters[j]


def meta_func(X,args):
	ret_val = args[0]
	arg_count = 1
	for i in range(0, len(sig.input_initial)-1):
		for j in range(1, Lin_approx_order+1):
			ret_val = ret_val + args[arg_count]*(X[i]**j)
			arg_count+=1
	return ret_val

def meta_func_wr(X, *args):
	return meta_func(X,args)
	
meta_params = [[10 for i in range(1+(len(sig.input_initial)-1)*Lin_approx_order)] for j in range(len(sig.input_initial))]

for i in range(0, len(sig.input_initial)):
	meta_params[i] = optimize.curve_fit(meta_func_wr, params[:len(sig.input_initial)-1], params[len(sig.input_initial)-1+i], meta_params[i], maxfev = 10000)[0]

	
names = ['']*(len(params))
arg_c = 0
for i in range(0, len(sig.input_initial)):
	if i != sig.input_shift:
		names[arg_c] = "input_" + str(sig.parameter_names[i]);
		arg_c += 1
	names[i+len(sig.input_initial)-1] = "output_" + str(sig.parameter_names[i]);
names[sig.input_length-1] = "input_pulse_length" ;

fw = open(folderpath + "/meta_fittings.txt", "w+")
for i in range(0, len(sig.input_initial)):
	print(names[len(sig.input_initial)-1+i] + "=");
	fw.write(names[len(sig.input_initial)-1+i] + "\n");
	print(str(meta_params[i][0])+ " +");
	fw.write("offset," + str(meta_params[i][0]) + "\n");
	arg_count = 1;
	for j in range(0, len(sig.input_initial)-1):
		for k in range(1, Lin_approx_order+1):
			if(j == len(sig.input_initial)-2 and k == Lin_approx_order):
				print(str(meta_params[i][arg_count]) + "*" + names[j] + "^" + str(k));				
			else:
				print(str(meta_params[i][arg_count]) + "*" + names[j] + "^" + str(k) + "+");
			fw.write(names[j] + "," + str(k) +  "," + str(meta_params[i][arg_count]) + "\n");
			arg_count+=1			
	print("");
	fw.write("\n");





	
count = len(params[0])
meta_fittings = [[0 for i in range(count)] for j in range(len(sig.input_initial))]
input_params_fitting = [[0 for i in range(len(sig.input_initial)-1)] for j in range(count)]

for i in range(0,count):
	for j in range(0,len(sig.input_initial)-1):
		input_params_fitting[i][j] = params[j][i]	

for i in range(0, len(sig.input_initial)):
	for j in range(0,count):
		meta_fittings[i][j] = meta_func(input_params_fitting[j], meta_params[i])		

		
fitting_sq_errors = [[0 for i in range(count)] for i in range(len(sig.input_initial))]
for i in range(0, len(sig.input_initial)):
	fitting_sq_errors[i] = np.subtract(params[len(sig.input_initial)-1+i],meta_fittings[i])**2

fw.write("\n\n");
rms_errors = [0 for i in range(len(sig.input_initial))]
for i in range(0, len(sig.input_initial)):
	rms_errors[i] = np.sqrt(sum(fitting_sq_errors[i])/len(fitting_sq_errors[i]))/Voltage
	print(names[len(sig.input_initial)-1+i] + " RMSE: " + str(rms_errors[i]))
	fw.write(names[len(sig.input_initial)-1+i] + " RMSE," + str(rms_errors[i]) + "\n");
fw.close()
	
meta_output_params = [[0.0 for i in range(len(sig.input_initial))] for j in range(count)]
for i in range(0,count):
	for j in range(0,len(sig.input_initial)):
		meta_output_params[i][j] = meta_fittings[j][i]
for i in range(0,count):
	meta_output_params[i][sig.output_shift] = input_shifts[i] + meta_output_params[i][sig.output_shift];
	meta_output_params[i][sig.output_length] = meta_output_params[i][sig.output_shift] + meta_output_params[i][sig.output_length];

for i in range(0, len(dat_file_content)):
	meta_output_fitting = [0]*len(dat_file_content[i].time)
	for j in range(0, len(dat_file_content[i].time)):
		meta_output_fitting[j] = pulse(dat_file_content[i].time[j], meta_output_params[i])
	plt.cla()
	plt.clf()
	fig = plt.gcf()
	fig.set_size_inches(8, 6)
	
	linew = 1;
	plt.plot(dat_file_content[i].time, dat_file_content[i].input_pulse,'r-', linewidth=linew)
	plt.plot(dat_file_content[i].time, dat_file_content[i].output_pulse,'g-', linewidth=linew)		
	plt.plot(dat_file_content[i].time,meta_output_fitting,'b--', linewidth=linew)	
	plt.text(0, Voltage/4, "Out RMSE: " + str(round(aux.calc_rms_error_data(dat_file_content[i].output_pulse,meta_output_fitting),5)),family="monospace")
	plt.title(dat_file_content[i].file_name + " Meta fitting")
	plt.legend(["Input","Output", "Meta Output fitting"], loc = 'center left')
	#plt.show()
	path = folderpath + "/" + dat_file_content[i].file_name
	imgpath = path[:len(path)-4]
	plt.savefig(imgpath + "_Meta_fitting.png")

