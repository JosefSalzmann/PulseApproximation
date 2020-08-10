import auxiliary as aux
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from os import walk
from mpl_toolkits import mplot3d
from timeit import default_timer as timer
import importlib
import sys

# First parameter: path to the folder that should be fitted.
# Second parameter: name of the model function.
# Third parameter: Vdd.
# Fourth parameter: Number of points used to fit the waveforms(every 10th item is usually enough to achieve the same fitting quality as with all data points)


folderpath = sys.argv[1]
sig_name = sys.argv[2]
sig = importlib.import_module(sig_name, package=None)
Voltage = float(sys.argv[3]) # 1.2 
num_points = int(sys.argv[4]) # 500


def pulse(t,args):
	args_left = args[0:sig.num_args]
	args_right = args[sig.num_args:2*sig.num_args]
	Correction = 0
	if(args_left[0] > 0 and args_right[0] < 0):
		Correction = 1
	return Voltage*(sig.sigmoid(t, args_left) + sig.sigmoid(t, args_right) - Correction)
	
def pulse_wr(t,*args): # Wrapper function which the fitting algorithm can access.
	return pulse(t,args)

	
	
def get_params(visualize, path): # Fit a single file and return the fitting parameters
	data = aux.read_file(path, num_points)
	
	input_is_lowside = True
	
	if data[1][len(data[1])-1] > Voltage / 2 and data[1][0] > Voltage / 2: # Determine which pulse is highside or lowside.
		input_is_lowside = False

	input_init = [0.0]*(2*sig.num_args)
	output_init = [0.0]*(2*sig.num_args)
	input_lower_bound = [0.0]*(2*sig.num_args)
	input_upper_bound = [0.0]*(2*sig.num_args)
	output_lower_bound = [0.0]*(2*sig.num_args)
	output_upper_bound = [0.0]*(2*sig.num_args)

	if input_is_lowside:
		input_init[:sig.num_args] = sig.rising_initial
		input_init[sig.num_args:2*sig.num_args] = sig.falling_initial
		input_lower_bound[:sig.num_args] = sig.rising_lower_bound
		input_lower_bound[sig.num_args:2*sig.num_args] = sig.falling_lower_bound
		input_upper_bound[:sig.num_args] = sig.rising_upper_bound
		input_upper_bound[sig.num_args:2*sig.num_args] = sig.falling_upper_bound
		output_init[:sig.num_args] = sig.falling_initial
		output_init[sig.num_args:2*sig.num_args] = sig.rising_initial
		output_lower_bound[:sig.num_args] = sig.falling_lower_bound
		output_lower_bound[sig.num_args:2*sig.num_args] = sig.rising_lower_bound
		output_upper_bound[:sig.num_args] = sig.falling_upper_bound
		output_upper_bound[sig.num_args:2*sig.num_args] = sig.rising_upper_bound
	else:	
		input_init[:sig.num_args] = sig.falling_initial
		input_init[sig.num_args:2*sig.num_args] = sig.rising_initial
		input_lower_bound[:sig.num_args] = sig.falling_lower_bound
		input_lower_bound[sig.num_args:2*sig.num_args] = sig.rising_lower_bound
		input_upper_bound[:sig.num_args] = sig.falling_upper_bound
		input_upper_bound[sig.num_args:2*sig.num_args] = sig.rising_upper_bound
		output_init[:sig.num_args] = sig.rising_initial
		output_init[sig.num_args:2*sig.num_args] = sig.falling_initial
		output_lower_bound[:sig.num_args] = sig.rising_lower_bound
		output_lower_bound[sig.num_args:2*sig.num_args] = sig.falling_lower_bound
		output_upper_bound[:sig.num_args] = sig.rising_upper_bound
		output_upper_bound[sig.num_args:2*sig.num_args] = sig.falling_upper_bound

	input_init[1]-=sig.default_pulse_length/2
	input_init[sig.num_args+1]+=sig.default_pulse_length/2
	output_init[1]-=sig.default_pulse_length/2
	output_init[sig.num_args+1]+=sig.default_pulse_length/2
	
	input_sigma = [1]*len(data[0]) # Possibilty of giving the datapoints differend weights.
	output_sigma = [1]*len(data[0]) 
	
	input_params = optimize.curve_fit(pulse_wr, data[0], data[1], input_init, sigma = input_sigma, bounds = (input_lower_bound, input_upper_bound), maxfev=5000)
	output_params = optimize.curve_fit(pulse_wr, data[0], data[2], output_init, sigma = output_sigma, bounds = (output_lower_bound, output_upper_bound), maxfev=5000)
	
	
	#lowside_params = optimize.curve_fit(lowside_pulse_wr, data[0], data[lowside_index], input_init, sigma = input_sigma, bounds = (sig.input_lower_bound, sig.input_upper_bound), maxfev=5000)
	#highside_params = optimize.curve_fit(highside_pulse_wr, data[0], data[highside_index], output_init, sigma = output_sigma, bounds = (sig.output_lower_bound, sig.output_upper_bound), maxfev=5000)
	
	
	#lowside_fitting = [0]*len(data[0])
	#highside_fitting = [0]*len(data[0])
	
	
	input_fitting = [0]*len(data[0])
	output_fitting = [0]*len(data[0])
	input_fitting_error = [0]*len(data[0])
	output_fitting_error = [0]*len(data[0])
	
	# Calculate the RMS error of the resulting function scaled by the operating voltage.
	input_rms_error = aux.calc_rms_error_func(pulse, input_params[0], data[0], data[1])/Voltage
	output_rms_error = aux.calc_rms_error_func(pulse, output_params[0], data[0], data[2])/Voltage
	
	#lowside_rms_error = aux.calc_rms_error_func(lowside_pulse, lowside_params[0], data[0], data[lowside_index])/Voltage
	#highside_rms_error = aux.calc_rms_error_func(highside_pulse, highside_params[0], data[0], data[highside_index])/Voltage
	
	#input_rms_error = lowside_rms_error
	#output_rms_error = highside_rms_error
	#if lowside_index == 2:
		#input_rms_error = highside_rms_error
		#output_rms_error = lowside_rms_error

	if visualize:
		plt.cla()
		plt.clf()
		fig = plt.gcf()
		fig.set_size_inches(8, 6)
		for i in range(0,len(data[0])):
			input_fitting[i] = pulse(data[0][i],input_params[0])
			output_fitting[i] = pulse(data[0][i],output_params[0])
			'''lowside_fitting[i] = lowside_pulse(data[0][i],lowside_params[0])
			highside_fitting[i] = highside_pulse(data[0][i],highside_params[0])
			if lowside_index == 1:
				input_fitting_error[i] = lowside_fitting[i] - data[1][i]
				output_fitting_error[i] = highside_fitting[i] - data[2][i]
			else:
				input_fitting_error[i] = highside_fitting[i] - data[1][i]
				output_fitting_error[i] = lowside_fitting[i] - data[2][i]'''
			input_fitting_error[i] = input_fitting[i] - data[1][i]
			output_fitting_error[i] = output_fitting[i] - data[2][i]
			
		linew = 1
		
		
			
		plt.plot(data[0],data[1],'r-', linewidth=linew)
		'''if lowside_index == 1:
			plt.plot(data[0],lowside_fitting,'r--', linewidth=linew)
		else:
			plt.plot(data[0],highside_fitting,'r--', linewidth=linew)'''
		plt.plot(data[0],input_fitting,'r--', linewidth=linew)
		plt.plot(data[0],input_fitting_error,'r-.', linewidth=linew)	
		plt.plot(data[0],data[2],'g-', linewidth=linew)	
		'''if lowside_index == 1:
			plt.plot(data[0],highside_fitting,'g--', linewidth=linew)
		else:
			plt.plot(data[0],lowside_fitting,'g--', linewidth=linew)'''
		plt.plot(data[0],output_fitting,'g--', linewidth=linew)
		plt.plot(data[0],output_fitting_error,'g-.', linewidth=linew)	
		
		file_name = path.split("/")[len(path.split("/"))-1]
		title = file[:len(file)-4]
		plt.title(title)
		plt.legend(["Input","Input fitting","Input fitting error", "Output","Output fitting","Output fitting error"], loc = 'center left')
		plt.ylabel("Voltage [V]")
		plt.xlabel("Time [s]")
		#plt.legend(["Input","Output","Input Rising Edge","Input Falling Edge","Output Rising Edge","Output Falling Edge", "Vdd/2"], loc = 'center left')
		plt.text(0, Voltage/4, "In  RMSE: " + str(round(input_rms_error,5)) + "\nOut RMSE: " + str(round(output_rms_error,5)),family="monospace")
		#plt.show()
		imgpath = path[:len(path)-3]
		plt.savefig(imgpath + "svg")	
	
	arr_len = len(input_params[0])
	ret = [0]*(2*arr_len+2)
	for i in range(0, arr_len):
		'''if lowside_index == 1:
			ret[i] = lowside_params[0][i]
			ret[input_arr_len+i] = highside_params[0][i]
		else:
			ret[input_arr_len+i] = lowside_params[0][i]
			ret[i] = highside_params[0][i]'''
		ret[i] = input_params[0][i]
		ret[arr_len+i] = output_params[0][i]
	ret[arr_len*2] = input_rms_error
	ret[arr_len*2+1] = output_rms_error
		
	return ret

f = []
for (dirpath, dirnames, filenames) in walk(folderpath):
    f = filenames
    break

fw = open(folderpath + "/fitting_parameters.txt", "w+")
input_param_names = ['']*(2*sig.num_args)
output_param_names = ['']*(2*sig.num_args)
for i in range(0,sig.num_args):
	input_param_names[i] = "left_" + sig.parameter_names[i]
	input_param_names[sig.num_args + i] = "right_" + sig.parameter_names[i]
	output_param_names[i] = "left_" + sig.parameter_names[i]
	output_param_names[sig.num_args + i] = "right_" + sig.parameter_names[i]
input_param_names[1] = "shift"
input_param_names[sig.num_args+1] = "length"
output_param_names[1] = "delay"
output_param_names[sig.num_args+1] = "length"
print("Filename : [ input_" + " input_".join(input_param_names) + " output_" + " output_".join(output_param_names) + " input_RMSE output_RMSE]")

start = timer()

sum_of_error = [0]*2 # 0 = input, 1 = output
max_error = [0]*2
count = 0
for file in filenames: # Iterate through every file in the specified folder and fit the .dat files.
	if file.split(".")[1] == "dat":
		res = get_params(True, folderpath + "/" + file)
		
		sum_of_error[0] = sum_of_error[0] + res[len(res)-2]
		sum_of_error[1] = sum_of_error[1] + res[len(res)-1]
		if res[len(res)-2] > max_error[0]:
			max_error[0] = res[len(res)-2]
		if res[len(res)-1] > max_error[1]:
			max_error[1] = res[len(res)-1]
		
		
		# Parameter correction
		# instead of the absoulte shift values obtained from the fitting algorithm the shift values get substracted from each other for better physical interpretation
		# input pulse length = absolute shift of the right input sigmoid - absolute shift of the left input sigmoid
		# output pulse length = absolute shift of the right output sigmoid - absolute shift of the left output sigmoid
		# output delay = absolute shift of the left output sigmoid - absolute shift of the left input sigmoid		
		# Note that all the shift values describe the shifts of the single sigmoids and not fitted curve as a whole.
		# Example: An input pulse shift of 2ns means that the left switching waveform of the input pulse crosses Vdd/2 at 2ns, but the input pulse as whole need not cross Vdd at this point.
		res[sig.num_args+1] = res[sig.num_args+1]-res[1] # Calculate input length from shifts
		res[sig.num_args*2 + sig.num_args + 1] = res[sig.num_args*2 + sig.num_args + 1]-res[sig.num_args*2 + 1]
		res[sig.num_args*2 + 1] = res[sig.num_args*2 + 1] - res[1]
		
		
		res = np.round_(np.array(res),decimals = 7)
		np.set_printoptions(linewidth =300,suppress=True)
		
		fw.write(file + "," + ','.join(map(str, res))  + "\n")
		
		print(file + " : " + str(res))	
		count = count + 1
		
end = timer()

sum_of_error[0] = sum_of_error[0]/count
sum_of_error[1] = sum_of_error[1]/count

fw.write("fitting function," + sig_name + "\n")
fw.write("time needed," + str(round(end-start,2)) + "s\n")
fw.write("average input fitting error," + str(round(sum_of_error[0],5)) + "\n")
fw.write("average output fitting error," + str(round(sum_of_error[1],5)) + "\n")
fw.write("max input fitting error," + str(round(max_error[0],5)) + "\n")
fw.write("max output fitting error," + str(round(max_error[1],5)) + "\n")
fw.close()

print("fitting function: " + sig_name)
print("time needed: " + str(round(end-start,2)) + "s")
print("average input fitting error: " + str(round(sum_of_error[0],5)))
print("average output fitting error: " + str(round(sum_of_error[1],5)))
print("max input fitting error: " + str(round(max_error[0],5)))
print("max output fitting error: " + str(round(max_error[1],5)))
