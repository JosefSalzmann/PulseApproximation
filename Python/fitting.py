import auxiliary as aux
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from os import walk
from mpl_toolkits import mplot3d
from timeit import default_timer as timer


import importlib
sig = importlib.import_module("atan_taylor", package=None)
Voltage = 1.2
num_points = 500
folderpath = "../WaveformData/t4_u_all/"

Lin_approx_order = 3

def lowside_pulse(t, args): 
	args_rising = args[0:len(sig.input_initial)/2]
	args_falling = args[(len(sig.input_initial)/2):len(sig.input_initial)]
	return (Voltage*(sig.sigmoid(t, args_rising) + sig.sigmoid(t, args_falling)) - Voltage)

def highside_pulse(t, args): 
	args_rising = args[(len(sig.input_initial)/2):len(sig.input_initial)]
	args_falling = args[0:len(sig.input_initial)/2]
	return Voltage*(sig.sigmoid(t, args_rising) + sig.sigmoid(t, args_falling))
	
def lowside_pulse_wr(t,*args): 
	return lowside_pulse(t,args)
	
def highside_pulse_wr(t,*args): 
	return highside_pulse(t,args)
	
	
def get_params(visualize, path, input_init, output_init):
	data = aux.read_file(path, num_points)
	
	lowside_index = 1
	highside_index = 2
	
	if data[1][len(data[1])-1] > Voltage / 2 and data[1][0] > Voltage / 2:
		lowside_index = 2
		highside_index = 1
	
	input_sigma = [1]*len(data[0])
	output_sigma = [1]*len(data[0]) 
	'''
	max = 0;
	min = Voltage;
	for i in range(0, len(data[0])):
		if data[lowside_index][i] > max:
			max = data[lowside_index][i]
		if data[highside_index][i] < min:
			min = data[highside_index][i]
	
	for i in range(0, len(data[0])):	# top 20% of pulse and bottom 1% to 20%  get weighted 2 times stronger
		if data[lowside_index][i] > max*0.8 or (data[lowside_index][i] > max*0.01 and data[lowside_index][i] < max*0.2):
			input_sigma[i] = 0.5
			
		if data[highside_index][i] < ((Voltage-min)*0.2+min) or (data[highside_index][i] < ((Voltage-min)*0.99+min) and data[highside_index][i] > ((Voltage-min)*0.8+min)):
			output_sigma[i] = 0.5
	'''	

	input_params = optimize.curve_fit(lowside_pulse_wr, data[0], data[lowside_index], input_init, sigma = input_sigma, bounds = (sig.input_lower_bound, sig.input_upper_bound), maxfev=3000)
	output_params = optimize.curve_fit(highside_pulse_wr, data[0], data[highside_index], output_init, sigma = output_sigma, bounds = (sig.output_lower_bound, sig.output_upper_bound), maxfev=3000)
	

	input_fitting = [0]*len(data[0])
	output_fitting = [0]*len(data[0])
	
	inp_rising = [0]*len(data[0])
	
	input_rms_error = aux.calc_rms_error_func(lowside_pulse, input_params[0], data[0], data[lowside_index])
	output_rms_error = aux.calc_rms_error_func(highside_pulse, output_params[0], data[0], data[highside_index])

	
	if visualize:
		plt.cla()
		plt.clf()
		fig = plt.gcf()
		fig.set_size_inches(8, 6)
		for i in range(0,len(data[0])):
			input_fitting[i] = lowside_pulse(data[0][i],input_params[0])
			output_fitting[i] = highside_pulse(data[0][i],output_params[0])
			
		plt.plot(data[0],data[lowside_index],'r-')
		plt.plot(data[0],input_fitting,'r--')
		plt.plot(data[0],data[highside_index],'g-')		
		plt.plot(data[0],output_fitting,'g--')	
		
		file_name = path.split("/")[len(path.split("/"))-1]
		title = file[:len(file)-4]
		plt.title(title)
		plt.legend(["Input","Input fitting","Output","Output fitting"], loc = 'center left')
		#plt.legend(["Input","Output","Input Rising Edge","Input Falling Edge","Output Rising Edge","Output Falling Edge", "Vdd/2"], loc = 'center left')
		plt.text(0, Voltage/4, "In  RMSE: " + str(round(input_rms_error,5)) + "\nOut RMSE: " + str(round(output_rms_error,5)),family="monospace")
		#plt.show()
		imgpath = path[:len(path)-3]
		plt.savefig(imgpath + "png")	
	
	input_arr_len = len(input_params[0])
	output_arr_len = len(output_params[0])
	ret = [0]*(input_arr_len+output_arr_len+2)
	for i in range(0,input_arr_len):
		ret[i] = input_params[0][i]
		ret[input_arr_len+i] = output_params[0][i]
	ret[input_arr_len+output_arr_len] = input_rms_error
	ret[input_arr_len+output_arr_len+1] = output_rms_error
		
	return ret

f = []
for (dirpath, dirnames, filenames) in walk(folderpath):
    f = filenames
    break
	
fw = open(folderpath + "/fitting_parameters.txt", "w+")
params = [[0 for i in range(200)] for j in range(2*len(sig.input_initial)+1)]
count = 0

input_init = sig.input_initial
output_init = sig.output_initial
start = timer()
for file in filenames:
	if file.split(".")[1] == "dat" and count < 200:
		res = get_params(True, folderpath + "/" + file, input_init, output_init)
		
		#input_init = res[:len(sig.input_initial)]
		#output_init = res[len(sig.input_initial):len(sig.input_initial)*2]
		
		res = np.round_(np.array(res),decimals = 7)
		np.set_printoptions(linewidth =300,suppress=True)
		
		
		#parameter korrektur
		res[sig.input_length] = res[sig.input_length]-res[sig.input_shift]
		#res[len(sig.input_initial) + sig.output_length] = res[len(sig.input_initial) + sig.output_length]-res[len(sig.input_initial) + sig.output_shift]
		#res[len(sig.input_initial) + sig.output_shift] = res[len(sig.input_initial) + sig.output_shift]-res[sig.input_shift]
		fw.write(file + " : " + ','.join(map(str, res))  + "\n")
		
				
		
		
		arg_c = 0
		for i in range(0,len(res)):
			
			if i != sig.input_shift:
				params[arg_c][count] = res[i]				
				arg_c = arg_c + 1
		print(file + " : " + str(res))
		
		
		
		count = count + 1
fw.close()
end = timer()


for i in range(0,len(params)):
	params[i] = params[i][:count]

print(end-start)







'''
def meta_func(X,args):
	ret_val = args[0]
	arg_count = 1
	for i in range(0, len(sig.input_initial)-1):
		for j in range(1, Lin_approx_order+1):
			ret_val = ret_val + args[arg_count]*(X[i]**j)
			arg_count = arg_count + 1
	return ret_val

def meta_func_wr(X, *args):
	return meta_func(X,args)
	
meta_params = [[1 for i in range(1+(len(sig.input_initial)-1)*Lin_approx_order)] for j in range(len(sig.input_initial))]

for i in range(0, len(sig.input_initial)):
	meta_params[i] = optimize.curve_fit(meta_func_wr, params[:len(sig.input_initial)-1], params[len(sig.input_initial)-1+i], meta_params[i])[0]
	print(meta_params[i])
	

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

rms_errors = [0 for i in range(len(sig.input_initial))]
for i in range(0, len(sig.input_initial)):
	rms_errors[i] = np.sqrt(sum(fitting_sq_errors[i])/len(fitting_sq_errors[i]))
	print(rms_errors[i])
	
meta_output_params = [[0 for i in range(len(sig.input_initial))] for j in range(count)]
for i in range(0,count):
	for j in range(0,len(sig.input_initial)):
		meta_output_params[i][j] = meta_fittings[j][i]

print(meta_output_params[0])

count = 0
for file in filenames:
	if file.split(".")[1] == "dat":
		data = aux.read_file(folderpath  + file, num_points)
		input_fitting = [0]*len(data[0])
		output_fitting = [0]*len(data[0])
		meta_output_fitting = [0]*len(data[0])
		for i in range(0,len(data[0])):
				#input_fitting[i] = lowside_pulse(data[0][i],input_params[0])
				#output_fitting[i] = highside_pulse(data[0][i],output_params[0])
				meta_output_fitting[i] = highside_pulse(data[0][i],meta_output_params[count])
		
		plt.cla()
		plt.clf()
		plt.plot(data[0],data[1],'r-')
		#plt.plot(data[0],input_fitting,'r-')
		plt.plot(data[0],data[2],'g-')		
		#plt.plot(data[0],output_fitting,'g--')	
		plt.plot(data[0],meta_output_fitting,'b--')	
		
		plt.title("Meta fitting")
		plt.legend(["Input","Output", "Meta Output fitting"], loc = 'center left')
		#plt.show()
		path = folderpath + file
		imgpath = path[:len(path)-4]
		plt.savefig(imgpath + "_Meta_fitting.png")
		count = count + 1
'''