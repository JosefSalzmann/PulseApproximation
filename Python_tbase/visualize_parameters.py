import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import exp as sig

def read_file(filepath):
	f = open(filepath + "/fitting_parameters.txt", "r")
	fr = f.read()
	line = fr.split("\n")
	n = len(line)
	line_cnt = 0
	for i in range(0,n):
		if len(line[i].split(","))>=3:
			line[i] = line[i][len(line[i].split(",")[0])+1:len(line[i])]
			#line[i] = line[i][:len(line[i])-1]
			line_cnt = line_cnt + 1
		else:
			break
	line_len = len(line[0].split(','))
	ret = [[0 for i in range(line_cnt-1)] for j in range(line_len)]
	for i in range(0,line_cnt-1):
		line_split = line[i].split(',')
		for j in range(0,line_len):
			ret[j][i] = float(line_split[j].strip());	
	return ret
	
	
	
	
	
	

params = read_file("../WaveformData/t4_d/")
	
# visualize parameters depending on input pulse length
colors = ['b-','g-','r-','c-','m-','y-','k-','b--','g--','r--','c--','m--','y--','k--']

names = ['']*(len(params)-1)
arg_c = 0
for i in range(0, sig.num_args):
	if i != 1:
		names[arg_c] = "input_left_" + str(sig.parameter_names[i])
		arg_c = arg_c + 1
	names[i+sig.num_args-1] = "input_right_" + str(sig.parameter_names[i])
	names[i+2*sig.num_args-1] = "output_left_" + str(sig.parameter_names[i])
	names[i+3*sig.num_args-1] = "output_right_" + str(sig.parameter_names[i])
names[sig.num_args] = "input_pulse_length"
	
plt.cla()
plt.clf()
for i in range(0, len(params)-2):
	if i != 1:
		plt.plot(params[sig.num_args+1],params[i],colors[i%len(colors)])


#names = ['input_left_steepness', 'input_right_steepness', 'output_left_steepness', 'output_right_steepness', 'const = -1.44']
plt.legend(names, loc = 'center left')
plt.title("Steepness parameter für t4_d in Abhängigkeit der Pulslänge")
plt.show()

			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			