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
		if len(line[i].split(":"))>=2:
			line[i] = line[i].split(":")[1]
			#line[i] = line[i][:len(line[i])-1]
			line_cnt = line_cnt + 1
	line_len = len(line[0].split(','))
	ret = [[0 for i in range(line_cnt-1)] for j in range(line_len)]
	for i in range(0,line_cnt-1):
		line_split = line[i].split(',');
		for j in range(0,line_len):
			ret[j][i] = float(line_split[j].strip());	
	return ret;
	
	
	
	
	
	

params = read_file("../WaveformData/t4_u_all/");
	
# visualize parameters depending on input pulse length
colors = ['b-','g-','r-','c-','m-','y-','k-','b--','g--','r--','c--','m--','y--','k--']

names = ['']*(len(params)-1)
arg_c = 0
for i in range(0, len(sig.input_initial)):
	if i != sig.input_shift:
		names[arg_c] = "input_" + str(sig.parameter_names[i]);
		arg_c = arg_c + 1
	names[i+len(sig.input_initial)-1] = "output_" + str(sig.parameter_names[i]);
names[sig.input_length-1] = "input_pulse_length" ;
	
plt.cla()
plt.clf()
for i in range(0, len(params)-2):
	if i != sig.input_shift:
		plt.plot(params[sig.input_length],params[i],colors[i%len(colors)]);
plt.legend(names, loc = 'upper left');
plt.show();

			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			