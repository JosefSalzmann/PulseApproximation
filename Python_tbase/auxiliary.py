import math
import numpy as np
import classes as cl

def read_file(path, num_points):
	f = open(path, "r")
	fr = f.read()
	line = fr.split(";\n")
	n = min(num_points, len(line))
	line[0] = line[0][1:]
	line[len(line)-1] = line[len(line)-1][:len(line[len(line)-1])-1]
	
	time = [0]*n
	input = [0]*n
	output = [0]*n
	for i in range(0,n,1):
		index = int(math.floor(len(line)/n)*i)
		number = line[index].split(" ")
		time[i] = float(number[0])
		input[i] = float(number[1])
		output[i] = float(number[2])
		
	return [time, input, output]
	
def read_file_with_name(path, num_points):
	f = open(path, "r")
	fr = f.read()
	line = fr.split(";\n")
	n = min(num_points, len(line))
	line[0] = line[0][1:]
	line[len(line)-1] = line[len(line)-1][:len(line[len(line)-1])-1]
	
	time = [0]*n
	input = [0]*n
	output = [0]*n
	for i in range(0,n,1):
		index = int(math.floor(len(line)/n)*i)
		number = line[index].split(" ")
		time[i] = float(number[0])
		input[i] = float(number[1])
		output[i] = float(number[2])
		
	path_split = path.split('/')
	filename = path_split[len(path_split)-1]
	return cl.Pulse_Set(filename, time, input, output)
	
def calc_rms_error_func(fitting_func, params, x, original):
	fitting = [0]*len(original)
	for i in range(0,len(original),1):
		fitting[i] = fitting_func(x[i], params)
	sq_error = np.subtract(original,fitting)**2	
	return np.sqrt(sum(sq_error)/len(sq_error))
	
	
def calc_rms_error_data(original, fitted):
	sq_error = np.subtract(original,fitted)**2	
	return np.sqrt(sum(sq_error)/len(sq_error))

def red_transfer_fnc_parameters(path):
	f = open(path, "r")
	fr = f.read()
	line = fr.split("\n")
	n = len(line)
	blank_indices_list = [0]
	for i in range (0,n): # parameters are seperated by blanks
		if line[i] == '':
			if blank_indices_list[len(blank_indices_list)-1] == (i):
				break
			else:
				blank_indices_list.append(i+1)
	del blank_indices_list[len(blank_indices_list)-1] # remove last element
	tr_fnc_parameters = [[0.0 for i in range(blank_indices_list[1]-2)] for j in range(len(blank_indices_list))]
	for i in range(0, len(blank_indices_list)):
		index = blank_indices_list[i]+1
		for j in range(0, blank_indices_list[1]-2):
			line_split = line[index+j].split(",")
			param = line_split[len(line_split)-1]
			tr_fnc_parameters[i][j] = float(param)
	return tr_fnc_parameters
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	