from os import walk
import fit_trace
import os
from timeit import default_timer as timer

transitionIndex = 0

folderpath = "../WaveformData/t4_u_all"
dir_path = os.path.dirname(os.path.realpath(__file__))

f = []
for (dirpath, dirnames, filenames) in walk(dir_path + "/" + folderpath):
    f = filenames
    break
f.sort()

start = timer()

for file in filenames: # Iterate through every file in the specified folder and fit the .dat files.
    if file.split(".")[1] == "dat":
        fit_trace.main(folderpath + "/" + file, "exp", 1.2, True)

end = timer()
print("time needed: " + str(round(end-start,2)) + "s")


f = []
for (dirpath, dirnames, filenames) in walk(dir_path + "/" + folderpath):
    f = filenames
    break
f.sort()

write_str = []

for file in filenames: # Write the fitting results (i.e. the transitions of interest) to one file
    if file[len(file)-12:len(file)] == "_fitting.csv":
        fittingFile = open(dir_path + "/" + folderpath + "/" + file, "r")
        fititngFileContet = fittingFile.read()
        lines = fititngFileContet.split("\n")
        outputIndex = 0
        while lines[outputIndex] != "Output parameters":
            outputIndex+=1
        RMSIndex = 0
        while lines[RMSIndex].split(",")[0] != "Input RMSE":
            RMSIndex+=1
        write_str.append(file + "," + lines[2+transitionIndex] + "," + lines[2+transitionIndex+1] + "," + 
                         lines[outputIndex+2+transitionIndex] + "," + lines[outputIndex+2+transitionIndex+1] + "," + 
                         lines[RMSIndex].split(",")[1] + "," + lines[RMSIndex+1].split(",")[1])

fw = open(dir_path + "/" + folderpath + "/fitting_parameters.txt", "w+")
fw.write('\n'.join(write_str))
fw.close()