### Convert all .svg files in a folder to .pdf_latex (so it works with the template)
import os

graphics_folder = "C:/Users/josef/Desktop/BscArbeit/Template/graphics"

f = []
for (dirpath, dirnames, filenames) in os.walk(graphics_folder):
    f = filenames
    break

for file in filenames:
    if file.split(".")[1] == "svg":
        # os.stat("file.dat") für dateiinformationen
        print("Converting " + file +  " to .pdf_tex...", end = '', flush=True)
        os.system('cmd /c \"\"C:/Program Files/Inkscape/inkscape\" -D -z --file=\"' + graphics_folder + '/' + file + 
                  '\" --export-pdf=\"' + graphics_folder + '/' + file.split(".")[0] + '.pdf\" --export-latex\"')
        print(" done", flush=True)

# string = '\"C:/Program Files/Inkscape/inkscape\" -D -z --file=\"C:/Users/josef/Desktop/BscArbeit/Template/graphics/inv_t4_u000440000Traces.svg\" --export-pdf=\"C:/Users/josef/Desktop/BscArbeit/Template/graphics/inv_t4_u000440000Traces.pdf\" --export-latex'

# os.system('cmd /c \"' + string + "\"")