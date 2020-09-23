class Pulse_Set:
	def __init__(self, file_name, time, input_pulse, output_pulse):
		self.file_name = file_name
		self.time = time
		self.input_pulse = input_pulse
		self.output_pulse = output_pulse
		
class Parameter_Set:
	def __init__(self, file_name, parameters):
		self.file_name = file_name
		self.parameters = parameters		