import torch 
import torch.nn as nn


class SCVNet(nn.Module):
	def __init__(self, spectrum_size=36, geometry_size=6, parameter_size=23):
		super(SCVNet, self).__init__()

		self.encoder_geometry = nn.Sequential(
			nn.Linear(geometry_size, 50),
			nn.BatchNorm1d(50),
			nn.ReLU(),
			nn.Linear(50, 50)
		)

		self.encoder_spectrum = nn.Sequential(
			nn.Linear(spectrum_size, 50),
			nn.BatchNorm1d(50),
			nn.ReLU(),
			nn.Linear(50, 50)
		)

		self.encoder_parameter = nn.Sequential(
			nn.Linear(parameter_size, 50),
			nn.BatchNorm1d(50),
			nn.ReLU(),
			nn.Linear(50, 50)
		)

		self.fc = nn.Sequential(
			nn.Linear(150, 500),
			nn.BatchNorm1d(500),
			nn.ReLU(),
			nn.Linear(500, 500),
			nn.BatchNorm1d(500),
			nn.ReLU(), 
			nn.Linear(500, 1)
		)



	def forward(self, spectrum, geometry, parameter):
		# spectrum: [batch, spectrum_size]
		# geometry: [batch, geometry_size]

		# encode the geometry

		# [batch, 50]
		geometry = self.encoder_geometry(geometry)

		# [batch, 50]
		parameter = self.encoder_parameter(parameter)

		# spectrum flatten
		spectrum = spectrum.view(spectrum.shape[0], -1)

		# [batch, 50]
		spectrum = self.encoder_spectrum(spectrum)

		# [batch, 100]
		processed_input = torch.cat([spectrum, geometry, parameter], 1)

		out = self.fc(processed_input)

		return out



class SCVNetV1(nn.Module):
	def __init__(self, spectrum_size=36, geometry_size=6, parameter_size=23):
		super(SCVNet, self).__init__()

		self.encoder_geometry = nn.Sequential(
			nn.Linear(geometry_size, 50),
			nn.BatchNorm1d(50),
			nn.ReLU(),
			nn.Linear(50, 50)
		)

		self.encoder_spectrum = nn.Sequential(
			nn.Linear(spectrum_size, 50),
			nn.BatchNorm1d(50),
			nn.ReLU(),
			nn.Linear(50, 50)
		)

		self.encoder_parameter = nn.Sequential(
			nn.Linear(parameter_size, 50),
			nn.BatchNorm1d(50),
			nn.ReLU(),
			nn.Linear(50, 50)
		)

		self.fc = nn.Sequential(
			nn.Linear(150, 100),
			nn.BatchNorm1d(100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.BatchNorm1d(100),
			nn.ReLU(), 
			nn.Linear(100, 1)
		)



	def forward(self, spectrum, geometry, parameter):
		# spectrum: [batch, spectrum_size]
		# geometry: [batch, geometry_size]

		# encode the geometry

		# [batch, 50]
		geometry = self.encoder_geometry(geometry)

		# [batch, 50]
		parameter = self.encoder_parameter(parameter)

		# spectrum flatten
		spectrum = spectrum.view(spectrum.shape[0], -1)

		# [batch, 50]
		spectrum = self.encoder_spectrum(spectrum)

		# [batch, 100]
		processed_input = torch.cat([spectrum, geometry, parameter], 1)

		out = self.fc(processed_input)

		return out
