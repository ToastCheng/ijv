import torch 
import torch.nn as nn



class SCVNet(nn.Module):
	def __init__(self, spectrum_size=31, geometry_size=5):
		super(SCVNet, self).__init__()

		self.encoder = nn.Sequential(
			nn.Linear(),
			nn.BatchNorm1d(),
			nn.ReLU(),
			nn.Linear()
		)

		self.fc = nn.Sequential(
			nn.Linear(),
			nn.BatchNorm1d(),
			nn.ReLU(),
			nn.Linear(),
			nn.BatchNorm1d(),
			nn.ReLU(), 
			nn.Linear()
		)



	def forward(self, spectrum, geometry):
		# spectrum: [batch, spectrum_size]
		# geometry: [batch, geometry_size]

		# encode the geometry
		geometry = self.encoder(geometry)

		# spectrum flatten
		spectrum = spectrum.view()

		processed_input = torch.cat([spectrum, geometry], 1)

		out = self.fc(processed_input)

		return out
