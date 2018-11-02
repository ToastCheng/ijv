import json 
from random import uniform
from collections import namedtuple
from mcx import MCX



class Generator:
	"""
	Author: Shih Cheng Tu

	A helper class that generate random input for MCX
	"""
	def __init__(self):
		
		# a light-weight structure (using collections.namedtuple)
		self.Parameter = namedtuple('parameters', ['max', 'min'])

		# a dict that contain parameters
		self.parameters = {}

		self.mcx = MCX()


	def run(self, config_file=None):
		if config_file is not None:
			self._load(config_file)

		assert len(self.parameters) > 0, "no parameter is set in the Generator"

		with open("parameters_gen.json", "rb") as f:
			parameters = json.load(f)

		for name, value in self.parameters.items():
			rand = uniform(value.min, value.max)
			








		
	def add_parameter(self, name, max, min):
		self.parameters[name] = self.Parameter(max, min)

	def _load(self, config_file):
		with open(config_file, 'rb') as f:
			config = json.load(f)

		for name, value in config.items():
			self.add_parameter(name, value["max"], value["min"])







