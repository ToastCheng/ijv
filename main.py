import os
import json 
from glob import glob
from argparse import ArgumentParser

import utils
from mcx import MCX
from generator import Generator



def get_args():
	parser = ArgumentParser()
	parser.add_argument("-c", "--config", type=str, default="config.json")
	parser.add_argument("-f", "--forward", action="store_true")
	parser.add_argument("-i", "--inverse", action="store_true")
	parser.add_argument("-t", "--train", action="store_true")


	return parser.parse_args()


def main():
	args = get_args()
	if args.forward:

		config_file = args.config
		forward(config_file)
	
	if args.inverse:

		inverse()

	if args.train:

		train()


def forward(config):

	# run mcx!
	mcx = MCX(config)
	mcx.run()
	mcx.calculate_reflectance()


def inverse():
	pass


def train():

	train_list = glob(os.path.join('generator', 'parameter', '*'))
	train_list.sort()
	config_train = "config_train.json"

	gen = Generator()
	# for idx in range(100):
	# 	gen.run(idx=idx)

	for idx, parameter in enumerate(train_list):
		with open('config_train.json', 'rb') as f:
			config = json.load(f)
			config["session_id"] = "train_%d" % idx
			config["input_file"] = parameter
		with open('config_train.json', 'w') as f:
			json.dump(config, f, indent=4)

		mcx = MCX("config_train.json")
		mcx.run()
		mcx.calculate_reflectance(plot=False)
		

	# from pprint import PrettyPrinter
	# pp = PrettyPrinter()
	# pp.pprint(train_list)


if __name__ == "__main__":
	main()
	os.system("sudo shutdown")
