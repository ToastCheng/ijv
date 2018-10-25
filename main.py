from argparse import ArgumentParser
import os

import utils
from mcx import MCX



def get_args():
	parser = ArgumentParser()
	parser.add_argument("-c", "--config", type=str, default="config.json")
	parser.add_argument("-f", "--forward", action="store_true")
	parser.add_argument("-i", "--inverse", action="store_true")

	return parser.parse_args()


def main():
	args = get_args()
	if args.forward:

		config_file = args.config
		forward(config_file)
	
	if args.inverse:

		inverse()


def forward(config):

	# run mcx!
	mcx = MCX(config)
	mcx.run()
	mcx.calculate_reflectance()


def inverse():
	pass


if __name__ == "__main__":
	main()
	os.system("sudo shutdown")
