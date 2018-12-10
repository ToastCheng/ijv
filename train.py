
from argparse import ArgumentParser

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from inverse.model import SCVNet
from inverse.dataset import SimData



def get_args():
	parser = ArgumentParser()

	parser.add_argument("-l", "--learning_rate", type=float, default=1e-4)
	parser.add_argument("-e", "--epoch", type=int, default=100)
	parser.add_argument("-p", "--pretrain", type=int, default=0)
	parser.add_argument("-m", "--model", type=str, default="")
	parser.add_argument("-b", "--batch_size", type=int, default=32)

	return parser.parse_args()


def train():
	args = get_args()

	net = SCVNet()

	net.load_state_dict(torch.load("saved_model/model.pth"))

	dataset = SimData()
	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

	loss_func = nn.MSELoss()
	optimizer = optim.Adam(params=net.parameters(), lr=args.learning_rate)

	losses = []

	for e in range(args.epoch):
		for idx, data in enumerate(loader):

			spectrum, geometry, parameter = data[0]
			scv = data[1]

			# default is double, but float is required!
			spectrum = spectrum.float()
			geometry = geometry.float()
			parameter = parameter.float()
			scv = scv.float()


			net.zero_grad()

			pred = net(spectrum, geometry, parameter)
			
			
			loss = loss_func(pred, scv)

			loss.backward()
			optimizer.step()

		losses.append(loss)

		if e%5 == 0:
			print("epoch: %d | loss: %f.4" %(e, loss))
			print("predict: ", pred[:5])
			print("target: ", scv[:5])


	plt.plot(losses)
	plt.xlabel('epoch')
	plt.ylabel('rms error')
	plt.savefig('log.png')

	torch.save(net.state_dict(), "saved_model/model.pth")



if __name__ == "__main__":
	train()



