import csv
import os 
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv3d, Module, Softmax, BatchNorm3d, BatchNorm2d, Conv2d, Dropout
from rand_walk_code import *
import random
import time

# Hyperparameters
num_epochs = 500
batch_size = 32
learning_rate = 0.001
# With 1 conv layer, converge to like 13 or something at 0.001, but jumps
# around a bunch
class Net(Module):
	def __init__(self, grid_size, output_size):
		super().__init__()
		self.conv_layers = Sequential(
			Conv2d(50, 16, kernel_size=3, stride=2, padding=1),
			BatchNorm2d(num_features=16),
			ReLU(),
			#Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
			#BatchNorm2d(num_features=16),
			#ReLU(),
			#Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
			#BatchNorm2d(num_features=16),
			#ReLU(),
			#Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
			#BatchNorm2d(num_features=16),
			#ReLU(),
			Conv2d(16, 1, kernel_size=1, stride=1))
		self.output_layer = Linear(int(grid_size / 2) ** 2, output_size)
		#some funky shit is going on with linear out of convolutional
		# self.conv_layers = Sequential(Linear((grid_size * grid_size * 50), 500),
		# 	Linear(500, 200),
		# 	Linear(200, 200),
		# 	Linear(200, 100))
		#self.output_layer = Linear(100, output_size)

	def forward(self, x):
		out = self.conv_layers(x)
		out = out.reshape(out.size(0), -1)
		out = self.output_layer(out)
		return out

def train(train_loader, val_loader):
	grid_size = None
	output_size = None
	print("here")
	for (walks, labels) in train_loader:
		grid_size = walks.shape[2]
		print("should be n")
		print(grid_size)
		output_size = labels.shape[1]
		print("should be num_properties")
		print(output_size)
		break
	model = Net(grid_size, output_size)
	model = model.double()
	loss_function = nn.L1Loss()
	#NOTE: If certain properties are on smaller or larger scales, their importance
	#in loss may be too little or too much. Should the values be normalized by an
	#invertible function?

	# Optimizer takes in the parameters to optimize
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	num_steps = len(train_loader)
	loss_list = []
	val_loss_list = []
	last_time = time.time()
	for epoch in range(num_epochs):
		for i, (walks, labels) in enumerate(train_loader):
			print(i)
			last_time = time.time()
			outputs = model(walks)
			loss = loss_function(outputs, labels)
			loss_list.append(loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			total = labels.size(0)
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, i + 1, num_steps, loss.item()))
	model.eval()
	num_steps = len(val_loader)
	with torch.no_grad():
		for i, (walks, labels) in enumerate(val_loader):
			outputs = model(walks)
			loss = loss_function(outputs, labels)
			val_loss_list.append(loss.item())

			total = labels.size(0)
			if ((i + 1) % 100) == 0:
				print('Epoch [{}/{}], Step [{}/{}], Validation Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, num_steps, loss.item()))
	model.train()
	return model

def test(model, test_loader):
		loss_function = nn.L1Loss()
		test_loss_list = []
		num_steps = len(test_loader)
		model.eval()
		# torch.no_grad disables autograd I think, which gets rid of fucntionality
		# that is useful for speeding up training but not useful for testing.
		with torch.no_grad():
			for i, (walks, labels) in enumerate(test_loader):
				outputs = model(walks)
				loss = loss_function(outputs, labels)
				test_loss_list.append(loss.item())
				if ((i + 1) % 100) == 0:
					print('Epoch [{}/{}], Step [{}/{}], Validation Loss: {:.4f}'
	                  .format(epoch + 1, num_epochs, i + 1, num_steps, loss.item()))
		model.train()

def get_loaders(train_fn, val_fn, test_fn, n):
	csv_train = open(train_fn)
	csv_val = open(val_fn)
	csv_test = open(test_fn)
	csv_reader_train = csv.reader(csv_train, delimiter=',')
	csv_reader_val = csv.reader(csv_val, delimiter=',')
	csv_reader_test = csv.reader(csv_test, delimiter=',')

	train_smiles_array = []
	val_smiles_array = []
	test_smiles_array = []
	train_properties_array = []
	val_properties_array = []
	test_properties_array = []

	for row in csv_reader_train:
		smiles = row[0:1]
		properties = list(map(float, row[1:]))
		train_smiles_array.append(smiles)
		train_properties_array.append(properties)

	for row in csv_reader_val:
		smiles = row[0:1]
		properties = list(map(float, row[1:]))
		val_smiles_array.append(smiles)
		val_properties_array.append(properties)

	for row in csv_reader_test:
		smiles = row[0:1]
		properties = list(map(float, row[1:]))
		test_smiles_array.append(smiles)
		test_properties_array.append(properties)

	train_smiles_array = np.array(train_smiles_array)
	val_smiles_array = np.array(val_smiles_array)
	test_smiles_array = np.array(test_smiles_array)
	train_properties_array = np.array(train_properties_array)
	val_properties_array = np.array(val_properties_array)
	test_properties_array = np.array(test_properties_array)

	train_dataset = WalkDataset(train_smiles_array, train_properties_array, n)
	val_dataset = WalkDataset(val_smiles_array, val_properties_array, n)
	test_dataset = WalkDataset(test_smiles_array, test_properties_array, n)
	train_loader = DataLoader(train_dataset, batch_size=batch_size)
	val_loader = DataLoader(val_dataset, batch_size=val_smiles_array.shape[0])
	test_loader = DataLoader(test_dataset, batch_size=test_smiles_array.shape[0])
	return train_loader, val_loader, test_loader

def main():
	train_fn = sys.argv[1]
	val_fn = sys.argv[2]
	test_fn = sys.argv[3]
	n = int(sys.argv[4])
	train_loader, validation_loader, test_loader = get_loaders(train_fn, val_fn, test_fn, n)
	model = train(train_loader, validation_loader)
	test(model, test_loader)

if __name__ == "__main__":
	main()






