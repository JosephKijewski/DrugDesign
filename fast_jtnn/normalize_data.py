import numpy as np
import sys
import csv
from sklearn.model_selection import train_test_split

# Documentation
# File takes in the file name of a csv, a percent training data, and a percent splitting data (percent represented as floats in [0, 1]).
# The csv file must have smiles strings in the first columns, and any number of properties in the remaining columns. The program then
# saves train, validation, and test csvs for the data.

def create_files(fn, splits):
	csv_file = open(fn)
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0


	smiles_array = []
	properties_array = []

	for row in csv_reader:
		if line_count == 0:
			print(f'Column names are {", ".join(row)}')
		else:
			smiles = row[0:1]
			properties = list(map(float, row[1:]))
			if (len(row[0:1][0]) == 1):
				print("single atom molecule")
				continue
			smiles_array.append(smiles)
			properties_array.append(properties)
		line_count += 1
		if (line_count % 1000 == 0):
			print(line_count)

	smiles_array = np.array(smiles_array)
	smiles_array = smiles_array.astype(object)
	properties_array = np.array(properties_array)

	num_mols = properties_array.shape[0]
	properties_array_mean = np.expand_dims(np.mean(properties_array, 0), 0)
	properties_array_mean = np.tile(properties_array_mean, (num_mols, 1))
	properties_array_std = np.expand_dims(np.std(properties_array, 0), 0)
	properties_array_std = np.tile(properties_array_std, (num_mols, 1))
	properties_array = (properties_array - properties_array_mean) / properties_array_std
	properties_array = properties_array.astype(object)

	x_train, x_split, y_train, y_split = train_test_split(smiles_array, properties_array, test_size=(1-splits[0]))
	x_val, x_test, y_val, y_test = train_test_split(x_split, y_split, test_size=((1-splits[0]-splits[1]) / (1-splits[0])))
	train = np.hstack((x_train, y_train))
	val = np.hstack((x_val, y_val))
	test = np.hstack((x_test, y_test))
	print(train.shape)
	print(val.shape)
	print(test.shape)

	file_wo_path = fn.split("/")[-1].split(".")[0]
	extension = ".csv"
	train_fn = file_wo_path + "-train" + extension
	val_fn = file_wo_path + "-val" + extension
	test_fn = file_wo_path + "-test" + extension

	np.savetxt(train_fn, train, delimiter=",", fmt='%s')
	np.savetxt(val_fn, val, delimiter=",", fmt='%s')
	np.savetxt(test_fn, test, delimiter=",", fmt='%s')

def main():
	fn = sys.argv[1]
	splits = (float(sys.argv[2]), float(sys.argv[3]))
	assert((splits[0] + splits[1]) <= 1), "Training and validation ratios must add up to 1 or less"
	create_files(fn, splits)


if __name__ == "__main__":
	main()