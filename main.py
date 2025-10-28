'''
Author: Jimmy Cheung
Email: jimmy.cheung.professional@gmail.com
Affilations: 
	Transitional Artifical Intelligence Research Group
	Univeristy of New South Wales, School of Mathematics and Statistics

DESCRIPTION
main.py is served as the Ideal User Interface for Quantile Deep Learning. 
Here, the user is able to choose between datasets 
	BTC, ETH (univariate/multivariate), Sunspots, MG, Lorenz
And also define parameters used throughout multiple models, such as
	input/output size, train/test ratio, number of experiments
'''

import subprocess # this library allows this file to run another file
import sys
import json # allows passing list to another file

# HELPER FUNCTIONS
def get_int_input(prompt, default, min_val, max_val):
    while True:
        val = input(f"{prompt} (Default: {default}) ")
        if val.strip() == "":
            return default
        try:
            val = int(val)
            if min_val <= val <= max_val:
                return val
            else:
                print(f"Please enter a number between {min_val} to {max_val}")
        except ValueError:
            print("Invalid input")

def get_float_input(prompt, default, min_val, max_val):
    while True:
        val = input(f"{prompt} (Default: {default}) ")
        if val.strip() == "":
            return default
        try:
            val = float(val)
            if min_val <= val <= max_val:
                return val
            else:
                print(f"Please enter a number between {min_val} to {max_val}")
        except ValueError:
            print("Invalid input")

# for asking number of hidden neurons
def get_hidden_neurons(prompt, default, min_val, max_val, min_size, max_size):
    while True:
        val = input(f"{prompt} (Default: {' '.join(map(str, default))}) ")
        if val.strip() == "":
            return default
        try:
            # Split the input and convert to integers
            neurons = [int(v) for v in val.split()]
            
            # Check number of layers
            if not (min_size <= len(neurons) <= max_size):
                print(f"Please enter between {min_size} and {max_size} values.")
                continue

            # Check each value range
            if all(min_val <= n <= max_val for n in neurons):
                return neurons
            else:
                print(f"Please enter numbers between {min_val} to {max_val}")
        except ValueError:
            print("Invalid input")

dim = 0 # placeholder

# INTERFACE STARTS 
print("\nWelcome to Quantile Multi-Step Ahead Time Series Prediction")
print("	Press '1' to select Bitcoin data")
print("	Press '2' to select Ethereum data")
print("	Press '3' to select Sunspots data")
print("	Press '4' to generate Mackey-Glass data")
print("	Press '5' to generate Lorenz data")
data = get_int_input("\nPlease select a dataset: ", default = 1, min_val = 1, 
	max_val = 5)

if data == 1 or data == 2: # BTC or ETH
	if data == 1:
		print("\nDataset: BITCOIN - This dataset has multiple features:")
	else:
		print("\nDataset: ETHEREUM - This dataset has multiple features:")

	print("	Press '1' to select univariate models")
	print("	Press '2' to select multivariate models")
	type = get_int_input("\nPlease select univariate or multivariate: ", 
		default = 1, min_val = 1, max_val = 2)
elif data == 3: # Sunspots
	print("\nDataset: SUNSPOTS")
elif data == 4: # MG
	print("\nDataset: MACKEY-GLASS")
elif data == 5: # Lorenz
	print("\nDataset: LORENZ - This dataset has multiple dimensions:")
	print("	Press '0' to predict x-values")
	print("	Press '1' to predict y-values")
	print("	Press '2' to predict z-values")
	dim = get_int_input("\nPlease select a dimension: ", default = 0, 
		min_val = 0, max_val = 2)

print("\nDefine Parameters: (Press Enter for Default Values)")
input_size = get_int_input("Please enter input size: ", default = 6, 
	min_val = 1, max_val = 100)
output_size = get_int_input("Please enter output size: ", default = 5, 
	min_val = 1, max_val = 100)
train_ratio = get_float_input("Please enter train ratio: ", default = 0.7, 
	min_val = 0.1, max_val = 0.9)
num_exp = get_int_input("Please enter number of experiments: ", default = 30, 
	min_val = 2, max_val = 100)

if data >= 3 and data <= 5 or type == 1: # Univariate

	subprocess.run([sys.executable, "models/U-Linear.py", str(input_size), 
		str(output_size), str(train_ratio), str(num_exp), str(data), str(dim)])
	subprocess.run([sys.executable, "models/U-QLinear.py", 
		str(input_size), str(output_size), str(train_ratio), str(num_exp), 
		str(data), str(dim)])
	
	print("\nMoving onto BDLSTM:")
	print("Define Model Hyperparameters: (Press Enter for Default Values)")
	hidden_neurons = get_hidden_neurons("Please enter hidden neurons:", 
		default=[64, 32], min_val=1, max_val=1000, min_size=1, max_size=10)
	num_epochs = get_int_input("Please enter number of epochs:", 
		default=100, min_val=5, max_val=1000)

	subprocess.run([sys.executable, "models/U-BDLSTM.py", 
		str(input_size), str(output_size), str(train_ratio), str(num_exp), 
		str(data), str(dim), json.dumps(hidden_neurons), str(num_epochs)])
	subprocess.run([sys.executable, "models/U-QBDLSTM.py", 
		str(input_size), str(output_size), str(train_ratio), str(num_exp), 
		str(data), str(dim), json.dumps(hidden_neurons), str(num_epochs)])
	
	print("\nMoving onto ConvLSTM:")
	print("Define Model Hyperparameters: (Press Enter for Default Values)")
	hidden_neurons = get_hidden_neurons("Please enter hidden neurons:", 
		default=[64, 32], min_val=1, max_val=1000, min_size=1, max_size=10)
	num_epochs = get_int_input("Please enter number of epochs:", 
		default=100, min_val=5, max_val=1000)
	
	subprocess.run([sys.executable, "models/U-ConvLSTM.py", 
		str(input_size), str(output_size), str(train_ratio), str(num_exp), 
		str(data), str(dim), json.dumps(hidden_neurons), str(num_epochs)])
	subprocess.run([sys.executable, "models/U-QConvLSTM.py", 
		str(input_size), str(output_size), str(train_ratio), str(num_exp), 
		str(data), str(dim), json.dumps(hidden_neurons), str(num_epochs)])

	print("\nMoving onto EDLSTM:")
	print("Define Model Hyperparameters: (Press Enter for Default Values)")
	hidden_neurons = get_hidden_neurons("Please enter hidden neurons:", 
		default=[64, 32], min_val=1, max_val=1000, min_size=1, max_size=10)
	num_epochs = get_int_input("Please enter number of epochs:", 
		default=100, min_val=5, max_val=1000)

	subprocess.run([sys.executable, "models/U-EDLSTM.py", 
		str(input_size), str(output_size), str(train_ratio), str(num_exp), 
		str(data), str(dim), json.dumps(hidden_neurons), str(num_epochs)])
	subprocess.run([sys.executable, "models/U-QEDLSTM.py", 
		str(input_size), str(output_size), str(train_ratio), str(num_exp), 
		str(data), str(dim), json.dumps(hidden_neurons), str(num_epochs)])

elif type == 2: # Multivariate - only BTC/ETH

	subprocess.run([sys.executable, "models/M-Linear.py", str(input_size), 
		str(output_size), str(train_ratio), str(num_exp), str(data)])
	subprocess.run([sys.executable, "models/M-QLinear.py", 
		str(input_size), str(output_size), str(train_ratio), str(num_exp), 
		str(data)])
	
	print("\nMoving onto BDLSTM:")
	print("Define Model Hyperparameters: (Press Enter for Default Values)")
	hidden_neurons = get_hidden_neurons("Please enter hidden neurons:", 
		default=[64, 32], min_val=1, max_val=1000, min_size=1, max_size=10)
	num_epochs = get_int_input("Please enter number of epochs:", 
		default=100, min_val=5, max_val=1000)

	subprocess.run([sys.executable, "models/M-BDLSTM.py", str(input_size), 
		str(output_size), str(train_ratio), str(num_exp), str(data), str(dim),
		json.dumps(hidden_neurons), str(num_epochs)])
	subprocess.run([sys.executable, "models/M-QBDLSTM.py", str(input_size), 
		str(output_size), str(train_ratio), str(num_exp), str(data), str(dim),
		json.dumps(hidden_neurons), str(num_epochs)])

	print("\nMoving onto ConvLSTM:")
	print("Define Model Hyperparameters: (Press Enter for Default Values)")
	hidden_neurons = get_hidden_neurons("Please enter hidden neurons:", 
		default=[64, 32], min_val=1, max_val=1000, min_size=1, max_size=10)
	num_epochs = get_int_input("Please enter number of epochs:", 
		default=100, min_val=5, max_val=1000)

	subprocess.run([sys.executable, "models/M-ConvLSTM.py", 
		str(input_size), str(output_size), str(train_ratio), str(num_exp), 
		str(data), str(dim), json.dumps(hidden_neurons), str(num_epochs)])
	subprocess.run([sys.executable, "models/M-QConvLSTM.py", 
		str(input_size), str(output_size), str(train_ratio), str(num_exp), 
		str(data), str(dim), json.dumps(hidden_neurons), str(num_epochs)])
	
	print("\nMoving onto EDLSTM:")
	print("Define Model Hyperparameters: (Press Enter for Default Values)")
	hidden_neurons = get_hidden_neurons("Please enter hidden neurons:", 
		default=[64, 32], min_val=1, max_val=1000, min_size=1, max_size=10)
	num_epochs = get_int_input("Please enter number of epochs:", 
		default=100, min_val=5, max_val=1000)

	subprocess.run([sys.executable, "models/M-EDLSTM.py", 
		str(input_size), str(output_size), str(train_ratio), str(num_exp), 
		str(data), str(dim), json.dumps(hidden_neurons), str(num_epochs)])
	subprocess.run([sys.executable, "models/M-QEDLSTM.py", 
		str(input_size), str(output_size), str(train_ratio), str(num_exp), 
		str(data), str(dim), json.dumps(hidden_neurons), str(num_epochs)])

print("")





