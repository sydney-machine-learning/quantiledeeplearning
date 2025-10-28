'''
Now that we have the information of Dataset & uni/multivariate
We break it down to each version - covering all possibilities

Call this file when appropriate. That is, if the user chose

	(BTC/ETH) - Univariate OR (Sunspots/MG/Lorenz)

The next model to cover is Convolutional Long Short-Term Memory model (ConvLSTM)
Feel free to change the following hyperparameters manually:
	learning rate 	(default: 0.0001)
	batch size 		(default: 16)
	num_filters		(default: 64)
	kernel size 	(default: 2)
'''

print("\nCurrent Model: ConvLSTM")
print("Importing Libraries, Please Wait...")

import sys
import time
import json
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import statsmodels.stats.api as sms
from tqdm import tqdm # progress bar
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
	mean_absolute_percentage_error)

# wraps dataset into tensor -> appropriate for pytorch -> DataLoader
class Time_Series_Dataset(Dataset):
	def __init__(self, inputs, outputs):
		self.inputs = inputs
		self.outputs = outputs

	def __len__(self):
		return len(self.inputs)

	def __getitem__(self, idx):
		x = self.inputs[idx]
		y = self.outputs[idx]
		return (torch.tensor(x, dtype = torch.float32), 
			torch.tensor(y, dtype = torch.float32))

class ConvLSTM(nn.Module):
	def __init__(self, num_features, hidden_sizes, num_layers, output_size, 
		num_filters, kernel_size):
		super(ConvLSTM, self).__init__()
		self.hidden_sizes = hidden_sizes
		self.num_layers = num_layers

		# Convolutional layer
		self.conv1 = nn.Conv1d(in_channels = num_features, 
			out_channels = num_filters, kernel_size = kernel_size)
		self.lstms = nn.ModuleList()
		self.lstms.append(nn.LSTM(num_filters, hidden_sizes[0], 
			batch_first=True, bidirectional=True))

		for i in range(1, num_layers):
			self.lstms.append(nn.LSTM(hidden_sizes[i-1]*2, hidden_sizes[i], 
				batch_first=True, bidirectional=True))

		self.fc1 = nn.Linear(hidden_sizes[-1] * 2, 20)  # Fully connected layer
		self.fc2 = nn.Linear(20, output_size)

	def forward(self, x):
		# Convolutional layer expects input of shape 
		x = x.permute(0, 2, 1) # (batch_size, in_channels, seq_length)
		x = torch.relu(self.conv1(x))
		# Convert back to (batch_size, seq_length, num_filters)
		x = x.permute(0, 2, 1)  
		h = x
		for lstm in self.lstms:
			out, _ = lstm(h)
			h = out
		out = torch.relu(self.fc1(out[:, -1, :]))
		out = self.fc2(out)
		return out

# Univariate data processing
def split_series(series, input_size, output_size, train_ratio, seed):
	# 1. split univariate series to input (X) and output(y)
	X, y = [], []
	for i in range(len(series) - input_size - output_size + 1):
		X.append(series[i:i+input_size]) # e.g. [10, 20, 30]
		y.append(series[i+input_size:i+input_size+output_size]) # e.g. [40, 50]
	# 2. shuffle batches and split into train/test
	X_train, X_test, y_train, y_test = train_test_split(X, y, 
		train_size = train_ratio, random_state = seed)
	return X_train, X_test, y_train, y_test

# Lorenz
def lorenz(xyz, *, s=10, r=28, b=2.667):
	x, y, z = xyz
	x_dot = s*(y - x)
	y_dot = r*x - y - x*z
	z_dot = x*y - b*z
	return np.array([x_dot, y_dot, z_dot])

# DATA HANDLING
data = int(sys.argv[5]) # BTC/ETH/Sun/MG/Lorenz

if data == 1 or data == 2: # BTC/ETH
	if data == 1:
		df = pd.read_csv('data/bitcoin.csv')
	else:
		df = pd.read_csv('data/ethereum.csv')
	target = df.iloc[:,4].copy() # only interested in close price
elif data == 3: # Sunspots
	df = pd.read_csv('data/sunspots.csv')
	target = df['Monthly Mean Total Sunspot Number']
elif data == 4 or data == 5: # MG/Lorenz
	if data == 4:
		from reservoirpy.datasets import mackey_glass 
		# Other values of tau can change the chaotic behaviour of the time series
		df = mackey_glass(n_timesteps = 3000, tau = 75, a = 0.2, b = 0.1, 
			n = 10, x0 = 1.2, h = 1.0, seed = 5925)
	else:
		dt = 0.01
		num_steps = 3000
		dim = int(sys.argv[6]) # x-dim: 0, y-dim: 1, z-dim: 2
		xyzs = np.empty((num_steps + 1, 3))  
		xyzs[0] = (0., 1., 1.05)  # initial values
		# Step through "time", calculating the partial derivatives 
		# at the current point and using them to estimate the next point
		for i in range(num_steps):
			xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt
		df = xyzs[:, dim] 
	
	df = df.flatten()
	target = pd.Series(df, name = 'Value')
	target.index = range(len(target))

target_reshaped = np.array(target).reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1)) # normalise response variable
target_scaled = scaler.fit_transform(target_reshaped).flatten()

# sys.argv[0] = script name, sys.argv[1:] = arguments passed
input_size = int(sys.argv[1]) 		# 6
output_size = int(sys.argv[2])		# 5
train_ratio = float(sys.argv[3])	# 0.7
num_exp = int(sys.argv[4])			# 3
seed = 5925

# Hyperparameters
num_features = 1 # the main difference between U & M
hidden_neurons = json.loads(sys.argv[7])
num_layers = len(hidden_neurons)
num_filters = 64  # Number of filters for Conv1D layer
kernel_size = 2  # Kernel size for Conv1D layer

all_rmse_per_timestep = [] # stores rmse values (5) across experiments (30)=(150)
exp_mean_rmse  = [] # stores mean rmse across experiments (30)
all_mae_per_timestep = [] # stores mae values (5) across experiments (30)=(150)
exp_mean_mae = [] # stores mean mae across experiments (30)
all_mape_per_timestep = [] # store mape values (5) across experiments (30)=(150)
exp_mean_mape = [] # store mean mape across experiments (30)

print("Import Successful. Models are running...")
start_time = time.time() # begin time

for i in tqdm(range(num_exp)): 

	# shuffle and sort data into input [10, 20, 30] and output [40, 50]
	X_train, X_test, y_train, y_test = split_series(target_scaled, input_size, 
		output_size, train_ratio, seed)

	# torch.tensor wrap X_train, X_test, y_train, y_test to feed DataLoader
	df_train = Time_Series_Dataset(X_train, y_train)
	df_test = Time_Series_Dataset(X_test, y_test)

	# DataLoader -> model
	# changing batch size affect model accuracy significantly (future work)
	# shuffle = False to maintain sequence structure within input_window
	df_train = DataLoader(df_train, batch_size = 16, shuffle = False)
	df_test = DataLoader(df_test, batch_size = 16, shuffle = False)

	model = ConvLSTM(num_features, hidden_neurons, num_layers, output_size, 
		num_filters, kernel_size)
	criterion = nn.MSELoss() # related work (MSE & ADAM = 0.001)
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001) 

	num_epochs = int(sys.argv[8])
	for epoch in range(num_epochs):
		model.train()
		for inputs, targets in df_train:
			inputs = inputs.unsqueeze(-1)
			outputs = model(inputs) # forward pass
			loss = criterion(outputs, targets)
			optimizer.zero_grad() # backward & optimise
			loss.backward()
			optimizer.step()

	# MODEL TESTING
	model.eval() # evaluate the model on the test set
	y_pred, y_test = [], []

	with torch.no_grad(): # disables gradient computation - no more improving
		for inputs, targets in df_test:
			inputs = inputs.unsqueeze(-1)
			outputs = model(inputs)
			# call cpu for safe conversion
			y_pred.append(outputs.detach().cpu().numpy())
			y_test.append(targets.detach().cpu().numpy())

	y_pred = np.concatenate(y_pred, axis = 0) # convert lists to arrays
	y_test = np.concatenate(y_test, axis = 0)

	# inverse transform (get back to original scale)
	pred_vals = torch.tensor(scaler.inverse_transform(y_pred)) 
	act_vals = torch.tensor(scaler.inverse_transform(y_test)) 

	rmse_per_timestep, mae_per_timestep, mape_per_timestep = [], [], []

	# Get RMSE & MAE per time step
	for step in range(output_size):
		mse_current_step = mean_squared_error(y_pred[:,step], y_test[:,step])
		mae_current_step = mean_absolute_error(act_vals[:,step], 
			pred_vals[:,step])
		mape_current_step = mean_absolute_percentage_error(act_vals[:,step], 
			pred_vals[:,step])

		rmse_per_timestep.append(np.sqrt(mse_current_step))
		mae_per_timestep.append(mae_current_step)
		mape_per_timestep.append(mape_current_step)
		
	all_rmse_per_timestep.append(rmse_per_timestep)
	exp_mean_rmse.append(np.mean(rmse_per_timestep)) 
	all_mae_per_timestep.append(mae_per_timestep)
	exp_mean_mae.append(np.mean(mae_per_timestep))
	all_mape_per_timestep.append(mape_per_timestep)
	exp_mean_mape.append(np.mean(mape_per_timestep))

	seed += 1 # otherwise values (RMSE, MAE, MAPE) would be the same across exp
	# print(f"Experiment {i + 1}/{num_exp} done")

end_time = time.time() # end time
start_str = time.strftime("%H:%M:%S", time.localtime(start_time)) # e.g. 20:34:31
end_str = time.strftime("%H:%M:%S", time.localtime(end_time)) # e.g. 20:35:41
elapsed = end_time - start_time # Calculate execution time
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)

rmse = np.mean(exp_mean_rmse) # the overall rmse across all experiments (1)
mae = np.mean(exp_mean_mae) # MAE across all experiments (1)
mape = np.mean(exp_mean_mape)

(lower_rmse, upper_rmse) = sms.DescrStatsW(exp_mean_rmse).tconfint_mean()
# the interval [rmse - lower_rmse] = [upper_rmse - rmse]
rmse_interval = rmse - lower_rmse 
(lower_mae, upper_mae) = sms.DescrStatsW(exp_mean_mae).tconfint_mean()
mae_interval = mae - lower_mae
(lower_mape, upper_mape) = sms.DescrStatsW(exp_mean_mape).tconfint_mean()
mape_interval = mape - lower_mape

all_rmse_per_timestep = torch.tensor(all_rmse_per_timestep)
all_mae_per_timestep = torch.tensor(all_mae_per_timestep)
all_mape_per_timestep = torch.tensor(all_mape_per_timestep)

avg_rmse_per_timestep  = [] # (size: 5 - per timestep across 3 exp)
avg_rmse_per_timestep_confint = [] # (size: 5, confidence interval value)
avg_mae_per_timestep = []
avg_mae_per_timestep_confint = []
avg_mape_per_timestep = []
avg_mape_per_timestep_confint = []

for step in range(output_size):
	# take rmse values of the particular timestep across each exp
	temp = all_rmse_per_timestep[:, step].tolist() 
	avg_rmse_per_timestep.append(np.mean(temp))
	(low, upp) = sms.DescrStatsW(temp).tconfint_mean()
	avg_rmse_per_timestep_confint.append(np.mean(temp) - low)

	# take mae values of the particular timestep across each exp
	temp = all_mae_per_timestep[:, step].tolist() 
	avg_mae_per_timestep.append(np.mean(temp))
	(low, upp) = sms.DescrStatsW(temp).tconfint_mean()
	avg_mae_per_timestep_confint.append(np.mean(temp) - low)

	temp = all_mape_per_timestep[:, step].tolist()
	avg_mape_per_timestep.append(np.mean(temp))
	(low, upp) = sms.DescrStatsW(temp).tconfint_mean()
	avg_mape_per_timestep_confint.append(np.mean(temp) - low)

print(f"\nAfter {num_exp} experimental runs, "
	f"here are the results for ConvLSTM Model:")
print("Results are in (mean ± 95% confidence interval)")

print(f"Across {output_size} predictive time steps,")
print(f"	RMSE = {rmse:.4f} ± {rmse_interval:.4f}")
print(f"	MAE = {mae:.2f} ± {mae_interval:.2f}")
if data == 1 or data == 2:
	print(f"	MAPE = {mape*100:.3f}% ± {mape_interval*100:.3f}%")

print("\nTaking a closer look at each time step:")
for step in range(output_size):
	if data == 1 or data == 2:
		print(f"At time step {step + 1}, "
			f"RMSE = {avg_rmse_per_timestep[step]:.4f} ± "
			f"{avg_rmse_per_timestep_confint[step]:.4f}, "
			f"MAE = {avg_mae_per_timestep[step]:.2f} ± "
			f"{avg_mae_per_timestep_confint[step]:.2f}, "
			f"MAPE = {avg_mape_per_timestep[step]*100:.3f}% ± "
			f"{avg_mape_per_timestep_confint[step]*100:.3f}%")
	else:
		print(f"At time step {step + 1}, "
			f"RMSE = {avg_rmse_per_timestep[step]:.4f} ± "
			f"{avg_rmse_per_timestep_confint[step]:.4f}, "
			f"MAE = {avg_mae_per_timestep[step]:.2f} ± "
			f"{avg_mae_per_timestep_confint[step]:.2f}")

print("\nTime Performance")
print(f"	The models began running from {start_str} and ended at {end_str}")
print(f"	Execution time: {minutes} minutes {seconds} seconds")




