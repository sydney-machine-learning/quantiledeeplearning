'''
Now that we have the information of Dataset & uni/multivariate
We break it down to each version - covering all possibilities

Call this file when appropriate. That is, if the user chose

	(BTC/ETH) - Univariate OR (Sunspots/MG/Lorenz)

The last model to cover is Encoder-Decoder Long Short-Term Memory model (EDLSTM)
Feel free to change the following hyperparameters manually:
	learning rate 	(default: 0.0001)
	batch size 		(default: 16)
EDLSTM has a suprisingly fast run time
'''

print("\nCurrent Model: EDLSTM")
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
    def __init__(self, inputs, decoder_inputs, outputs):
        self.inputs = inputs
        self.decoder_inputs = decoder_inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        decoder_input = self.decoder_inputs[idx]
        y = self.outputs[idx]
        return (torch.tensor(x, dtype=torch.float32), 
        	torch.tensor(decoder_input, dtype=torch.float32), 
        	torch.tensor(y, dtype=torch.float32))

class Encoder(nn.Module):
    def __init__(self, num_features, hidden_sizes):
        super(Encoder, self).__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        
        self.lstms = nn.ModuleList()
        input_size = num_features
        for hs in hidden_sizes:
            self.lstms.append(nn.LSTM(input_size, hs, batch_first=True))
            input_size = hs

    def forward(self, x):
        hidden_states, cell_states = [], []
        for i, lstm in enumerate(self.lstms):
            h = torch.zeros(1, x.size(0), self.hidden_sizes[i], device=x.device)
            c = torch.zeros(1, x.size(0), self.hidden_sizes[i], device=x.device)
            x, (h, c) = lstm(x, (h, c))
            hidden_states.append(h)
            cell_states.append(c)
        return hidden_states, cell_states

class Decoder(nn.Module):
    def __init__(self, num_features, hidden_sizes):
        super(Decoder, self).__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        
        self.lstms = nn.ModuleList()
        input_size = num_features
        for hs in hidden_sizes:
            self.lstms.append(nn.LSTM(input_size, hs, batch_first=True))
            input_size = hs
        
        self.fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x, hidden_states, cell_states):
        for i, lstm in enumerate(self.lstms):
            h, c = hidden_states[i], cell_states[i]
            x, (h, c) = lstm(x, (h, c))
        out = self.fc(x)
        return out, hidden_states, cell_states

class EDLSTM(nn.Module):
    def __init__(self, num_features, hidden_sizes):
        super(EDLSTM, self).__init__()
        self.encoder = Encoder(num_features, hidden_sizes)
        self.decoder = Decoder(1, hidden_sizes)

    def forward(self, encoder_inputs, decoder_inputs):
        hidden_states, cell_states = self.encoder(encoder_inputs)
        decoder_inputs = decoder_inputs.unsqueeze(-1)
        outputs, hidden_states, cell_states = self.decoder(decoder_inputs, 
        	hidden_states, cell_states)
        return outputs

def split_series(series, input_size, output_size, train_ratio, seed):
    X, y, decoder_inputs = [], [], []
    total_size = input_size + output_size
    for i in range(len(series) - total_size + 1):
        X.append(series[i:i + input_size]) # encoder input
        # target output (the future sequence)
        y.append(series[i + input_size:i + total_size])
        # decoder input (previous target shifted by one step)
        decoder_inputs.append(
            series[i + input_size - 1:i + input_size + output_size - 1])
    # Split into train/test
    (X_train, X_test, y_train, y_test, decoder_train, 
    	decoder_test) = train_test_split(
        X, y, decoder_inputs, train_size=train_ratio, random_state=seed)
    return X_train, X_test, y_train, y_test, decoder_train, decoder_test

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
num_epochs = int(sys.argv[8])

all_rmse_per_timestep = [] # stores rmse values (5) across experiments (30)=(150)
exp_mean_rmse  = [] # stores mean rmse across experiments (30)
all_mae_per_timestep = [] # stores mae values (5) across experiments (30)=(150)
exp_mean_mae = [] # stores mean mae across experiments (30)
all_mape_per_timestep = [] # store mape values (5) across experiments (30)=(150)
exp_mean_mape = [] # store mean mape across experiments (30)

print("Import Successful. Models are running...")
start_time = time.time() # begin time

for i in tqdm(range(num_exp)): 

	X_train, X_test, y_train, y_test, decoder_train, decoder_test = split_series(
		target_scaled, input_size, output_size, train_ratio, seed)
	# torch.tensor wrap X_train, X_test, y_train, y_test to feed DataLoader
	df_train = Time_Series_Dataset(X_train, decoder_train, y_train)
	df_test = Time_Series_Dataset(X_test, decoder_test, y_test)
	# DataLoader -> model
	# changing batch size affect model accuracy significantly (future work)
	# shuffle = False to maintain sequence structure within input_window
	df_train = DataLoader(df_train, batch_size = 16, shuffle = False)
	df_test = DataLoader(df_test, batch_size = 16, shuffle = False)

	model = EDLSTM(num_features, hidden_neurons)
	criterion = nn.MSELoss() # loss & optimiser
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

	for epoch in range(num_epochs):
		model.train()
		for encoder_inputs, decoder_inputs, targets in df_train:
			batch_size = encoder_inputs.shape[0] # Determine batch size 
			# Correctly reshape the inputs
			encoder_inputs = encoder_inputs.view(
				batch_size, input_size, num_features)
			# Output steps dimension only
			decoder_inputs = decoder_inputs.view(batch_size, output_size) 
			# Add feature dimension
			targets = targets.view(batch_size, output_size, 1)  
			outputs = model(encoder_inputs, decoder_inputs) # Forward pass
			# Reshape outputs to match targets
			outputs = outputs.view(batch_size, output_size, 1)
			loss = criterion(outputs, targets)

			# Backward and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	# MODEL TESTING
	model.eval()
	y_pred, y_test = [], []

	with torch.no_grad():
		for encoder_inputs, decoder_inputs, targets in df_test:
			# Determine batch size dynamically
			batch_size = encoder_inputs.shape[0]
			# Correctly reshape the inputs
			encoder_inputs = encoder_inputs.view(
				batch_size, input_size, num_features)
			# Output steps dimension only
			decoder_inputs = decoder_inputs.view(batch_size, output_size)  
			# Add feature dimension
			targets = targets.view(batch_size, output_size, 1)  
			# Forward pass
			outputs = model(encoder_inputs, decoder_inputs)
			# Reshape outputs to match targets
			outputs = outputs.view(batch_size, output_size, 1)

			y_pred.append(outputs.detach().cpu().numpy())
			y_test.append(targets.detach().cpu().numpy())

	y_pred = np.concatenate(y_pred, axis = 0).squeeze(-1) # convert list to numpy
	y_test = np.concatenate(y_test, axis = 0).squeeze(-1)

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
	f"here are the results for EDLSTM Model:")
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




