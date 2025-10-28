'''
Multivariate Quantile Convolutional Long Short-Term Memory Model

For BTC/ETH - Multivariate
	1. Linear Regression
	2. BDLSTM
	3. ConvLSTM (we are here)
	4. EDLSTM

Feel free to change hyperparameters manually:
	learning rate, batch size, num_filters, kernel_size
'''

print("\nCurrent Model: Quantile ConvLSTM")
print("Importing Libraries...")

import sys
import json
import time
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import statsmodels.stats.api as sms
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
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

# Convolutional LSTM Model for Quantile Regression with Multi-Step Prediction
class QConvLSTM(nn.Module):
    def __init__(self, num_features, hidden_sizes, num_layers, num_quantiles, 
    	num_steps_ahead, num_filters, kernel_size):
        super(QConvLSTM, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.num_quantiles = num_quantiles
        self.num_steps_ahead = num_steps_ahead

        # Convolutional layer
        self.conv1 = nn.Conv1d(in_channels=num_features, 
        	out_channels=num_filters, kernel_size=kernel_size)
        
        self.lstms = nn.ModuleList()
        self.lstms.append(nn.LSTM(num_filters, hidden_sizes[0], 
        	batch_first=True, bidirectional=True))

        for i in range(1, num_layers):
            self.lstms.append(nn.LSTM(hidden_sizes[i-1]*2, hidden_sizes[i], 
            	batch_first=True, bidirectional=True))
    
        self.fc_layers = nn.ModuleList([nn.Linear(hidden_sizes[-1] * 2, 
        	num_quantiles) for _ in range(num_steps_ahead)])

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
 		# Compute the outputs for each step ahead
        step_outputs = [fc(out[:, -1, :]) for fc in self.fc_layers] 
        # Stack the step outputs [batch_size, num_steps_ahead, num_quantiles]
        output = torch.stack(step_outputs, dim=1)  
        return output

# Multivariate data processing
def split_data(data, input_size, output_size, train_ratio, seed):
	# 1. split data into input features (X) and output (y)
	X, y = [], []
	total_size = input_size + output_size
	for i in range(len(data) - total_size + 1):
		X.append(features[i:i+input_size])
		y.append(target[i+input_size:i+total_size])
	# 2. shuffle batches and split into train/test
	X_train, X_test, y_train, y_test = train_test_split(X, y, 
		train_size = train_ratio, random_state = seed)
	return X_train, X_test, y_train, y_test

# Additional function for quantile regression - Quantile loss function 
def quantile_loss(preds, target, quantiles): 
	losses = [] # idea is the same as tilted loss
	for i, quantile in enumerate(quantiles):
		errors = targets[:, :, i] - preds[:, :, i]
		losses.append(torch.mean(torch.max((quantile - 1) * errors, 
			quantile * errors)))
	return torch.mean(torch.stack(losses))

def evaluate_model(model, test_dataloader, quantiles):
	model.eval() 
	y_hat, y_true = [], []

	with torch.no_grad(): # disable gradient calculation
		for x, y in test_dataloader:
			y = y.unsqueeze(-1).expand(-1, -1, len(quantiles)) 
			outputs = model(x) # forward pass
			y_hat.append(outputs)
			y_true.append(y)
	y_hat = torch.cat(y_hat, dim = 0)
	y_true = torch.cat(y_true, dim = 0)
	return y_hat, y_true


# DATA HANDLING
data = int(sys.argv[5]) # BTC/ETH

if data == 1 or data == 2: # BTC/ETH
	if data == 1:
		df = pd.read_csv('data/bitcoin.csv')
	else:
		df = pd.read_csv('data/ethereum.csv')
	df = df.drop(columns = ['Adj Close'])
	features = df[['High', 'Low', 'Open', 'Close', 'Volume']]
	features = MinMaxScaler().fit_transform(features) # normalise input
	target = df['Close']
	target_reshaped = np.array(target).reshape(-1,1) # normalise output
	scaler = MinMaxScaler(feature_range=(0,1)) 
	target = scaler.fit_transform(target_reshaped).flatten()

quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
num_quantiles = len(quantiles) 

input_size = int(sys.argv[1])		# 6
output_size = int(sys.argv[2])		# 5
train_ratio = float(sys.argv[3])	# 0.7
num_exp = int(sys.argv[4])			# 3
seed = 5925

# stores the median rmse & mae per timestep (5) across all exp (30) = 150
all_rmse, all_mae, all_mape = [], [], []
# taking the avg rmse & mae across each exp = (30)
rmse_per_exp, mae_per_exp, mape_per_exp = [], [], []
# special structure: {0.05: [], 0.25: [], 0.5: [], 0.75: [], 0.95: []}
quantile_rmse = {q: [] for q in quantiles} 

print("Import Successful. Models are running...")
start_time = time.time() # begin time

for i in tqdm(range(num_exp)):

	X_train, X_test, y_train, y_test = split_data(df, input_size, 
		output_size, train_ratio, seed)
	train_dataset = Time_Series_Dataset(X_train, y_train)
	test_dataset = Time_Series_Dataset(X_test, y_test)
	# changing batch size affect model accuracy significantly
	train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False) 
	test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

	# Hyperparameters
	num_features = 5  
	hidden_neurons = json.loads(sys.argv[7]) # [64, 32]
	num_layers = len(hidden_neurons)
	num_filters = 64  # Number of filters for Conv1D layer
	kernel_size = 2  # Kernel size for Conv1D layer

	# ConvLSTM-Q
	model = QConvLSTM(num_features, hidden_neurons, num_layers, 
		num_quantiles, output_size, num_filters, kernel_size)
	# Loss and optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

	# Training loop
	num_epochs = int(sys.argv[8]) # default: 100
	for epoch in range(1, num_epochs + 1):
		model.train()
		running_loss = 0.0
		# inputs = X_train, targets = y_train
		for inputs, targets in train_dataloader: 
			
			# [size, num_steps_ahead, num_quantiles]
			targets = targets.unsqueeze(-1).expand(-1, -1, len(quantiles))
			optimizer.zero_grad()
			# Forward pass - [size, num_steps_ahead, num_quantiles]
			outputs = model(inputs)  
			loss = quantile_loss(outputs, targets, quantiles)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()

	pred_vals, act_vals = evaluate_model(model, test_dataloader, quantiles)
	pred_vals = pred_vals.numpy()
	y_test = act_vals.numpy()[:, :, 0]

	results = {q: None for q in quantiles} # dictionary
	for a, q in enumerate(quantiles):
		results[q] = pred_vals[:, :, a]
		quantile_mse = mean_squared_error(results[q], y_test)
		quantile_rmse[q].append(np.sqrt(quantile_mse))

	rmse_curr_step, mae_curr_step, mape_curr_step = [], [], [] 

	predicted_values = scaler.inverse_transform(results[0.5])
	actual_values = scaler.inverse_transform(y_test)

	for step in range(output_size): # for each time step (5)
		# get the rmse & mae of that particular time step
		mse_per_step = mean_squared_error(results[0.5][:, step], y_test[:, step]) 
		rmse_curr_step.append(np.sqrt(mse_per_step))
		mae_curr_step.append(mean_absolute_error(actual_values[:, step], 
			predicted_values[:, step]))
		mape_curr_step.append(mean_absolute_percentage_error(
			actual_values[:, step], predicted_values[:, step]))

	# [[rmse per exp] * num_exp] - e.g. [[5 values] * 30]
	all_rmse.append(rmse_curr_step) 
	all_mae.append(mae_curr_step)
	all_mape.append(mape_curr_step)

	# average rmse & mae each experiment - (30)
	rmse_per_exp.append(np.mean(all_rmse))
	mae_per_exp.append(np.mean(all_mae))
	mape_per_exp.append(np.mean(all_mape))

	seed += 1

# Models finished running, record time
end_time = time.time() 
start_str = time.strftime("%H:%M:%S", time.localtime(start_time)) # e.g. 20:34:31
end_str = time.strftime("%H:%M:%S", time.localtime(end_time)) # e.g. 20:35:41
elapsed = end_time - start_time # Calculate execution time
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)

# 1. Overall Prediction (Median RMSE, MAE)
(low, upp) = sms.DescrStatsW(rmse_per_exp).tconfint_mean()
rmse_interval = np.mean(rmse_per_exp) - low
(low, upp) = sms.DescrStatsW(mae_per_exp).tconfint_mean()
mae_interval = np.mean(mae_per_exp) - low
(low, upp) = sms.DescrStatsW(mape_per_exp).tconfint_mean()
mape_interval = np.mean(mape_per_exp) - low

# 2. Each Time Step Median RMSE, MAE
all_rmse = torch.tensor(all_rmse)
all_mae = torch.tensor(all_mae)
all_mape = torch.tensor(all_mape)

rmse_step, rmse_step_CI = [], [] 
mae_step, mae_step_CI = [], []
mape_step, mape_step_CI = [], []

for step in range(output_size):
	rmse_current_step = all_rmse[:, step].tolist()
	rmse_step.append(np.mean(rmse_current_step))
	(low, upp) = sms.DescrStatsW(rmse_current_step).tconfint_mean()
	# rmse confidence interval for this particular time step
	rmse_step_CI.append(np.mean(rmse_current_step) - low)

	mae_current_step = all_mae[:, step].tolist()
	mae_step.append(np.mean(mae_current_step))
	(low, upp) = sms.DescrStatsW(mae_current_step).tconfint_mean()
	mae_step_CI.append(np.mean(mae_current_step) - low)

	mape_current_step = all_mape[:, step].tolist()
	mape_step.append(np.mean(mape_current_step))
	(low, upp) = sms.DescrStatsW(mape_current_step).tconfint_mean()
	mape_step_CI.append(np.mean(mape_current_step) - low)

# 3. Quantile Summary
q_interval = {q: [] for q in quantiles} # quantile rmse interval
for q in quantiles:
	(low, upp) = sms.DescrStatsW(quantile_rmse[q]).tconfint_mean()
	q_interval[q] = np.mean(quantile_rmse[q]) - low

print(f"\nAfter {num_exp} experimental runs, "
	f"here are the results for Quantile ConvLSTM model:")
print("Results are in (mean ± 95% confidence interval). "
	"Using 0.5 quantile median to predict:")

print(f"Across {output_size} predictive time steps,")
print(f"	RMSE = {np.mean(rmse_per_exp):.4f} ± {rmse_interval:.4f}")
print(f"	MAE = {np.mean(mae_per_exp):.2f} ± {mae_interval:.2f}")
if data == 1 or data == 2:
	print(f"	MAPE = {np.mean(mape_per_exp)*100:.3f}% ± "
		f"{mape_interval*100:.3f}%")

print("\nTaking a closer look at each time step:")
for step in range(output_size):
	if data == 1 or data == 2:
		print(f"At time step {step + 1}, "
			f"RMSE = {rmse_step[step]:.4f} ± {rmse_step_CI[step]:.4f}, "
			f"MAE = {mae_step[step]:.2f} ± {mae_step_CI[step]:.2f}, "
			f"MAPE = {mape_step[step]*100:.3f}% ± {mape_step_CI[step]*100:.3f}%")
	else:
		print(f"At time step {step + 1}, "
			f"RMSE = {rmse_step[step]:.4f} ± {rmse_step_CI[step]:.4f}, "
			f"MAE = {mae_step[step]:.2f} ± {mae_step_CI[step]:.2f}")

print("\nQuantile Performance Summary")
for q in quantiles:
	print(f"	At Quantile {q}, it has RMSE: {np.mean(quantile_rmse[q]):.4f}"
		f" ± {q_interval[q]:.4f}")

print("\nTime Performance")
print(f"	The models began running from {start_str} and ended at {end_str}")
print(f"	Execution time: {minutes} minutes {seconds} seconds")




