'''
Now that we have the information of Dataset & uni/multivariate
We break it down to each version - covering all possibilities

Call this file when appropriate. That is, if the user chose

	(BTC/ETH) - Multivariate

The first model to cover is LINEAR REGRESSION
'''

print("\nCurrent Model: Linear Regression")
print("Importing Libraries, Please Wait...")

import time
import sys
import torch
import pandas as pd
import numpy as np
import statsmodels.stats.api as sms
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
	mean_absolute_percentage_error)

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

'''
DATA HANDLING

Due to high correlation between covariates, including all features
is not recommended. Feature selection is needed for model to perform
optimally (future work). Below code demonstrates how to implement
linear model in the case of multivariate data.
'''

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

# sys.argv[0] = script name, sys.argv[1:] = arguments passed
input_size = int(sys.argv[1]) 		# 6
output_size = int(sys.argv[2])		# 5
train_ratio = float(sys.argv[3])	# 0.7
num_exp = int(sys.argv[4])			# 3
seed = 5925

all_rmse_per_timestep = [] # stores rmse values (5) across experiments (30)=(150)
exp_mean_rmse  = [] # stores mean rmse across experiments (30)
all_mae_per_timestep = [] # stores mae values (5) across experiments (30)=(150)
exp_mean_mae = [] # stores mean mae across experiments (30)
all_mape_per_timestep = [] # store mape values (5) across experiments (30)=(150)
exp_mean_mape = [] # store mean mape across experiments (30)

print("Import Successful. Models are running...")
start_time = time.time() # begin time

for i in tqdm(range(num_exp)): 
	X_train, X_test, y_train, y_test = split_data(df, input_size, 
		output_size, train_ratio, seed)
	X_train, X_test, y_train, y_test = (np.array(X_train), 
		np.array(X_test), np.array(y_train), np.array(y_test))
	X_train = X_train.reshape(X_train.shape[0], -1)
	X_test = X_test.reshape(X_test.shape[0], -1)

	'''
	Multivariate Linear Model loses structure of variables 
	due to flattening before being fed to LinearRegression. 
	Deep learning model prevents this:
		- they preserve & learn inter-variable structure
	'''

	# LINEAR REGRESSSION
	model = LinearRegression()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test) # shape = (1021, 5)
	y_pred = torch.tensor(np.array(y_pred))
	y_test = torch.tensor(np.array(y_test))

	# inverse transform (get back to original scale)
	pred_vals = torch.tensor(scaler.inverse_transform(y_pred)) # shape=(1021, 5)
	act_vals = torch.tensor(scaler.inverse_transform(y_test))  # shape=(1021, 5)

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

	seed += 1 # otherwise values (RMSE, MAE) would be the same across exp
	# print(f"Experiment {i + 1}/{num_exp} done")

end_time = time.time() # end time
start_str = time.strftime("%H:%M:%S", time.localtime(start_time)) # e.g. 20:34:31
end_str = time.strftime("%H:%M:%S", time.localtime(end_time)) # e.g. 20:35:41
elapsed = end_time - start_time # Calculate execution time
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)

rmse = np.mean(exp_mean_rmse) # the overall rmse across all experiments (1)
# note that this is also the same as np.mean(all_rmse_per_timestep)
mae = np.mean(exp_mean_mae) # MAE across all experiments (1)
# note that this is also the same as np.mean(all_mae_per_timestep)
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
	f"here are the results for Linear Regression Model:")
print("Results are in (mean ± 95% confidence interval)")

print(f"Across {output_size} predictive time steps,")
print(f"	RMSE = {rmse:.4f} ± {rmse_interval:.4f}")
print(f"	MAE = {mae:.2f} ± {mae_interval:.2f}")
if data == 1 or data == 2:
	print(f"	MAPE = {mape*100:.3f}% ± {mape_interval*100:.3f}%")

print("\nTaking a closer look at each time step:")
for step in range(output_size):
	if data == 1 or data == 2:
		print(f"At time step {step + 1}, RMSE = {avg_rmse_per_timestep[step]:.4f}"
			f" ± {avg_rmse_per_timestep_confint[step]:.4f}, "
			f"MAE = {avg_mae_per_timestep[step]:.2f} ± "
			f"{avg_mae_per_timestep_confint[step]:.2f}, "
			f"MAPE = {avg_mape_per_timestep[step]*100:.3f}% ± "
			f"{avg_mape_per_timestep_confint[step]*100:.3f}%")
	else:
		print(f"At time step {step + 1}, RMSE = {avg_rmse_per_timestep[step]:.4f}"
			f" ± {avg_rmse_per_timestep_confint[step]:.4f}, "
			f"MAE = {avg_mae_per_timestep[step]:.2f} ± "
			f"{avg_mae_per_timestep_confint[step]:.2f}")

print("\nTime Performance")
print(f"	The models began running from {start_str} and ended at {end_str}")
print(f"	Execution time: {minutes} minutes {seconds} seconds")



