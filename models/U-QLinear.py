'''
Now that Linear Regression is completed, we run Quantile Linear Regression
'''

print("\nCurrent Model: Quantile Linear Regression")
print("Importing Libraries, Please Wait...")

import time
import sys
import torch
import pandas as pd
import numpy as np
import statsmodels.stats.api as sms
from tqdm import tqdm # progress bar
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
	mean_absolute_percentage_error)


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

# tilted loss function - crux of quantile regression
def tilted_loss(beta, X, y, tau):
	y_pred = np.dot(X, beta.reshape(X.shape[1], -1)) # multi-step ahead version
	u = y - y_pred
	loss = np.where(u >= 0, tau * u, (tau - 1) * u)
	return np.sum(loss)

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
		df = mackey_glass(n_timesteps = 3000, tau=75, a=0.2, b=0.1, 
			n=10, x0=1.2, h=1.0, seed=5925)
	else:
		dt = 0.01
		num_steps = 3000
		dim = int(sys.argv[6]) # x-dim: 0, y-dim: 1, z-dim: 2
		xyzs = np.empty((num_steps+1, 3))  
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

quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]

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
	X_train, X_test, y_train, y_test = split_series(target_scaled, input_size, 
		output_size, train_ratio, seed)
	X_train, X_test, y_train, y_test = (np.array(X_train), np.array(X_test), 
		np.array(y_train), np.array(y_test))

	results = {q: None for q in quantiles} # dictionary

	for q in quantiles:
		# QUANTILE LINEAR REGRESSION
		initial_beta = np.zeros((X_train.shape[1], output_size)).flatten()
		result = minimize(tilted_loss, initial_beta, 
			args = (X_train, y_train, q), method = 'BFGS')
		beta_hat = result.x.reshape(X_train.shape[1], output_size)
		y_pred_test = np.dot(X_test, beta_hat)
		results[q] = y_pred_test 
		# shape of results is now {0.05: [1021, 5], 0.25: [1021, 5], 
		# 0.5: [1021, 5], 0.75: [1021, 5], 0.95: [1021, 5]}

		# comparing quantile results with y_test values 
		# each have shape [[5 output_size] * 1021 test predictions]
		quantile_mse = mean_squared_error(results[q], y_test) # 1 value
		quantile_rmse[q].append(np.sqrt(quantile_mse)) # e.g. 0.05: 0.003

	rmse_curr_step, mae_curr_step = [], [] # size = output_size
	mape_curr_step = []

	# Inverse transform
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

	# print(f"Experiment {i + 1}/{num_exp} done")
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
	f"here are the results for Quantile Linear Regression Model:")
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
	print(f"	At Quantile {q}, it has RMSE: {np.mean(quantile_rmse[q]):.4f} ± "
		f"{q_interval[q]:.4f}")

print("\nTime Performance")
print(f"	The models began running from {start_str} and ended at {end_str}")
print(f"	Execution time: {minutes} minutes {seconds} seconds")




