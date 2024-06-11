# Very Simple Quantile Regression Neural Network
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as forward_pass
import matplotlib.pyplot as plt

Bitcoin = pd.read_csv('data/coin_Bitcoin.csv')
data_temp = Bitcoin.iloc[:100, :] # Dataset is simplified to 100 observations

# penalty function for quantile regression
def tilted_loss(quantile, true, pred):
    res = (true-pred)
    return torch.mean(torch.max(quantile*res, (quantile-1)*res), dim=-1)

class MLP(nn.Module): # Simple Multi-Layered Perceptron
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__() # initialisation
        self.fc1 = nn.Linear(input_size, hidden_size) # fc = fully connected, input to hidden
        self.fc2 = nn.Linear(hidden_size, output_size) # hidden to output

    def forward(self, x): # forward pass: input -> hidden -> output
        x = forward_pass.relu(self.fc1(x)) # activation function: ReLU
        x = self.fc2(x) 
        return x

features = data_temp[['High', 'Low', 'Open', 'Volume', 'Marketcap']].values
target = data_temp['Close'].values

X_train = torch.from_numpy(features) # converts NumPy array to PyTorch tensor
y_train = torch.from_numpy(target)

# print(X_train.shape)
# print(y_train.shape)

quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
input_size = X_train.shape[1] # 5 features
hidden_size = 64 # no. of neurons in hidden layer
output_size = 5 # no. of quantiles = no. of output neurons
learning_rate = 0.001
num_epochs = 200

model = MLP(input_size, hidden_size, output_size).double()
optimser = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):

    optimser.zero_grad() # Zero the parameter gradients
    y_pred = model(X_train) # Forward pass

    # Calculate loss
    loss = 0.0
    for i, q in enumerate(quantiles):
        loss += tilted_loss(q, y_train, y_pred[:, i])

    # Backward pass and optimization
    loss.backward()
    optimser.step()

    # Print statistics
    running_loss = loss.item()
    print('[%d] loss: %.3f' % (epoch + 1, running_loss))

print('Finished Training')

colors = ['green', 'orange', 'red', 'blue', 'purple']
plt.scatter(data_temp['Date'], data_temp['Close'], color='black', marker='+', label='target')

for i, q in enumerate(quantiles):
    plt.plot(data_temp['Date'], y_pred.detach().numpy()[:, i], color=colors[i], label=q)

plt.legend()
plt.savefig('images/quantile_regression')

###########################################################################
# Predicting next value
input_data = [105.75, 106.75, 106.75, 0, 1229098150] # High, Low, Open, Volume, Marketcap

# Convert the input data to a PyTorch tensor
input_tensor = torch.tensor([input_data], dtype=torch.double)
model.eval() # Ensure the model is in evaluation mode

# Make the prediction
with torch.no_grad():  # Disable gradient calculation for inference
    prediction = model(input_tensor)

# The prediction is a tensor, you can convert it to a NumPy array or get the value
predicted_value = prediction.numpy()

print('Predicted value: [5th, 25th, 50th, 75th, 95th] percentile')
print(f'Predicted value: {predicted_value}')

