# quantiledeeplearning

Repository for Quantile Deep learning models for multi-step ahead time series prediction. 

This project is done as a capstone course in the University of New South Wales (UNSW) School of Mathematics and Statistics Masters Program. Our group consists of Jimmy Cheung, Smruthi Rangarajan, Amelia Maddocks and Xizhe Chen. Many thanks to our supervisor Rohitash Chandra for his continuous guidance throughout the project.

## Content

This directory contains the following:

1. [Python Notebooks](all_notebooks)

(write description here)

2. [dashboard](dashboard)

(write description here)

3. [data](data/)

Directory containing all the datasets used in this project. 

* The Bitcoin and Ethereum datasets were obtained through the [Github repository](https://github.com/sydney-machine-learning/deeplearning-crypto) by Wu et al.
* The Sunspots dataset was retrieved from [here](https://www.kaggle.com/datasets/robervalt/sunspots/data)
* Mackey-Glass and Lorenz are generative chaotic time series, more information on how to generate them can be found [here](https://reservoirpy.readthedocs.io/en/latest/api/generated/reservoirpy.datasets.mackey_glass.html) for Mackey-Glass and [here](https://en.wikipedia.org/wiki/Lorenz_system#:~:text=The%20Lorenz%20system%20is%20a,solutions%20of%20the%20Lorenz%20system.) for Lorenz system.


4. [images](images/)

This directory contains all kinds of figures from visualising the dataset to plots of multi-step ahead quantile values. They are generated by the python notebooks of this project, there is a directory for each corresponding notebook. 


5. [Old Files](old_files)

Directory which contains previous python notebooks of various models and types of datasets. They are important and should remain in the directory because they were the first models of the project. Please read below for further details.

> a. [Linear Regression](old_files/Linear_Regression)
>
> Multi-step Ahead Linear Regression models on crypto datasets (Bitcoin, Ethereum) as well as further applications such as Sunspots, Mackey-Glass and Lorenz. Within the directory, the crypto notebooks contains univariate linear regression, multivariate linear regression and their respective quantile versions. The further application notebook contains linear regression and quantile linear regression as well as some different way of result visualisations and extreme multi-step ahead predictions.
>
> Data values have been shuffled, normalised and split into training and testing sets (80%/20%). A linear model was fitted and their average Root Mean Squared Error (RMSE) across multi-steps predictions were calculated. Furthermore, the RMSE for each time step were also calculated to demonstrate lower accuracy for further predictions. Other model performance metrics such as Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE) were used after inverse transforming the prediction values. This was done so that readers can be able to interpret price predictions and errors. Both the average MAE and MAPE value across multi-step predictions and their average value in respective time steps were calculated to show that prediction accuracy drops with future time steps. 
>
> The models were repeated 30 times for reliable results. For the cryptocurreny dataset, the input window was fixed at 6 whilst the output window was 5. This allows us to compare our results with our [related work](https://arxiv.org/abs/2405.11431). For the other datasets, the input and output window size were adjusted to demonstrate our model's capability to take in any flexible amount of input and output sizes, as long as it is reasonable for the dataset. For Mackey-Glass and Lorenz, due to the systems being a chaotic time series, it is impossible predict their values and patterns over long periods of time. We can verify that with a case of extreme multi-step ahead chaotic time Series prediction, where we specified the input window of 200 and an output size of 100. 
> 
> b. [Multivariate DLNN](old_files/Multivariate_DLNN)
>
> This directory contains multi-step ahead prediction on multivariate crypto data using best performing deep learning neural networks. Similar to the univariate case, with the difference of input features being more than 1. The best performing models were also different from the univariate case, so new models were developed specifically to handle the multivariate scenario. The models implemented in this file are Encoder-Decoder LSTM (ED-LSTM), bi-directional BD-LSTM and convolutional neural network (CNN). Each model's quantile version were also integrated with the use of tilted loss function. The dimension of the inputs were complex to handle due to the nature of the question at hand.
>
> c. [Univariate DLNN](old_files/Univariate_DLNN)
>
> Multi-step Ahead Prediction on crypto datasets using best performing deep learning neural networks. Further applications such as Sunspots, Mackey-Glass and Lorenz datasets were also tested by fitting a Bi-Directional Long Short Term Memory Model (BD-LSTM) model. This is to showcase our quantile models does not only limit to fitting cryptocurrency datasets. The notebook contains classic deep learning neural network implementation for univariate data regression as well as quantile deep learning models for multi-step ahead time series prediction. The time series are divided into an input window and output window. The goal of the model is to take input data and predict output data. Due to the nature of our project being multi-step ahead, output window should be greater than 1.
>
> Similar to Linear Regression, the data handling involved shuffling, normalising and train test splitting (80%/20%). The best two deep learning models for Bitcoin datasets were BD-LSTM and Convolutional LSTM Model (Conv-LSTM). The best two deep learning models for the Ethereum datasets were Long Short Term Memory (LSTM) model and BD-LSTM. We implemented the corresponding models and their quantile versions and used the same model evaluation metrics. We applied these deep learning models as well as their quantile versions to get prediction RMSE across individual time steps to showcase a decrease in prediction accuracy with future time steps. In addition, we inverse transformed the prediction values to get a visualisation on price prediction and errors. MAE and MAPE for each time steps were also calculated to aid result interpretation. Interesting results can be observed due to the nature of these datasets. In Sunspots dataset, it is not uncommon to have observation to be near 0, which caused the model evaluation metric MAPE to spike up dramatically. Therefore, instead of coming to the conclusion that the models are awful, we should rely on other metrics such as RMSE and MAE. 
>
> d. [Data_Exploration.ipynb](old_files/Data_Exploration.ipynb)
>
> A notebook which focuses on visualising and understanding datasets, and making necessary changes to the dataset in order to fit models. Includes all datasets used in this project which are [Bitcoin](data/coin_Bitcoin.csv), [Ethereum](data/coin_Ethereum.csv), [Sunspots](data/Sunspots.csv) as well as generated chaotic time series Mackey-Glass and Lorenz system.



