# quantiledeeplearning

Repository for Quantile Deep learning models for multi-step ahead time series prediction. 

This project is done as a capstone course in the University of New South Wales (UNSW) School of Mathematics and Statistics Masters Program. Our group consists of Jimmy Cheung, Smruthi Rangarajan, Xizhe Chen and Amelia Maddocks. Many thanks to our supervisor Rohitash Chandra for his continuous guidance throughout the project.

## Content

This directory contains the following:

1. [Python_Notebooks](Python_Notebooks)

Directory which contains python notebooks of various models and types of datasets. Please read below for further details.

> a. [Crypto_Linear_Regression.ipynb](Python_Notebooks/Crypto_Linear_Regression.ipynb)
>
> Multi-step Ahead Linear Regression models on crypto datasets. The notebook contains univariate linear regression, multivariate linear regression and their respective quantile versions. 
>
> Data values have been shuffled, normalised and split into training and testing sets (80%/20%). A linear model was fitted and their average Root Mean Squared Error (RMSE) across multi-steps predictions were calculated. Furthermore, the RMSE for each time step were also calculated to demonstrate lower accuracy for further predictions. Other model performance metrics such as Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE) were used after inverse transforming the prediction values. This was done so that readers can be able to interpret price predictions and errors. Both the average MAE and MAPE value across multi-step predictions and their average value in respective time steps were calculated to show that prediction accuracy drops with future time steps. 
>
> Most of the models were repeated 30 times for more reliable results. The multivariate quantile linear regression model were only repeated 10 times due to the more computationally expensive nature of the experiment. These models were applied on Bitcoin and Ethereum datasets.
> 
> b. [Crypto_Multivariate_DLNN.ipynb](Python_Notebooks/Crypto_Multivariate_DLNN.ipynb)
>
> Multi-step ahead prediction on multivariate crypto data using best performing deep learning neural networks. Similar to the univariate case, with the difference of input features being more than 1. The best performing models were also different from the univariate case, so new models were developed specifically to handle the multivariate scenario. The models implemented in this file are Encoder-Decoder LSTM (ED-LSTM), BD-LSTM and convolutional neural network (CNN). Each model's quantile version were also integrated with the use of tilted loss function. The dimension of the inputs were complex to handle due to the nature of the question at hand.
>
> c. [Crypto_Univariate_DLNN.ipynb](Python_Notebooks/Crypto_Univariate_DLNN.ipynb)
>
> Multi-step Ahead Prediction on crypto datasets using [best performing](https://arxiv.org/abs/2405.11431) neural networks. The notebook contains classic neural network implementation for univariate data regression as well as quantile deep learning models for multi-step ahead time series prediction. The time series are divided into an input window and output window. The goal of the model is to take input data and predict output data. Due to the nature of our project being multi-step ahead, output window should be greater than 1.
>
> Similar to Linear Regression, the data handling involved shuffling, normalising and train test splitting (80%/20%). The best two deep learning models for Bitcoin datasets were Bi-Directional Long Short Term Memory Model (BD-LSTM) and Convolutional Long Short Term Memory Model (Conv-LSTM). We applied those two deep learning models as well as their quantile versions to get prediction RMSE across individual time steps to showcase a decrease in prediction accuracy with future time steps. In addition, we inverse transformed the prediction values to get a visualisation on price prediction and errors. MAE and MAPE for each time steps were also calculated to aid result interpretation. The best two deep learning models for the Ethereum datasets were Long Short Term Memory (LSTM) model and BD-LSTM. We implemented the corresponding models and their quantile versions and used the same model evaluation metrics.
>
> d. [Data_Exploration.ipynb](Python_Notebooks/Data_Exploration.ipynb)
>
> A notebook which focuses on visualising and understanding datasets, and making necessary changes to the dataset in order to fit models. Includes all datasets used in this project which are [Bitcoin](data/coin_Bitcoin.csv), [Ethereum](data/coin_Ethereum.csv), [Sunspots](data/Sunspots.csv) as well as generated chaotic time series Mackey-Glass and Lorenz system.
>
> e. [Linear_Regression.ipynb](Python_Notebooks/Linear_Regression.ipynb)
>
> Multi-step Ahead Linear Regression models for other datasets such as Sunspots, Mackey-Glass and Lorenz. This is to showcase our quantile models does not only limit to fitting cryptocurrency datasets. The data at hand are all of univariate time series nature. This notebook contains linear regression and their respective quantile versions. Interesting results can be observed due to the nature of these datasets. In Sunspots dataset, it is not uncommon to have observation to be near 0, which caused the model evaluation metric MAPE to spike up dramatically. Therefore, instead of coming to the conclusion that the models are awful, we should rely on other metrics such as RMSE and MAE. For Mackey-Glass and Lorenz, due to the systems being a chaotic time series, it is impossible predict their values and patterns over long periods of time. We can verify that with a case of extreme multi-step ahead chaotic time Series prediction, where we specified the input window of 200 and an output size of 100. 
>
> f. [Univariate_DLNN.ipynb](Python_Notebooks/Univariate_DLNN.ipynb)
>
> Multi-Step Ahead Prediction on Sunspots, Mackey-Glass and Lorenz datasets using BD-LSTM. The quantile version were also implemented. Compared to the Crypto Regression Notebooks (a - c) which has an input and output window size of 5 and 3, our multi-step ahead prediction models (e, f) here takes in 10 data points and outputs 6 future time steps. The reason for this modification is to demonstrate our model's capability to take in any flexible amount of input and output sizes, as long as it is reasonable for the dataset.


2. [data](data/)

Directory containing all the datasets used in this project. 

* The Bitcoin and Ethereum datasets were obtained through the [Github repository](https://github.com/sydney-machine-learning/deeplearning-crypto) by Wu et al.
* The Sunspots dataset was retrieved from [here](https://www.kaggle.com/datasets/robervalt/sunspots/data)
* Mackey-Glass and Lorenz are generative chaotic time series, more information on how to generate them can be found [here](https://reservoirpy.readthedocs.io/en/latest/api/generated/reservoirpy.datasets.mackey_glass.html) for Mackey-Glass and [here](https://en.wikipedia.org/wiki/Lorenz_system#:~:text=The%20Lorenz%20system%20is%20a,solutions%20of%20the%20Lorenz%20system.) for Lorenz system.


3. [images](images/)

Images generated by the above python notebooks. They are categorised by datasets, within it contains all kinds of figures from visualising the dataset to predicting multi-step ahead quantile values.


