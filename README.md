# quantiledeeplearning

Repository for Quantile Deep learning models for multi-step ahead time series prediction. The purpose of this project is to take in past data (univariate/multivariate) and perform multi-step ahead prediction using basic (linear) and various deep learning models (BDLSTM, ConvLSTM, EDLSTM) across multiple datasets (BTC, ETH, SUN, MG, LOR). We will showcase the implementation of classic and quantile versions of such DL models as well as providing evaluation metrics (RMSE, MAE, MAPE), time performance and prediction visualisations.

Note that this project expects the user to have already installed an appropriate python environment and packages to be able to run the libraries mentioned in the files.

## Contents

Our Repository contains the following:

1. main.py

This is the interface of the project. Users only need to run this file to see all model performance. 

The file asks the user its desired dataset (BTC/ETH/SUN/MG/LOR), type of problem if applicable (uni/multi-variate) and runs all the models to that specific problem (e.g. BTC univariate - linear, BDLSTM, ConvLSTM, EDLSTM). Users can customise parameters such as input/output size, train/test split ratio, number of experiments as well as model hyperparameters such as number of hidden layers/neurons. Default values are provided for convenience.

2. models

This directory contains all models used in the project. Each model has uni/multi-variate as well as normal/quantile implementations.

We have implemented the following models:

	1. Linear Regression
	2. Bi-Directional Long Short-Term Memory (BDLSTM)
	3. Convolutional Long Short-Term Memory (ConvLSTM)
	4. Encoder-Decoder Long Short-Term Memory (EDLSTM)

Each model is run to the user specified parameters. Note that normal/quantile models of the same type will follow the same hyperparameters for consistency purposes.

3. visualisations

This directory contains the following:

	1. Code implementations in forms of python notebook (ipynb) for each model
	2. Prediction visualisation for every problem, ran in default values (see results directory)
	3. Visualisations of datasets to familarise the time series (see dataset visualisations directory)

4. data

Directory containing all the datasets used in this project.

	* The Bitcoin and Ethereum datasets were obtained through ...?
	* The Sunspots dataset was retrieved from ...?
	* Mackey-Glass and Lorenz are generative chaotic time series and can be generated through a function in python. More information about Mackey-Glass and Lorenz system and be found below.

	hyperlink 1
	hyperlink 2

5. install_packages.txt

We provide instructions to set up virtual environment for this project so that anyone can run the code on their computer smoothly. 

Here are the instructions for Windows:
	1. Download quantiledeeplearning Repository on Github
	2. Use Powershell (Windows Terminal) and navigate to Project folder
	3. Create virtual environment, type in terminal: 'python -m venv quantiledeeplearning'
	4. Activate the virtual environment: '.\quantiledeeplearning\Scripts\Activate'
		4.1. If encountered error, run 'Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser' and repeat step 4
	5. Run 'pip install -r install_packages.txt'
		5.1. If there is an issue installing statsmodels (such as ERROR: Failed building wheel for statsmodels), install Visual Studio Community 2022. More details: https://github.com/statsmodels/statsmodels/issues/8457
	6. Environment is ready to run main.py. Type 'deactivate' in terminal when finished
To activate the virtual environment in the future, repeat steps 2 & 4.

Here are the instructions for Mac:
	1. Download quantiledeeplearning Repository on Github
	2. Use terminal and navigate to project folder
	3. Create virtual environment: 'python3 -m venv .quantiledeeplearning'
	4. Activate the virtual environment: 'source .quantiledeeplearning/bin/activate'
	5. Install required packages: 'pip install -r install_packages.txt'
	6. Environment is ready to use. Type 'deactivate' in terminal when finished
To access the virual environment in the future, repeat steps 2 & 4.

## Acknowledgement

This project was done as a capstone course in the University of New South Wales (UNSW) School of Mathematics and Statistics Masters Program. It is also a published article (include link when available) in Applied Soft Computing (Elsevier). Our group consists of Jimmy Cheung, Smruthi Rangarajan, Amelia Maddocks and Xizhe Chen. Many thanks to our supervisor Rohitash Chandra for his continuous guidance throughout the project and managing the review process.
