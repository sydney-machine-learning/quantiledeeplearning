# quantiledeeplearning

Repository for Quantile Deep learning models for multi-step ahead time series prediction. The purpose of this project is to take in past data (univariate/multivariate) and perform multi-step ahead prediction using basic (linear) and various deep learning models (BDLSTM, ConvLSTM, EDLSTM) across multiple datasets (BTC, ETH, SUN, MG, LOR). We will showcase the implementation of classic and quantile versions of such DL models as well as providing evaluation metrics (RMSE, MAE, MAPE), time performance and prediction visualisations.

## Contents

Our repository contains the following:

1. [main.py](main.py)

This is the interface of the project. Users **only need to run this file** to see all model performance. 

```
E.g. BTC univariate → linear, Qlinear, BDLSTM, QBDLSTM, ConvLSTM, QConvLSTM, EDLSTM, QEDLSTM
```

The file will ask the user for its desired dataset, type of problem (if applicable) and parameter values. It will then run models specific to that problem. When the model changes, the program will ask for hyperparameters for that model. Once inputted, it will continue to run classic + quantile versions of the model before moving on to another model variation. Users can customise parameters such as input/output size, train/test split ratio, number of experiments as well as model hyperparameters such as number of hidden layers/neurons and train epochs. Default values are provided for convenience. See below code chunk for example.

```
Select Dataset: BTC/ETH/SUN/MG/LOR
This Dataset has various features: univariate/multivariate
Select train/test split and number of experiments: 0.7 30
Running models: linear + quantile linear
	(Model & Time Performance)
Moving onto BDLSTM, please enter hidden neurons and num epochs: [64, 32] 100
Running models: BDLSTM, QBDLSTM
	(Model & Time Performance)
Moving onto ConvLSTM, ...
```

2. [models](models/)

This directory contains all models used in the project. We have implemented the following models:

>	1. Linear Regression
>	2. Bi-Directional Long Short-Term Memory (BDLSTM)
>	3. Convolutional Long Short-Term Memory (ConvLSTM)
>	4. Encoder-Decoder Long Short-Term Memory (EDLSTM)

Each model has uni/multi-variate as well as classic/quantile implementations, which makes up 4 different variations. In total, there is $4 * 4 = 16$ files in the directory. Each model is run to the user specified parameters from [main.py](main.py). Note that normal/quantile models of the same type will follow the same hyperparameters for consistency purposes.

3. [visualisations](visualisations/)

This directory contains the following:

* Visualisation code in the form of python notebook (ipynb) for each model: 16 ipynb files total.
* [Prediction](visualisations/results) visualisation for every problem, ran in default values. Categorised in 4 models (linear, bdlstm, convlstm, edlstm), which is then subcategorised in uni/multi-variate.
* Visualisations of [datasets](visualisations/dataset%20visualisations) to familarise the time series problems we are working with.

4. data

Directory containing all the datasets used in this project.

* The Bitcoin and Ethereum datasets were obtained through [Kaggle](https://www.kaggle.com/datasets/kapturovalexander/bitcoin-and-ethereum-prices-from-start-to-2023?select=BTC-USD+%282014-2024%29.csv) by ALEXANDER KAPTUROV. Specifically, the downloaded files were `BTC-USD (2014-2024).csv` and `ETH-USD (2017-2024).csv`.

* The Sunspots dataset was retrieved from [Kaggle](https://www.kaggle.com/datasets/robervalt/sunspots) by ESPECULOIDE.

* Mackey-Glass and Lorenz are generative chaotic time series and can be generated through a function in python. More information about Mackey-Glass and Lorenz system and be found below.

> https://reservoirpy.readthedocs.io/en/latest/api/generated/reservoirpy.datasets.mackey_glass.html
> https://en.wikipedia.org/wiki/Lorenz_system#:~:text=The%20Lorenz%20system%20is%20a,solutions%20of%20the%20Lorenz%20system.

5. [install_packages.txt](install_packages.txt)

We provide instructions to set up virtual environment for this project so that anyone can run the code on their computer smoothly. 

Here are the instructions for Windows:

>	1. Download quantiledeeplearning Repository on Github
>	2. Use Powershell (Windows Terminal) and navigate to Project folder
>	3. Create virtual environment, type in terminal: `python -m venv quantiledeeplearning`
>	4. Activate the virtual environment: `.\quantiledeeplearning\Scripts\Activate`
>		4.1. If encountered error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` and repeat step 4
>	5. Run `pip install -r install_packages.txt`
>		5.1. If there is an issue installing statsmodels (such as ERROR: Failed building wheel for statsmodels), install Visual Studio Community 2022. For more details: https://github.com/statsmodels/statsmodels/issues/8457
>	6. Environment is ready to run [main.py](main.py). Type `deactivate` in terminal when finished
>	To activate the virtual environment in the future, repeat steps 2 & 4.

Here are the instructions for Mac:

>	1. Download quantiledeeplearning Repository on Github
>	2. Use terminal and navigate to project folder
>	3. Create virtual environment: `python3 -m venv .quantiledeeplearning`
>	4. Activate the virtual environment: `source .quantiledeeplearning/bin/activate`
>	5. Install required packages: `pip install -r install_packages.txt`
>	6. Environment is ready to use. Type `deactivate` in terminal when finished
>	To access the virual environment in the future, repeat steps 2 & 4.

## Acknowledgement

This project was done as a capstone course in the University of New South Wales (UNSW) School of Mathematics and Statistics Masters Program. It is also a [published article](https://doi.org/10.1016/j.asoc.2025.114043) in Applied Soft Computing (Elsevier). Our group consists of Jimmy Cheung, Smruthi Rangarajan, Amelia Maddocks and Xizhe Chen. Many thanks to our supervisor Rohitash Chandra for his continuous guidance throughout the project and managing the review process.
