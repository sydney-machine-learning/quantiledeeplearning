# quantiledeeplearning
Quantile Deep learning models for multi-step ahead time series prediction.

Please note that this project is in progress and workspace environment may change drastically. Python notebook files will be replaced by polished python files in the near future. Please stay tuned!

## Content

This directory contains the following:

1. [Data_Exploration.ipynb](Data_Exploration.ipynb)

Data Exploration and visualisation of crypto dataset(s). 

2. [Quantile_Regression_PyTorch.ipynb](Quantile_Regression_PyTorch.ipynb)

A workable python notebook involving some dataset (Bitcoin), basic neural network (Linear) and produces a plot along with some quantile, albeit with very low accuracy. Note that the code is very raw and no optimisation has been attempted yet. This is a rework from Week 3's [Simple_Quantile_Reg_NN.py](https://github.com/sydney-machine-learning/quantiledeeplearning/blob/a092b0d9f47d421cad9c6cba221d43eea0d54472/Simple_Quantile_Reg_NN.py).

3. [data](data/)

Directory containing crypto (Bitcoin, Ethereum) data. May have more datasets in the future (Sunspot, Mackey-Glass and Lorenz). The crypto dataset is obtained from the [literature paper](https://arxiv.org/abs/2405.11431) by Wu et al.

4. [images](images/)

Directory which will contain images generated by python files. When converting notebook file to python, images will be saved here instead of displaying as output in notebook.
