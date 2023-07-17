# Bankruptcy-Prediction

This repository contains code for predicting bankruptcy using a machine learning model. The code performs exploratory data analysis (EDA), applies Principal Component Analysis (PCA) for dimensionality reduction, and trains multiple classification models for bankruptcy prediction.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Installation](#installation)

## Introduction
In this project, we aim to predict bankruptcy using machine learning techniques. We start by conducting exploratory data analysis to gain insights into the dataset. We then apply Principal Component Analysis (PCA) to reduce the dimensionality of the features. After that, we train several classification models, including decision tree, random forest, and logistic regression, to predict bankruptcy based on the reduced feature set.

## Dataset
The dataset used for this project is stored in the `data.csv` file. It contains financial metrics and other attributes of companies. The target variable, "Bankrupt?", indicates whether a company has filed for bankruptcy (1) or not (0). The dataset is imbalanced, with a majority of non-bankrupt companies.

## Usage
Clone this repository to your local machine.
Make sure you have the dataset file (data.csv) in the same directory.
Run the code in your preferred Python environment (e.g., Jupyter Notebook, Anaconda).
Follow the code comments and adjust any parameters or settings as needed.
The code will perform EDA, PCA, oversampling, model training, and evaluation.
You can install the required packages by running the following command:

## Results
The code evaluates multiple models, including decision tree, random forest, and logistic regression, for bankruptcy prediction. It measures accuracy and AUC-ROC score as evaluation metrics. The random forest model achieves the highest accuracy and AUC-ROC score among the models tested.

## Installation
To run the code in this repository, please ensure you have the following dependencies installed:
- Python 3
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn


