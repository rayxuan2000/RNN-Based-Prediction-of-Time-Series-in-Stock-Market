# RNN-Based-Prediction-of-Time-Series-in-Stock-Market

## Introduction
In this project, I perform a time series prediction using a Recurrent Neural Network regressor. For this example, I will predict Apple's stock price 7 days in advance.

The particular network architecture I will employ for my RNN is a Long Term Short Memory (LTSM), which helps significantly avoid technical problems with optimization of RNNs.

## Data and Code
The data is from this [source](https://www.superdatascience.com/pages/deep-learning). It also has been uploaded to the repo. The code is in the Jupyter notebook format in the repo.

## Some notes
- What is input and output of our model?


- How to understand the parameters of lstm?

## Summary
- Performed a time series prediction using a LSTM (Long Short - Term Memory) model via PyTorch.
- Implemented data normalization, divided dataset into two consecutive chunk (first full 4/5 and last 1/5) as training, testing set and loaded data into GPU memory.
- Built a RNN (Recurrent Neural Network) model by changing activation and regularization function and set sliding window size to be 7 for predicting.
- Trained RNN model with 300 epoch, presented parameters and loss figure with final loss 0.0011.
