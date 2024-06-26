# RNN-Based-Prediction-of-Time-Series-in-Stock-Market

## Introduction
In this project, I perform a time series prediction using a Recurrent Neural Network regressor. For this example, I will predict Apple's stock price 7 days in advance.

The particular network architecture I will employ for my RNN is a Long Term Short Memory (LTSM), which helps significantly avoid technical problems with optimization of RNNs.

## Data and Code
The data is from this [source](https://www.superdatascience.com/pages/deep-learning). It also has been uploaded to the repo. The code is in the Jupyter notebook format in the repo.

## Some notes
- What is input and output of our model?
In this project, we are dealing with time series data. In the prepossessing stage, we introduced the concept of sliding window with each input vector size = 7 and output a single number. 

- How to understand the parameters of lstm?
Be aware that input size does not have to be equal to hidden size; Hidden size denotes the number of features in the hidden state h. This refers to the number of LSTM units (also called cells or neurons) in a single LSTM layer.

-How lstm works?
Into the LSTM cell, we feed in timestep 1 by itself. The input is a vector with size n_input_features. This is fed through all our fancy gates and stuff and we are left with a hidden state and cell state. Then we go into the next LSTM cell (which is actually just the same “cell”, we’re just looping back to the entrance of it). This time though our input will be the last hidden state, cell state, and now timestep 2. This repeats for each time step. So now we have a hidden state vector for each time step. We started with timesteps x n_input_features and finish with timesteps x n_units. But commonly people will just take the hidden state vector of the final timestep as the output of their LSTM instead of keeping all the hidden states. So that’s how we go from timesteps x n_input_features all the way to a single vector with size n_units.

- Also note that we can stack multiple single lstm layers!

## Summary
- Performed a time series prediction using a LSTM (Long Short - Term Memory) model via PyTorch.
- Implemented data normalization, divided dataset into two consecutive chunk (first full 4/5 and last 1/5) as training, testing set and loaded data into GPU memory.
- Built a RNN (Recurrent Neural Network) model by changing activation and regularization function and set sliding window size to be 7 for predicting.
- Trained RNN model with 300 epoch, presented parameters and loss figure with final loss 0.0011.
