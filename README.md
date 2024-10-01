# Stock-Market-Prediction-Using-LSTM-and-Monte-Carlo-Simultion 
This repository presents two approaches for predicting stock market prices: Long Short-Term Memory (LSTM), a neural network for time-series data, and Monte Carlo Simulation, a probabilistic model to assess future stock price distributions. The project demonstrates how to implement both techniques using historical stock data of Apple Inc. (AAPL) and provides a comprehensive comparison of their performance in predicting future stock prices.

Table of Contents
Introduction
Techniques Overview
LSTM
Monte Carlo Simulation
Key Features
Prerequisites
Installation
Usage
Running the LSTM Model
Running the Monte Carlo Simulation
Model Evaluation
Visualizations
Contributing
License
Introduction
Stock price prediction is a challenging task due to the volatility and randomness in financial markets. This project uses two distinct methodologies to predict future stock prices:

LSTM (Long Short-Term Memory): A deep learning model designed for time-series prediction.
Monte Carlo Simulation: A statistical technique that generates multiple scenarios to assess the probability distribution of possible stock prices.
By combining these techniques, we aim to provide a robust forecasting framework that captures both short-term trends (LSTM) and long-term uncertainties (Monte Carlo).

Techniques Overview
LSTM
LSTM is a type of Recurrent Neural Network (RNN) that excels in learning long-term dependencies in time-series data. It is used here to predict stock prices based on historical data, learning intricate patterns such as trends, seasonality, and noise.

Data Preprocessing: The stock prices are normalized using MinMaxScaler to improve model performance.
Model Architecture: The model is built with multiple LSTM layers followed by a fully connected layer.
Evaluation: The model is evaluated using Root Mean Squared Error (RMSE).
Monte Carlo Simulation
Monte Carlo Simulation is a statistical method that models the probability distribution of future stock prices by generating multiple random samples based on historical stock return data.

Geometric Brownian Motion: Used to simulate possible future stock prices.
Risk Analysis: The simulation provides insights into potential risks and returns.
Scenario Analysis: Multiple simulations are run to explore different stock price scenarios.
Key Features

LSTM Model: Predicts future stock prices based on historical data.
Monte Carlo Simulation: Models future stock price distribution and assesses risk.
Data Preprocessing: MinMax scaling of stock prices.
Evaluation Metrics: Root Mean Squared Error (RMSE) for LSTM predictions.
Visualizations: Plots for stock price trends, simulation outcomes, and volatility analysis.
Prerequisites
Before running this project, ensure you have the following installed:

Python 3.x
Jupyter Notebook
Required Python packages: tensorflow, keras, pandas, numpy, matplotlib, seaborn, sklearn
Installation
Clone this repository to your local machine:

bash
Copy code
git clone https://github.com/yourusername/StockMarketPrediction-LSTM-MonteCarlo.git
cd StockMarketPrediction-LSTM-MonteCarlo
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Running the LSTM Model
Open the Jupyter notebook:
bash
Copy code
jupyter notebook Stock Market Prediction Using LSTM and Monte Carlo Simulation.ipynb
Load the stock data (AAPL.csv) into the notebook.
Run the LSTM model cells to train the model and predict future stock prices.
Visualize the predicted stock prices alongside the actual historical data.
Running the Monte Carlo Simulation
In the same notebook, run the Monte Carlo Simulation cells.
This will generate multiple simulated stock price paths and provide statistical analysis such as expected returns and risk percentiles.
Visualize the simulation results with plots showing possible future stock price distributions.
Model Evaluation
LSTM: The model's performance is evaluated using RMSE, which measures the difference between predicted and actual stock prices.
Monte Carlo Simulation: The results are evaluated through scenario analysis, mean returns, and risk assessment (e.g., 5th percentile risk).
Visualizations



The project includes the following visualizations:
Stock Price Predictions (LSTM): Compares predicted and actual stock prices.
Monte Carlo Simulations: Displays multiple simulation paths for future stock prices.
Risk Analysis: Visualizes the distribution of final stock prices from the Monte Carlo simulations.
Volatility Clustering: Shows rolling volatility trends in the stock returns.
Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to submit a pull request.
Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -am 'Add new feature').
Push to the branch (git push origin feature-branch).
Open a Pull Request.
License
This project is licensed under the MIT License. See the LICENSE file for details.
