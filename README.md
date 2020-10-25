# Tagenalgo

## Techanical Analysis & Genatic Algorithm
The is a genetic algorithm incorporated with technical analysis indicator to optimize parameters of a strategy. The innovation of the algorithm is  that you can customize your strategy and use the algorithm to optimize the parameters included in the strategy. 

This algorithm provides:

* Optimize parameters of common technical analysis indicators such as **"Relative Strength Indicator"** (RSI) or **"Simple Moving Average"** (SMA).    
* Optimize parameters of combined technical analysis indicators. E.g. **"Relative Strength Indicator"** + **"Simple Moving Average"**.
* Optimize parameters of customized strategy. E.g. **"Relative Strength Indicator"** + **"self-defined stop loss mechanism"**.

## Installation
```
pip install tagenalgo
```

## How to use it
As mentioned above, this algorithm is mainly used to optimize parameters of a strategy. In order to fullfill the requirement, you should do:

* Prepare historical data.
* Check if the parameters of the strategy you want to optimize is included in this software. If the software does not include the strategy, you could define the strategy by yourself. (See doc)

Once done the preparation, we can start to train the algorithm!

Import required module.
```python
from tagenalgo import TAGenAlgo
from sklearn.model_selection import train_test_split
```
Train the model.
```python
# Seperate historical data into train set and test set.
X_train, X_test = train_test_split(hist_data, shuffle=False)

# Input the "genetic algo required" parameters and "name of strategy".
model = TAGenAlgo(price=X_train, generations=3, population_size=100, 
                  crossover_prob=0.9, mutation_prob=0, method='single', strategy='rsi')

# Initialize the model by setting the range of indicators. 
_, init_pop = model.ta_initialize(indicator_set={'rsi': {
    'window': [5, 180],
    'down_thres': [5, 50],
    'up_thres': [51, 90]}})
    
model.fit(init_pop)
```

## Performance Improvement (Common RSI v.s. Modified RSI)
RSI strategy usually takes 14 candlesticks as the time window to calculate the indicator value. When the indicator value goes down across the value of 30, it indicates as a buy signal. On the other hand, when it goes up across the value of 70, it represents a sell signal. In this case, the parameters set for RSI strategy is (14, 30, 70). 

By implementing tagenalgo training, we got the modified parameters set of (10, 31, 88). Clearly, the performance of the modified RSI strategy is way better than common RSI strategy as we can see from the chart below.

![comparison](/image/rsi_simple.png)

