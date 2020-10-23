# Tagenalgo

## Techanical Analysis & Genatic Algorithm
The is a genetic algorithm incorporated with technical analysis indicator to optimize parameters of a strategy. The innovation of the algorithm is  that you can customize your strategy and use the algorithm to optimize the parameters included in the strategy. 

This algorithm provides:

* Optimize parameters of common technical analysis indicators such as **"Relative Strength Indicator"** or **"Simple Moving Average"**.    
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
model = TAGenAlgo(X_train, 3, 100, 0.9, 0, 'single', 'rsi')

# Initialize the model by setting the range of indicators. 
_, init_pop = model.ta_initialize(indicator_set={'rsi': {
    'window': [5, 180],
    'down_thres': [5, 50],
    'up_thres': [51, 90]}})
    
model.fit(init_pop)
```
