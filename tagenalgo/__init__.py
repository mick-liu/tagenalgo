
"""Evolution programming in python with talib and gplearn inspired API
``tagenalgo`` is a parameter optimization algorithm for trading strategy.
"""

from __future__ import absolute_import
from .tagenalgo import TAGenAlgo
from .backtest import rsi_signal, sma_signal, rsinsma_signal, rsi_stoploss_takeprofit
from .utils import position_change, params_mod, strategy_ret, strategy_cum_ret, trade_timing, trading_metric

__version__ = '1.0.7'
__license__ = 'BSD'