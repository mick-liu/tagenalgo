
import pandas as pd
import numpy as np


def position_change(low, high):
    new_high = np.where(low > high, low, high)
    new_low = np.where(low > high, high, low)
    return new_low, new_high


def params_mod(pop, position):
    pop = np.array(pop)
    ori_low = pop[:, position[0]]
    ori_high = pop[:, position[1]]
    new_low, new_high = position_change(ori_low, ori_high)
    pop[:, position[0]] = new_low
    pop[:, position[1]] = new_high
    return pop


def _cumulative_return(ret):
    cum_ret_list = [ret[0]]
    n = ret.shape[0]
    for i in range(1, n):
        cum_ret = (1 + ret[i]) * (1 + cum_ret_list[-1]) - 1
        cum_ret_list.append(cum_ret)
    cum_ret_list.insert(0, np.nan)
    return cum_ret_list


def strategy_ret(signal, ret):
    strat_ret = np.array(signal[0:-1]) * np.array(ret[1::])
    strat_ret = np.insert(strat_ret, 0, np.nan)
    return strat_ret


def strategy_cum_ret(strat_ret):
    strat_cum_ret = _cumulative_return(strat_ret[1::])
    return strat_cum_ret


def trade_timing(signal):
    n = len(signal)
    enter = []
    exit = []
    for i in range(1, n):
        before_trade = signal[i - 1]
        trade = signal[i]
        if before_trade == 0 and (trade == 1 or trade == -1):
            enter.append(i)
        elif (before_trade == 1 or before_trade == -1 or before_trade == 2 or before_trade == -2) and trade == 0:
            exit.append(i)
    return enter, exit


def trading_metric(ret, enter, exit):
    if len(enter) == len(exit):
        n = len(enter)
    else:
        assert len(enter) == len(exit) + 1
        n = len(exit)
    acc_ret_list = []
    win_pos = []  # used to record the position of the win trade
    for i in range(n):
        sec_ret = ret[enter[i] + 1:exit[i] + 1]
        acc_ret = _cumulative_return(sec_ret)[-1]
        acc_ret_list.append(acc_ret)
        win_pos.append(i)
    try:
        winrate = len([i for i in acc_ret_list if i > 0]) / len(acc_ret_list)
    except ZeroDivisionError as e:
        winrate = 0
        win_pos = None
    return winrate, win_pos
