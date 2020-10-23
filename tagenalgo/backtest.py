
import pandas as pd
import numpy as np
import talib
import itertools
from datetime import datetime, timedelta
import numba as nb
from tqdm import tqdm


@nb.njit
def rsi_stoploss_takeprofit(price, rsi, buy_sig, sell_sig, stop_loss, take_profit, error_tol):
    """This is customized strategy function."""
    order_status = 'no_order'
    ta_direc = 'no_direc'
    pos = [0]
    enter_price = None
    for i in range(1, len(price)):
        pre_rsi = rsi[i - 1]
        cur_rsi = rsi[i]

        if order_status == 'order_placed':
            if ta_direc == 'long':
                unrealized_ret = (price[i] / enter_price) - 1
                if (pre_rsi < cur_rsi and (abs(cur_rsi - sell_sig) / sell_sig) < error_tol) or (cur_rsi > sell_sig) \
                        or (unrealized_ret <= stop_loss) or (unrealized_ret >= take_profit):
                    order_status = 'no_order'
                    ta_direc = 'no_direc'
                    enter_price = None
                    pos.append(0)
                else:
                    pos.append(1)
            elif ta_direc == 'short':
                unrealized_ret = 1 - (price[i] / enter_price)
                if (pre_rsi > cur_rsi and (abs(cur_rsi - buy_sig) / buy_sig) < error_tol) or (cur_rsi < buy_sig) \
                        or (unrealized_ret <= stop_loss) or (unrealized_ret >= take_profit):
                    order_status = 'no_order'
                    ta_direc = 'no_direc'
                    enter_price = None
                    pos.append(0)
                else:
                    pos.append(-1)

        elif order_status == 'no_order':
            if (cur_rsi >= sell_sig > pre_rsi) or (pre_rsi > sell_sig and pre_rsi > cur_rsi
                                                   and (abs(cur_rsi - sell_sig) / sell_sig) <= error_tol):
                order_status = 'order_placed'
                ta_direc = 'short'
                enter_price = price[i]
                pos.append(-1)

            elif (cur_rsi <= buy_sig < pre_rsi) or (pre_rsi < buy_sig and pre_rsi < cur_rsi
                                                    and (abs(cur_rsi - buy_sig) / buy_sig) <= error_tol):
                order_status = 'order_placed'
                ta_direc = 'long'
                enter_price = price[i]
                pos.append(1)
            else:
                order_status = 'no_order'
                pos.append(0)

    idc = np.array(pos)
    return idc


@nb.njit
def rsinsma_signal(price, rsi, buy_sig, sell_sig, sma_short, sma_long, error_tol, exit_sig):
    order_status = 'no_order'
    ta_direc = 'no'
    pos = [0]

    for i in range(1, len(price)):
        pre_rsi = rsi[i - 1]
        cur_rsi = rsi[i]

        pre_sma_short = sma_short[i - 1]
        cur_sma_short = sma_short[i]
        pre_sma_long = sma_long[i - 1]
        cur_sma_long = sma_long[i]

        if order_status == 'order_placed':
            if exit_sig == 0:
                if ta_direc == 'long':
                    if (pre_rsi < cur_rsi and (abs(cur_rsi - sell_sig) / sell_sig) < error_tol) or (cur_rsi > sell_sig):
                        order_status = 'no_order'
                        ta_direc = 'no_direc'
                        pos.append(0)
                    else:
                        pos.append(1)
                elif ta_direc == 'short':
                    if (pre_rsi > cur_rsi and (abs(cur_rsi - buy_sig) / buy_sig) < error_tol) or (cur_rsi < buy_sig):
                        order_status = 'no_order'
                        ta_direc = 'no_direc'
                        pos.append(0)
                    else:
                        pos.append(-1)
            elif exit_sig == 1:
                if ta_direc == 'long':
                    if pre_sma_short > pre_sma_long and cur_sma_short < cur_sma_long:
                        order_status = 'no_order'
                        ta_direc = 'no_direc'
                        pos.append(0)
                    else:
                        pos.append(1)

                elif ta_direc == 'short':
                    if pre_sma_short < pre_sma_long and cur_sma_short > cur_sma_long:
                        order_status = 'no_order'
                        ta_direc = 'no_direc'
                        pos.append(0)
                    else:
                        pos.append(-1)

        elif order_status == 'no_order':
            if ((cur_rsi >= sell_sig > pre_rsi) or (pre_rsi > sell_sig and pre_rsi > cur_rsi
                                                    and (abs(cur_rsi - sell_sig) / sell_sig) <= error_tol)) \
                    and ((pre_sma_long < pre_sma_short) and (cur_sma_long > cur_sma_short)):
                ta_direc = 'short'
                order_status = 'order_placed'
                pos.append(-1)

            elif ((cur_rsi <= buy_sig < pre_rsi) or (pre_rsi < buy_sig and pre_rsi < cur_rsi
                                                     and (abs(cur_rsi - buy_sig) / buy_sig) <= error_tol)) \
                    and ((pre_sma_long > pre_sma_short) and (cur_sma_long < cur_sma_short)):
                ta_direc = 'long'
                order_status = 'order_placed'
                pos.append(1)
            else:
                order_status = 'no_order'
                pos.append(0)

    idc = np.array(pos)
    return idc


@nb.njit
def rsi_signal(price, rsi, buy_sig, sell_sig, error_tol):  # modified ver.
    order_status = 'no_order'
    ta_direc = 'no_direc'
    pos = [0]

    for i in range(1, len(price)):
        pre_rsi = rsi[i - 1]
        cur_rsi = rsi[i]

        if order_status == 'order_placed':
            if ta_direc == 'long':
                if (pre_rsi < cur_rsi and (abs(cur_rsi - sell_sig) / sell_sig) < error_tol) or (cur_rsi > sell_sig):
                    order_status = 'no_order'
                    ta_direc = 'no_direc'
                    pos.append(0)
                else:
                    pos.append(1)
            elif ta_direc == 'short':
                if (pre_rsi > cur_rsi and (abs(cur_rsi - buy_sig) / buy_sig) < error_tol) or (cur_rsi < buy_sig):
                    order_status = 'no_order'
                    ta_direc = 'no_direc'
                    pos.append(0)
                else:
                    pos.append(-1)

        elif order_status == 'no_order':
            if (cur_rsi >= sell_sig > pre_rsi) or (pre_rsi > sell_sig and pre_rsi > cur_rsi
                                                   and (abs(cur_rsi - sell_sig) / sell_sig) <= error_tol):
                ta_direc = 'short'
                order_status = 'order_placed'
                pos.append(-1)

            elif (cur_rsi <= buy_sig < pre_rsi) or (pre_rsi < buy_sig and pre_rsi < cur_rsi
                                                    and (abs(cur_rsi - buy_sig) / buy_sig) <= error_tol):
                ta_direc = 'long'
                order_status = 'order_placed'
                pos.append(1)
            else:
                order_status = 'no_order'
                pos.append(0)

    idc = np.array(pos)
    return idc


@nb.njit
def sma_signal(price, sma_short, sma_long):  # modified ver.
    order_status = 'no_order'
    ta_direc = 'no_direc'
    pos = [0]

    for i in range(1, len(price)):
        pre_sma_short = sma_short[i - 1]
        cur_sma_short = sma_short[i]

        pre_sma_long = sma_long[i - 1]
        cur_sma_long = sma_long[i]

        if order_status == 'order_placed':
            if ta_direc == 'long':
                if (pre_sma_long < pre_sma_short) and (cur_sma_long > cur_sma_short):
                    order_status = 'no_order'
                    ta_direc = 'no_direc'
                    pos.append(0)
                else:
                    pos.append(1)

            elif ta_direc == 'short':
                if (pre_sma_long > pre_sma_short) and (cur_sma_long < cur_sma_short):
                    order_status = 'no_order'
                    ta_direc = 'no_direc'
                    pos.append(0)
                else:
                    pos.append(-1)

        elif order_status == 'no_order':
            if (pre_sma_long < pre_sma_short) and (cur_sma_long > cur_sma_short):
                ta_direc = 'short'
                order_status = 'order_placed'
                pos.append(-1)
            elif (pre_sma_long > pre_sma_short) and (cur_sma_long < cur_sma_short):
                ta_direc = 'long'
                order_status = 'order_placed'
                pos.append(1)
            else:
                order_status = 'no_order'
                pos.append(0)

    idc = np.array(pos)
    return idc
