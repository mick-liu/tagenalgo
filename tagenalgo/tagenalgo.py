
# Author: Mick Liu

import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import random
import talib
from joblib import Parallel, delayed
import itertools
from .backtest import rsi_signal, sma_signal, rsinsma_signal, rsi_stoploss_takeprofit
from .utils import position_change, params_mod, strategy_ret, strategy_cum_ret, trade_timing, \
    trading_metric


class TAGenAlgo(object):

    """A parameter optimization algorithm for trading strategy.
    
    The is a genetic algorithm incorporated with technical analysis indicator 
    to optimize parameters of a strategy. The innovation of the algorithm is 
    that you can customize your strategy and use the algorithm to optimize the 
    parameters included in the strategy. In general, this algorithm provides 
    the following three main functions.
    
    1. Optimize parameters of common technical analysis indicators such as 
        "Relative Strength Indicator" or "Simple Moving Average".    
    2. Optimize parameters of combined technical analysis indicators. E.g. 
        "Relative Strength Indicator" + "Simple Moving Average".
    3. Optimize parameters of customized strategy. E.g. 
        "Relative Strength Indicator" + "self-defined stop loss mechanism".
    
    Parameters
    ----------
    price : np.array
        Train vectors X.

    generations : int
        The number of generation for evolution.
        
    population_size : int
        The number of individual in a population.

    crossover_prob : float
        The probability of executing crossover function.

    mutation_prob : float
        The probability of executing mutation function.
        
    method : string (default='single')
        An identification of a strategy type. E.g. if a strategy only include
        one indicator, then we define it as "single". On the other hand, if 
        a strategy contains more than one indicator, then we define it as
        "multiple".

    strategy : string
        The name of a technical analysis indicator or a customized strategy.
    
    References
    ----------
    .. [1] Fayek, M. B. et al. "Multi-objective Optimization of Technical 
            Stock Market Indicators using GAs", 2013.

    .. [2] Simões, et al. "An Innovative GA Optimized Investment Strategy 
            based on a New Technical Indicator using Multiple MAS", 2010

    .. [3] J. Koza, "Genetic Programming", 1992.

    .. [4] R. Poli, et al. "A Field Guide to Genetic Programming", 2008.

    """

    def __init__(self,
                 price=None,
                 generations=None,
                 population_size=None,
                 crossover_prob=None,
                 mutation_prob=None,
                 method='single',
                 strategy=None):
        self.init = None
        self.price = price
        self.asset_ret = np.array(DataFrame(self.price, columns=['return'])['return'].pct_change())
        self.generations = generations
        self.pop_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.method = method
        self.strategy = strategy
        self.idc = None
        self.idx = np.array(range(len(price)))
        self.best_params = None

    def ta_initialize(self, indicator_set):
        """Initialize the model and generate training population.

        Parameters
        ----------
        indicator_set : dict
            Information about the evolution.

        For example
        -----------
        >>> In: _ta_initialization(indicator_set={'rsi':{'window:[1,20], 'low':[3, 50], 'high':[51,99]}}
        >>> Out: {'indicators': ['rsi'],
                  'paramaters': ['rsi_window', 'rsi_low', 'rsi_high'],
                  'initial population': [[3, 3, 71], [2, 10, 63], [13, 29, 61]]}

        Warning
        -------
        The order of the 'indicator_set' needs to obey the following rules.
        1. 'window -> 'low threshold' -> 'high threshold' (E.g. RSI indicator)
        2. 'short term window' -> 'long term window' (E.g. SMA indicator)
        3. 'window' -> 'low threshold' -> 'high threshold' -> 'short term window' -> 'long term window'
            (E.g. Customized indicator)

        """
        init = dict()
        pop = []
        for k in range(self.pop_size):
            pop_val = []
            for i in indicator_set:
                for j in indicator_set[i]:
                    param_rng = random.randint(indicator_set[i][j][0], indicator_set[i][j][1])
                    pop_val.append(param_rng)
            pop.append(pop_val)

        init['indicators'] = list(indicator_set.keys())
        if self.method == 'single':
            init['parameters'] = [i + '_' + j for i in indicator_set for j in indicator_set[i]]
            init['initial_population'] = pop
            self.idc = init['indicators']
        elif self.method == 'multiple':
            init['parameters'] = [i + '_' + j for i in indicator_set for j in indicator_set[i]] + ['exit_sig']
            pop = [(i + [random.randint(0, len(init['indicators']))]) for i in pop]
            init['initial_population'] = pop
            self.idc = init['indicators']
        self.init = init
        return init, pop

    def _verbose_reporter(self, run_details=None):
        """A report of the progress of the evolution process.

        Parameters
        ----------
        run_details : dict
            Information about the evolution.

        """
        dash_len = len(self.init['parameters'][0])
        for i in range(len(self.init['parameters'])):
            dash_len = dash_len + len(self.init['parameters'][i])

        if run_details is None:
            print('    |{:^29}|'.format('Individual Performance') +
                  ('{:^' + '{}'.format(dash_len) + '}|').format('Parameters'))
            print('-' * 4 + ' ' + '-' * 29 + ' ' + '-' * dash_len + ' ' + '-' * 4)

            line_format = '{:>4} {:>18} {:>10}'
            fix_str = line_format.format('Gen', 'Expected Return', 'Win%')
            for j in range(len(self.init['parameters'])):
                fix_str += ' {:>' + '{}'.format(len(self.init['parameters'][j])) + '} '
                fix_str = fix_str.format(self.init['parameters'][j])
            print(fix_str)
        else:
            line_format2 = '{:>4d} {:>18.4f} {:>10.2f}'
            fix_str2 = line_format2.format(run_details['generation'],
                                           run_details['expected_return'],
                                           run_details['win_rate'])
            for j in range(len(self.init['parameters'])):
                fix_str2 += ' {:>' + '{}'.format(len(self.init['parameters'][j])) + '} '
                fix_str2 = fix_str2.format(run_details['parameters'][j])
            print(fix_str2)

    def _fitness_cal(self, params, sig_tol=0.03):
        fitness_val_cumret = None
        fitness_val_winrate = None
        if self.method == 'single':
            if self.strategy == 'rsi_stoploss_takeprofit':
                window = params[0]
                buy_sig = params[1]
                sell_sig = params[2]
                sl = params[3] * 0.001 * (-1)
                tp = params[4] * 0.001

                rsi = talib.RSI(self.price, timeperiod=window)
                idc = rsi_stoploss_takeprofit(self.price, rsi=rsi, buy_sig=buy_sig,
                                              sell_sig=sell_sig, stop_loss=sl,
                                              take_profit=tp, error_tol=sig_tol)
                enter_pos, exit_pos = trade_timing(idc)
                strat_ret = strategy_ret(idc, self.asset_ret)
                fitness_val_cumret = strategy_cum_ret(strat_ret)[-1]
                fitness_val_winrate, win_pos = trading_metric(strat_ret, enter_pos, exit_pos)

            elif self.strategy == 'rsi':
                window = params[0]
                buy_sig = params[1]
                sell_sig = params[2]

                rsi = talib.RSI(self.price, timeperiod=window)
                idc = rsi_signal(self.price, rsi=rsi, buy_sig=buy_sig, sell_sig=sell_sig, error_tol=sig_tol)
                enter_pos, exit_pos = trade_timing(idc)
                strat_ret = strategy_ret(idc, self.asset_ret)
                fitness_val_cumret = strategy_cum_ret(strat_ret)[-1]
                fitness_val_winrate, win_pos = trading_metric(strat_ret, enter_pos, exit_pos)

            elif self.strategy == 'sma':
                short_window = params[0]
                long_window = params[1]

                sma_short = talib.MA(self.price, timeperiod=short_window)
                sma_long = talib.MA(self.price, timeperiod=long_window)
                idc = sma_signal(self.price, sma_short=sma_short, sma_long=sma_long)
                enter_pos, exit_pos = trade_timing(idc)
                strat_ret = strategy_ret(idc, self.asset_ret)
                fitness_val_cumret = strategy_cum_ret(strat_ret)[-1]
                fitness_val_winrate, win_pos = trading_metric(strat_ret, enter_pos, exit_pos)

        elif self.method == 'multiple':
            if self.strategy == 'RSInSMA':
                rsi_window = params[0]
                buy_sig = params[1]
                sell_sig = params[2]
                short_window = params[3]
                long_window = params[4]
                exit_sig = params[5]  # param (indicator) that we use to decide when to close our position

                rsi = talib.RSI(self.price, timeperiod=rsi_window)
                sma_short = talib.MA(self.price, timeperiod=short_window)
                sma_long = talib.MA(self.price, timeperiod=long_window)
                idc = rsinsma_signal(self.price, rsi=rsi, buy_sig=buy_sig, sell_sig=sell_sig,
                                     sma_short=sma_short, sma_long=sma_long, error_tol=sig_tol, exit_sig=exit_sig)
                enter_pos, exit_pos = trade_timing(idc)
                strat_ret = strategy_ret(idc, self.asset_ret)
                fitness_val_cumret = strategy_cum_ret(strat_ret)[-1]
                fitness_val_winrate, win_pos = trading_metric(strat_ret, enter_pos, exit_pos)

        return fitness_val_cumret, fitness_val_winrate

    def _fitness_evaluation(self, pop):
        result = {}
        fit_vals_lst = []
        winrate = []
        win_pat = []
        params = []
        for ind in pop:
            fit_vals_lst.append(self._fitness_cal(ind)[0])
            winrate.append(self._fitness_cal(ind)[1])
            win_pat.append(self._fitness_cal(ind)[2])
            params.append(ind)

        # to ensure the denominator is not equal to zero
        if max(fit_vals_lst) - min(fit_vals_lst) != 0:
            norm_fit = [((i - min(fit_vals_lst)) / (max(fit_vals_lst) - min(fit_vals_lst))) for i in fit_vals_lst]
            norm_fit_wgh = [np.round(i / sum(norm_fit), 3) for i in norm_fit]
        else:
            norm_fit_wgh = [0 for i in range(len(fit_vals_lst))]
        result['fit_vals'] = fit_vals_lst
        result['win_rate'] = winrate
        result['win_patterns'] = win_pat
        result['fit_weight'] = norm_fit_wgh
        result['params'] = np.array(params)
        return result

    def _crossover(self, parentA, parentB):
        gene_len_a = len(parentA)
        child = np.array([np.nan for i in range(gene_len_a)])
        if self.method == 'single':
            if self.idc[0] == 'rsi' or self.idc[0] == 'sma':
                gene_diversity = list(itertools.combinations(np.array(range(gene_len_a)),
                                                             random.randint(1, gene_len_a)))
                gene_select = list(gene_diversity[random.randint(0, len(gene_diversity))])
                inheritedGeneA = parentA[gene_select]
                child[gene_select] = inheritedGeneA

                crossover_pos = [i for i in range(gene_len_a) if np.isnan(child[i])]
                child[crossover_pos] = parentB[crossover_pos]
        elif self.method == 'multiple':
            gene_diversity = list(itertools.combinations(np.array(range(gene_len_a)),
                                                         random.randint(1, gene_len_a)))
            gene_select = list(gene_diversity[random.randint(0, len(gene_diversity))])
            inheritedGeneA = parentA[gene_select]
            child[gene_select] = inheritedGeneA

            crossover_pos = [i for i in range(gene_len_a) if np.isnan(child[i])]
            child[crossover_pos] = parentB[crossover_pos]

        return child

    def fit(self, pop, if_params_mod=None):
        """Fit to the data.

        Parameters
        ----------
        pop : list
            Generated population.

        if_params_mod : dict
            Used to adjust the position of the parameter, when you expect the lower value might exceeds
            higher value after crossover process.


        For example
        -----------
        >>> In: model.fit(init_pop, if_params_mod={'position': [3, 4]})

        """
        self._verbose_reporter()

        if if_params_mod is not None:
            pop = params_mod(pop, position=if_params_mod['position'])

        pop_fit = self._fitness_evaluation(pop)
        best_fit_global = np.max(pop_fit['fit_vals'])
        best_fit_idx = pop_fit['fit_vals'].index(best_fit_global)
        best_params_global = pop_fit['params'][best_fit_idx]
        best_winrate_global = pop_fit['win_rate'][best_fit_idx]

        run_result = {'generation': 0,
                      'expected_return': best_fit_global,
                      'win_rate': best_winrate_global,
                      'parameters': best_params_global}
        self._verbose_reporter(run_result)

        # Iterate over all generations
        for g in range(1, self.generations):
            new_pop = Parallel(n_jobs=2)(delayed(self._tournament_evolve)(pop_fit)
                                         for i in range(self.pop_size))
            if if_params_mod is not None:
                new_pop = params_mod(new_pop, position=if_params_mod['position'])

            pop_fit = self._fitness_evaluation(new_pop)
            best_fit = np.max(pop_fit['fit_vals'])
            best_fit_idx = pop_fit['fit_vals'].index(best_fit)
            best_params = pop_fit['params'][best_fit_idx]
            best_winrate = pop_fit['win_rate'][best_fit_idx]

            if (best_fit >= best_fit_global) and (best_winrate >= 0.75):
                best_fit_global = best_fit
                best_params_global = best_params
                best_winrate_global = best_winrate
                run_result = {'generation': g,
                              'expected_return': best_fit_global,
                              'win_rate': best_winrate_global,
                              'parameters': best_params_global}
                self._verbose_reporter(run_result)

            elif (best_fit > best_fit_global) and (best_winrate >= best_winrate_global):
                best_fit_global = best_fit
                best_params_global = best_params
                best_winrate_global = best_winrate
                run_result = {'generation': g,
                              'expected_return': best_fit_global,
                              'win_rate': best_winrate_global,
                              'parameters': best_params_global}
                self._verbose_reporter(run_result)

            else:
                run_result = {'generation': g,
                              'expected_return': best_fit_global,
                              'win_rate': best_winrate_global,
                              'parameters': best_params_global}
                self._verbose_reporter(run_result)

        self.best_params = best_params_global
        return pop

    def _tournament_evolve(self, parents_fit_evals):
        def _selection(fit_evals, tournament_size=3):
            contenders = random.randint(0, self.pop_size, [2, tournament_size])
            selected_parent_lst = []
            for i in range(len(contenders)):
                contenders_fit = [fit_evals['fit_weight'][i] for i in contenders[i]]
                contenders_params = [fit_evals['params'][i] for i in contenders[i]]
                contenders_index = int(np.argmax(contenders_fit))
                selected_parent = contenders_params[contenders_index]
                selected_parent_lst.append(selected_parent)
            selected_parentA = selected_parent_lst[0]
            selected_parentB = selected_parent_lst[1]
            return selected_parentA, selected_parentB

        parentA, parentB = _selection(parents_fit_evals)
        new_chromosome = parentA
        if random.random() <= self.crossover_prob:
            new_chromosome = self._crossover(parentA, parentB)
        new_chromosome = list(new_chromosome)
        return new_chromosome

    def predict(self, test_set, error_tol=0.03):
        """Predict future return and win rate on test vectors X.

        Parameters
        ----------
        test_set : np.array
            Test vectors X.

        error_tol : float
            This parameter is an acceptable error when calculating trading signals, if the number is high, the
            trading frequency will be high as well.


        For example
        -----------
        >>> In: from sklearn.model_selection import train_test_split
        >>> In: X_train, X_test = train_test_split(asset_price, shuffle=False)
        >>> In: pred_ret, pred_winrate = model.predict(X_test, error_tol=0.03)

        """
        test_ret = DataFrame(test_set, columns=['return'])['return'].pct_change().values
        params = self.best_params
        order_status = 'no_order'
        ta_direc = 'no'
        pos = [0] * 30
        
        n = len(test_set)
        for i in range(1, n):
            if order_status == 'no_order':
                if self.method == 'single':
                    if self.strategy == 'rsi':
                        window = params[0]
                        buy_sig = params[1]
                        sell_sig = params[2]

                        rsi = talib.RSI(test_set, timeperiod=window)
                        pre_rsi = rsi[i - 1]
                        cur_rsi = rsi[i]

                        if (cur_rsi >= sell_sig > pre_rsi) or (pre_rsi > sell_sig and pre_rsi > cur_rsi
                                                               and (abs(
                                    cur_rsi - sell_sig) / sell_sig) <= error_tol):
                            ta_direc = 'short'
                            order_status = 'order_placed'
                            pos.append(-1)

                        elif (cur_rsi <= buy_sig < pre_rsi) or (pre_rsi < buy_sig and pre_rsi < cur_rsi
                                                                and (abs(cur_rsi - buy_sig)
                                                                     / buy_sig) <= error_tol):
                            ta_direc = 'long'
                            order_status = 'order_placed'
                            pos.append(1)
                        else:
                            order_status = 'no_order'
                            pos.append(0)

                    elif self.strategy == 'sma':
                        short_window = params[0]
                        long_window = params[1]
                        sma_short = talib.MA(self.price, timeperiod=short_window)
                        sma_long = talib.MA(self.price, timeperiod=long_window)

                        pre_sma_short = sma_short[i - 1]
                        cur_sma_short = sma_short[i]
                        pre_sma_long = sma_long[i - 1]
                        cur_sma_long = sma_long[i]

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

                elif self.method == "multiple":
                    if self.strategy == 'RSInSMA':
                        window = params[0]
                        buy_sig = params[1]
                        sell_sig = params[2]
                        short_window = params[3]
                        long_window = params[4]
                        rsi = talib.RSI(test_set, timeperiod=window)
                        sma_short = talib.MA(self.price, timeperiod=short_window)
                        sma_long = talib.MA(self.price, timeperiod=long_window)

                        pre_rsi = rsi[i - 1]
                        cur_rsi = rsi[i]

                        pre_sma_short = sma_short[i - 1]
                        cur_sma_short = sma_short[i]
                        pre_sma_long = sma_long[i - 1]
                        cur_sma_long = sma_long[i]

                        if ((cur_rsi >= sell_sig > pre_rsi) or (pre_rsi > sell_sig and pre_rsi > cur_rsi
                                                                and (abs(
                                    cur_rsi - sell_sig) / sell_sig) <= error_tol)) \
                                and ((pre_sma_long < pre_sma_short) and (cur_sma_long > cur_sma_short)):
                            ta_direc = 'short'
                            order_status = 'order_placed'
                            pos.append(-1)

                        elif ((cur_rsi <= buy_sig < pre_rsi) or (pre_rsi < buy_sig and pre_rsi < cur_rsi
                                                                 and (abs(
                                    cur_rsi - buy_sig) / buy_sig) <= error_tol)) \
                                and ((pre_sma_long > pre_sma_short) and (cur_sma_long < cur_sma_short)):
                            ta_direc = 'long'
                            order_status = 'order_placed'
                            pos.append(1)
                        else:
                            order_status = 'no_order'
                            pos.append(0)
                else:
                    pos.append(0)

            elif order_status == 'order_placed':
                if self.method == 'single':
                    if self.strategy == 'rsi':
                        window = params[0]
                        buy_sig = params[1]
                        sell_sig = params[2]

                        rsi = talib.RSI(test_set, timeperiod=window)
                        pre_rsi = rsi[i - 1]
                        cur_rsi = rsi[i]

                        if ta_direc == 'long':
                            if (pre_rsi < cur_rsi and (abs(cur_rsi - sell_sig) / sell_sig) < error_tol) or (
                                    cur_rsi > sell_sig):
                                order_status = 'no_order'
                                ta_direc = 'no_direc'
                                pos.append(0)
                            else:
                                pos.append(1)
                        elif ta_direc == 'short':
                            if (pre_rsi > cur_rsi and (abs(cur_rsi - buy_sig) / buy_sig) < error_tol) or (
                                    cur_rsi < buy_sig):
                                order_status = 'no_order'
                                ta_direc = 'no_direc'
                                pos.append(0)
                            else:
                                pos.append(-1)

                    elif self.strategy == 'sma':
                        short_window = params[0]
                        long_window = params[1]
                        sma_short = talib.MA(self.price, timeperiod=short_window)
                        sma_long = talib.MA(self.price, timeperiod=long_window)

                        pre_sma_short = sma_short[i - 1]
                        cur_sma_short = sma_short[i]
                        pre_sma_long = sma_long[i - 1]
                        cur_sma_long = sma_long[i]

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

                elif self.method == 'multiple':
                    if self.strategy == 'RSInSMA':
                        window = params[0]
                        buy_sig = params[1]
                        sell_sig = params[2]
                        short_window = params[3]
                        long_window = params[4]
                        exit_sig = params[5]

                        rsi = talib.RSI(test_set, timeperiod=window)
                        sma_short = talib.MA(self.price, timeperiod=short_window)
                        sma_long = talib.MA(self.price, timeperiod=long_window)

                        pre_rsi = rsi[i - 1]
                        cur_rsi = rsi[i]

                        pre_sma_short = sma_short[i - 1]
                        cur_sma_short = sma_short[i]
                        pre_sma_long = sma_long[i - 1]
                        cur_sma_long = sma_long[i]

                        if exit_sig == 0:
                            if ta_direc == 'long':
                                if (pre_rsi < cur_rsi and (abs(cur_rsi - sell_sig) / sell_sig) < error_tol) or (
                                        cur_rsi > sell_sig):
                                    order_status = 'no_order'
                                    ta_direc = 'no_direc'
                                    pos.append(0)
                                else:
                                    pos.append(1)
                            elif ta_direc == 'short':
                                if (pre_rsi > cur_rsi and (abs(cur_rsi - buy_sig) / buy_sig) < error_tol) or (
                                        cur_rsi < buy_sig):
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

        strat_ret = strategy_ret(pos, test_ret)
        pred_cum_ret = strategy_cum_ret(strat_ret)[-1]
        enter_pos, exit_pos = trade_timing(pos)

        pred_winrate, _ = trading_metric(strat_ret, enter_pos, exit_pos)
        print(f"The Predicted Strategy Return: {pred_cum_ret}")
        print(f"The Predicted Win Rate: {pred_winrate}")
        return pred_cum_ret, pred_winrate