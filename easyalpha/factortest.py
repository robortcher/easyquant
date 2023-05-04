"""function and class for factor testing"""

from __future__ import annotations

from typing import Callable,Iterable
import pandas as pd
from dataclasses import dataclass
from functools import partial
import time
from multiprocessing import cpu_count


__date_col__, __symbol_col__ = 'date', 'code'


@dataclass
class FactorTesting():
    """用于做因子分析的类，封装了常用方法"""
    factor_data:pd.DataFrame 
    price_data:pd.DataFrame=None
    price_api:Callable|None=None
    freq:str|None=None
    
    def __post_init__(self):
        if  self.freq is not None:
            from easyprocessor.nmprocessor import bar_resample
            self.factor_data = bar_resample(self.factor_data,frequency=self.freq,symbol_level=__symbol_col__)
            self.price_data = bar_resample(self.price_data,frequency=self.freq,symbol_level=__symbol_col__)
        
        assert self.price_data is not None or self.price_api is not None, "you must input at least one of price_data or price_api"
        self.date_period = sorted(list(set(self.factor_data.index.get_level_values(__date_col__))))
        self.early_start = self.__get_early_start__()
        self.last_end = self.__get_last_end__()
        if self.price_data is None:
            self.price_data = self.price_api(start=self.early_start,end_date=self.last_end)
            print("debug code......")    
    
    def __get_early_start__(self):
        """根据factor_data获取最早日期"""
        return min(self.date_period)

    def __get_last_end__(self):
        """根据factor_data获取最终日期"""
        return max(self.date_period)


    def calculate_forward_return(
        self,
        base:str='close',
        forward_period:int=1,
        cumulative:bool=True,
        universe:pd.Series|None=None,
        excess_return:bool=False,
        benchmark_weights:pd.Series|None=None,
        # compound:bool=False
    )->pd.DataFrame:
        """计算stock每日收益"""
        # format_base = 'base'.replace(' ','').split('-')
        # if format_base.__len__()==2:
        #     _c,_o = format_base
        #     return_data = (self.price_data[_c].unstack() / self.price_data[_o].unstack() - 1)
        # elif format_base.__len__()==1:
        #     return_data  = self.price_data[base].unstack().pct_change()
        daily_return = self.price_data[base].unstack().sort_index().pct_change().stack()
        if universe is not None:
            daily_return = daily_return[universe.reindex(daily_return.index)>0]

        if excess_return:
            if benchmark_weights is not None:
                if universe is not None:
                    benchmark_weights = benchmark_weights[universe.reindex(benchmark_weights.index)>0]
                if isinstance(benchmark_weights,pd.DataFrame):
                    benchmark_weights = benchmark_weights.iloc[:,0]
            from easyprocessor.csprocessor import CsCapweighted
            from easyprocessor.nmprocessor import _add_ts_data
            
            cs_cap = CsCapweighted(weight_data=benchmark_weights) # 构造加权处理器
            benchmark_return = cs_cap(daily_return) # 将时序数据在截面上扩充
            benchmark_return_tcs = _add_ts_data(tcs_data=daily_return, ts_data=benchmark_return,merge=False)
            daily_return = daily_return - benchmark_return_tcs
        
# ##########

            
#             if forward_period == 0: # 0 期收益指定为0 
#                 return daily_return

#             elif forward_period>0:
#                 if cumulative:
#                     return_data= (price.shift(-forward_period) / price - 1).shift(-add_shift)
#                 else:
#                     return_data= (price.shift(-forward_period) / price.shift(-forward_period+1) - 1).shift(-add_shift)
#             else:
#                 if cumulative:
#                     return_data= (price / price.shift(-forward_period) - 1).shift(-add_shift)
#                 else:
#                     return_data= (price.shift(-forward_period-1) / price.shift(-forward_period) - 1).shift(-add_shift)
# #########

        if cumulative:
            return_data  = daily_return.unstack().rolling(forward_period).sum().shift(-forward_period)
        else:
            return_data  = daily_return.unstack().shift(-forward_period)
        return_data = return_data.stack()
        return return_data

    
    def _get_factor_ic(
        self,
        lag:int=1,
        method:str='normal',
        forward_period:int=1,
        cumulative:bool=True,
        universe:pd.Series|None=None,
        excess_return:bool=False,
        benchmark_weights:pd.Series|None=None,
    )->pd.DataFrame:
        """内部get factor ic,universe开发"""
        
        if universe is not None:
            shift_factor = self.factor_data.reindex(universe.index)[universe>0].groupby(level=__symbol_col__).apply(lambda x:x.shift(lag))
        else:
            shift_factor = self.factor_data.groupby(level=__symbol_col__).apply(lambda x:x.shift(lag))
        return_data = self.calculate_forward_return(forward_period=forward_period,cumulative=cumulative,excess_return=excess_return,benchmark_weights=benchmark_weights,universe=universe)

        from easyprocessor.csprocessor import CsCorr
        cs_corr = CsCorr(method=method)
        ic = cs_corr(shift_factor,return_data)
        return ic

    def get_factor_ic(
        self,
        lags:Iterable=(1,),
        method:str='normal',
        forward_period:int=1,
        cumulative:bool=True,
        universe:pd.Series|None=None,
        excess_return:bool=False,
        benchmark_weights:pd.Series|None=None,
    )->pd.DataFrame:
        ic_func = partial(self._get_factor_ic,method=method,forward_period=forward_period,cumulative=cumulative,universe=universe,excess_return=excess_return,benchmark_weights=benchmark_weights)
        
        from multiprocessing import Pool
        cpu_worker_num = min(lags.__len__(),8)
        process_args=lags
        print(f'| lag inputs:  {process_args}')
        start_time = time.time()
        with Pool(cpu_worker_num) as p:
            outputs = p.map(ic_func, process_args)
        print(f'| TimeUsed: {time.time() - start_time:.1f}    \n')
        outputs = {lags[_i]: outputs[_i] for _i in range(len(lags))}
        outputs = pd.concat(outputs)._set_axis_name(name=['lag','date'])
        return outputs



    def get_quantile_factor(self,groups:int=10,lag:int=1,universe:pd.Series|None=None):
        """获取factor的quantile"""
        
        from easyprocessor.csprocessor import Csqcut
        cs_qcut = Csqcut(q=groups)
        if universe is not None:
            shift_factor = self.factor_data[universe.reindex(self.factor_data.index)>0].groupby(level=__symbol_col__).apply(lambda x:x.shift(lag))
        else:
            shift_factor = self.factor_data.groupby(level=__symbol_col__).apply(lambda x:x.shift(lag))
        # shift
        return cs_qcut(shift_factor)
    

    def get_factor_group_return(
            self,
            groups:int=10,
            forward_period:int=1,
            lag:int=1,
            universe:pd.Series|None=None,
            base:str='close',
            cumulative:bool=True,
            excess_return:bool=False,
            benchmark_weights:pd.Series|None=None
        ):
        """获取不同分组的因子收益"""
        quantile_factor = self.get_quantile_factor(groups=groups,lag=lag,universe=universe)
        return_data = self.calculate_forward_return(base=base,forward_period=forward_period,benchmark_weights=benchmark_weights,cumulative=cumulative,excess_return=excess_return)
        factor_group_return = quantile_factor.apply(lambda x:return_data.groupby(x).apply(lambda x:x.groupby(level=__date_col__).mean()))
        # factor_turnover = self.get_quantile_turnover(quantile_data=quantile_factor)
        quantile_turnover = quantile_factor.apply(get_quantile_turnover)
        return factor_group_return, quantile_turnover
    
def get_quantile_turnover(quantile_data:pd.Series):
    """根据quantile group计算换手"""
    default_weight = pd.Series(1,index=quantile_data.index)
    def _func(x):
        daily_weight = x.unstack().apply(lambda x:x/x.sum(),axis=1)
        daily_turnover = daily_weight.fillna(0).diff().abs().sum(axis=1) / 2 # /2 是因为求的单边换手
        daily_turnover.iloc[0] = daily_weight.iloc[0].sum()
        return daily_turnover
    quantile_turnover = default_weight.groupby(quantile_data).apply(_func)
    return quantile_turnover

