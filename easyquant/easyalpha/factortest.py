"""function and class for factor testing"""

from __future__ import annotations

from typing import Callable,Iterable
import pandas as pd
from dataclasses import dataclass
from functools import partial
import time
from multiprocessing import cpu_count
import numpy as np
from itertools import product
from ..easyperformance.common import performance_indicator
from ..easyprocessor.csprocessor import CsCapweighted
from ..easyprocessor.nmprocessor import _add_ts_data
from ..easyprocessor.csprocessor import CsCorr,Csqcut

__date_col__, __symbol_col__ = 'date', 'code'


@dataclass
class FactorTesting():
    """用于做因子分析的类，封装了常用方法"""
    factor_data:pd.DataFrame 
    price_data:pd.DataFrame=None
    price_api:Callable|None=None
    freq:str|int=1
    start_date:str|None=None
    end_date:str|None=None
    def __post_init__(self):
        # if  self.freq is not None:
        #     from ..easyprocessor.nmprocessor import bar_resample
        #     self.factor_data = bar_resample(self.factor_data,frequency=self.freq,symbol_level=__symbol_col__)
        #     self.price_data = bar_resample(self.price_data,frequency=self.freq,symbol_level=__symbol_col__)
        if isinstance(self.factor_data,pd.Series):
            self.factor_data = pd.DataFrame(self.factor_data)
        if self.start_date is not None:
            self.factor_data = self.factor_data[self.factor_data.index.get_level_values(__date_col__)>=self.start_date]
        if self.end_date is not None:
            self.factor_data = self.factor_data[self.factor_data.index.get_level_values(__date_col__)<=self.end_date]

        assert self.price_data is not None or self.price_api is not None, "you must input at least one of price_data or price_api"
        self.date_period = sorted(list(set(self.factor_data.index.get_level_values(__date_col__))))
        self.early_start = self.__get_early_start__()
        self.last_end = self.__get_last_end__()
        self.factor_data = self.factor_data.groupby(level=__symbol_col__).apply(lambda x:x.droplevel(__symbol_col__).reindex(self.date_period)).swaplevel(0,1)
        
        if self.price_data is None:
            self.price_data = self.price_api(start=self.early_start,end_date=self.last_end)
            print("debug code......")    
        self.price_data = self.price_data[(self.price_data.index.get_level_values(0)>=self.early_start) & (self.price_data.index.get_level_values(__date_col__)<=self.last_end)]
        

    def __get_early_start__(self):
        """根据factor_data获取最早日期"""
        return min(self.date_period)

    def __get_last_end__(self):
        """根据factor_data获取最终日期"""
        return max(self.date_period)

    def _get_factor_data(self,factor_list:str|list|None=None):
        if factor_list is None:
            return self.factor_data
        elif isinstance(factor_list,list):
            return self.factor_data[factor_list]
        elif isinstance(factor_list,str):
            return self.factor_data[[factor_list]]


    # def __all_date_range__(self):
    #     return sorted(list(set(self.factor_data.index.get_level_values(__date_col__))))

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
        daily_return = self.price_data.sort_index()[base].groupby(level=__symbol_col__).apply(lambda x:x.pct_change())
        # daily_return = self.price_data[base].groupby(level=__symbol_col__).apply(lambda x:x.shift(-forward_period)/x-1)
        
        if universe is not None:
            daily_return = daily_return[universe.reindex(daily_return.index)>0]

        if excess_return:
            if benchmark_weights is not None:
                if universe is not None:
                    benchmark_weights = benchmark_weights[universe.reindex(benchmark_weights.index)>0]
                if isinstance(benchmark_weights,pd.DataFrame):
                    benchmark_weights = benchmark_weights.iloc[:,0]

            
            cs_cap = CsCapweighted(weight_data=benchmark_weights) # 构造加权处理器
            benchmark_return = cs_cap(daily_return) # 将时序数据在截面上扩充
            benchmark_return_tcs = _add_ts_data(tcs_data=daily_return, ts_data=benchmark_return,merge=False)
            daily_return = daily_return - benchmark_return_tcs.iloc[:,0]
        

        if cumulative:
            # return_data  = daily_return.unstack().rolling(forward_period).sum().shift(-forward_period)
            return_data = daily_return.groupby(level=__symbol_col__).apply(lambda x:x.rolling(forward_period).sum().shift(-forward_period))
        else:
            return_data  = daily_return.groupby(level=__symbol_col__).apply(lambda x:x.shift(-forward_period))
        # return_data = return_data.stack()
        return return_data

    def _get_factor_ic(
        self,
        lag:int=1,
        base:str='close',
        method:str='normal',
        forward_period:int=1,
        cumulative:bool=True,
        universe:pd.Series|None=None,
        excess_return:bool=False,
        benchmark_weights:pd.Series|None=None,
        factor_list:str|list|None=None
    )->pd.DataFrame:
        """内部get factor ic,universe开发"""
        
        factor_data = self._get_factor_data(factor_list=factor_list)

        if universe is not None:
            shift_factor = factor_data.reindex(universe.index)[universe>0].groupby(level=__symbol_col__).apply(lambda x:x.shift(lag))
        else:
            shift_factor = factor_data.groupby(level=__symbol_col__).apply(lambda x:x.shift(lag))
        return_data = self.calculate_forward_return(forward_period=forward_period,cumulative=cumulative,base=base,excess_return=excess_return,benchmark_weights=benchmark_weights,universe=universe)

        
        cs_corr = CsCorr(method=method)
        ic = cs_corr(shift_factor,return_data)
        return ic


    def get_factor_ic(
        self,
        lags:Iterable=(1,),
        base:str='close',
        method:str='normal',
        forward_period:int=1,
        cumulative:bool=True,
        universe:pd.Series|None=None,
        excess_return:bool=False,
        benchmark_weights:pd.Series|None=None,
        factor_list:str|list|None=None
    )->pd.DataFrame:
        ic_func = partial(self._get_factor_ic,method=method,base=base,forward_period=forward_period,cumulative=cumulative,universe=universe,excess_return=excess_return,benchmark_weights=benchmark_weights,factor_list=factor_list)
        
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

    def get_quantile_factor(self,groups:int=10,lag:int=1,universe:pd.Series|None=None,holding_period:int=1,factor_list:str|list|None=None):
        """获取factor的quantile"""
        
        # from ..easyprocessor.csprocessor import Csqcut
        assert lag>=0, 'please input a positive int of lag'
        cs_qcut = Csqcut(q=groups)
        factor_data = self._get_factor_data(factor_list=factor_list)
        if universe is not None:
            shift_factor = factor_data[universe.reindex(factor_data.index)>0].groupby(level=__symbol_col__).apply(lambda x:x.shift(lag).ioc[lag:])
        else:
            shift_factor = factor_data.groupby(level=__symbol_col__).apply(lambda x:x.shift(lag).iloc[lag:])
        # shift
        if shift_factor.index[0].__len__()==3:
            shift_factor = shift_factor.droplevel(0)
        quantile_data = cs_qcut(shift_factor).astype(np.float)
        if holding_period >1 :
            quantile_data = quantile_data.fillna(-1).groupby(level=__symbol_col__).apply(lambda x:x[::holding_period].reindex(x.index).ffill()) 
            quantile_data = quantile_data.replace(-1,np.nan)
        return quantile_data
    
    def get_factor_group_return(
            self,
            groups:int=10,
            # forward_period:int=1,
            lag:int=1,
            universe:pd.Series|None=None,
            base:str='close',
            cumulative:bool=True,
            excess_return:bool=False,
            benchmark_weights:pd.Series|None=None,
            holding_period:int=1,
            factor_list:str|list|None=None, 
        ):
        """获取不同分组的因子收益"""
        quantile_factor = self.get_quantile_factor(groups=groups,lag=lag,universe=universe,holding_period=holding_period,factor_list=factor_list)
        return_data = self.calculate_forward_return(base=base,universe=universe,benchmark_weights=benchmark_weights,cumulative=cumulative,excess_return=excess_return)
        factor_group_return = quantile_factor.apply(lambda x:return_data.groupby(x).apply(lambda x:x.groupby(level=__date_col__).mean()))
        # factor_turnover = self.get_quantile_turnover(quantile_data=quantile_factor)
        Long_short_return = factor_group_return.loc[groups-1] -factor_group_return.loc[0]
        Long_short_return.index = pd.MultiIndex.from_tuples(product(['Long-Short'],Long_short_return.index))
        factor_group_return = pd.concat([factor_group_return,Long_short_return])
        quantile_turnover = quantile_factor.apply(get_quantile_turnover)
        # factor_group_return=factor_group_return.groupby(level=0).apply(lambda x:x.shift(lag+1))
        return factor_group_return, quantile_turnover
    
    def get_summary_report(self,
            lag:int=1,
            base:str='close',
            method:str='normal',
            forward_period:int=1,
            cumulative:bool=True,
            universe:pd.Series|None=None,
            excess_return:bool=False,
            benchmark_weights:pd.Series|None=None,
            groups:int=10,
            holding_period:int=1,
            factor_list:str|list|None=None
        ):

            ic = self._get_factor_ic(
                lag=lag,
                base=base,
                method=method,
                forward_period=forward_period,
                cumulative=cumulative,
                universe=universe,
                excess_return=excess_return,
                benchmark_weights=benchmark_weights,
                factor_list=factor_list
            )

            group_returns,group_turnover = self.get_factor_group_return(
                groups=groups,
                lag=lag,
                holding_period=holding_period,
                universe=universe,
                excess_return=excess_return,
                benchmark_weights=benchmark_weights,
                cumulative=cumulative,
                factor_list=factor_list,
                base=base
            )
            universe_name = universe if hasattr(universe,'name') else universe
            describe = f'|{self.early_start}~{self.last_end}|groups:{groups}|is_excess:{excess_return}|unverse:{universe_name}'

            result_ic = ic_analysis(ic)
            # result_group_return = group_returns.stack().groupby(level=0).apply(lambda x:performance_indicator((1+x.unstack()).cumprod(),freq=self.freq,language='en'))
            # group_returns=group_returns.rename(index={0:'Short',groups-1:'Long'})
            # group_turnover=group_turnover.rename(index={0:'Short',groups-1:'Long'})
            result_group_return = group_returns.groupby(level=0).apply(lambda x:performance_indicator((1+x).cumprod(),freq=self.freq,language='en'))
            # result_group_return = group_returns.apply(partial(_group_return_analysis,freq=self.freq))
            result_group_return = result_group_return[(result_group_return.index.get_level_values(1).isin(['AnnRet','SR'])) & (result_group_return.index.get_level_values(0).isin([0,groups-1,'Long-Short']))].swaplevel(0,1)
            result_group_return=result_group_return.rename(index={0:'Short',groups-1:'Long'},level=1)

            result_group_turnover = group_turnover.groupby(level=0).mean()
            result_group_turnover = result_group_turnover.loc[[0,groups-1]]
            result_group_turnover.index = pd.MultiIndex.from_tuples(product(['TO'],result_group_turnover.index))
            result_group_turnover=result_group_turnover.rename(index={0:'Short',groups-1:'Long'},level=1)

            result = pd.concat([result_ic,result_group_return,result_group_turnover])
            print(describe)
            return result
            

    def get_detail_report(
            self,
            factor_name:str,
            lag:int=1,
            lags:Iterable=range(-6,12),
            base:str='close',
            method:str='normal',
            forward_period:int=1,
            cumulative:bool=True,
            universe:pd.Series|None=None,
            excess_return:bool=False,
            benchmark_weights:pd.Series|None=None,
            groups:int=10,
            holding_period:int=1,
            plot:str|None=None
        ):
            result={}
            ic_series=self._get_factor_ic(
                lag=lag,
                factor_list=factor_name,
                base=base,
                forward_period=forward_period,
                universe=universe,
                excess_return=excess_return,
                benchmark_weights=benchmark_weights,
                cumulative=cumulative,
                method=method
            )

            ic_decay=self.get_factor_ic(
                lags=lags,
                base=base,
                method=method,
                forward_period=forward_period,
                universe=universe,
                excess_return=excess_return,
                benchmark_weights=benchmark_weights,
                factor_list=factor_name,
                cumulative=False
            )
            result['ic_decay'] = ic_decay[factor_name].unstack().T
            result['ic_series'] = ic_series[factor_name]
            result['groups'] = {}
            result['factor'] = factor_name
            result['universe'] = universe.name if hasattr(universe,'name') else universe
            groups = (groups,) if isinstance(groups,int) else groups
            for _group in groups:
                performance_result = {}
                group_return,group_turnover = self.get_factor_group_return(
                    groups=_group,
                    lag=lag,
                    base=base,
                    universe=universe,
                    cumulative=cumulative,
                    excess_return=excess_return,
                    benchmark_weights=benchmark_weights,
                    holding_period=holding_period,
                    factor_list=factor_name
                )
                group_return = group_return[factor_name].unstack().T
                group_nvs = (1+group_return).cumprod().drop(['Long-Short'],axis=1)
                
                group_nvs.name = 'holding_period=' + str(holding_period)
                group_turnover = group_turnover[factor_name].unstack().T.mean()


                perfs = performance_indicator(group_nvs,ret_data=True,language='en',freq=self.freq)
                perfs.loc['turnover_ratio'] = group_turnover
        
                performance_result['group_return_short'] = group_return[0]
                performance_result['group_return_long'] = group_return[_group-1]

                performance_result['group_nvs'] = group_nvs
                performance_result['ann_ret'] = perfs.loc['AnnRet']
                performance_result['SR'] = perfs.loc['SR']
                performance_result['TO'] = perfs.loc['turnover_ratio']
                
                performance_result['excess_performance'] = round(perfs,4)
                result['groups'][f'G{_group}'] = performance_result
                
                print
        
            if plot is not None:
                try:
                    from plotting import summary_plot
                except:
                    from .plotting import summary_plot
                summary_plot(result,excess=excess_return,mode=plot)
            return result
    
    def factor_ana(self,factor,ep_col,liquidity_col,universe=None,**kwargs):
        """获取不同组别factor的"""
        def get_ep_liq_group_ic(factor,group_df,**kwargs):
            ics = {}
            
            group_set = ['low','medium','high']#set(group_df.values)
            for group in group_set:
                ics[group] = ic_analysis(FactorTesting(price_api=self.price_data,factor_data=self.factor_data[factor],freq=self.freq,winzorize=self._winzorize,standardize=self._standardize,industry_neutralize=self._industry_neutralize),universe=(group_df==group).astype(np.int), **kwargs)
            ics['total'] = ic_analysis(FactorTesting(bar_data=self.bar_data,factor_data=self.factor_data[factor],freq=self.freq,winzorize=self._winzorize,standardize=self._standardize,industry_neutralize=self._industry_neutralize),universe=universe,**kwargs)
            ics = pd.concat(ics)
            # ics = ics.reindex(['total'] + ['low','moderate_low','medium','moderate_high','high'],axis=1)
            return ics
        res = {}

        ep_data = self.bar_data[ep_col]
        if universe is not None:
            ep_data = ep_data[universe>0]
        ep_group = ep_data.groupby(level=0).apply(lambda x:pd.qcut(x,3,labels=['low','medium','high']))

        res['ep groups'] = get_ep_liq_group_ic(factor,group_df=ep_group,**kwargs)

        liquidity_data = self.bar_data[liquidity_col]
        if universe is not None:
            liquidity_data = liquidity_data[universe>0]
        liquidity_group = liquidity_data.groupby(level=0).apply(lambda x:pd.qcut(x,3,labels=['low','medium','high']))
        
        
        # 不同liquidity组别ic均值输出
        res['liquidity groups'] = get_ep_liq_group_ic(factor,group_df=liquidity_group,**kwargs)
        # bar3 = px.bar(t[t.index.get_level_values(2)=='ic.mean'].droplevel(2).unstack().droplevel(0,axis=1),barmode='group',title=f'avg.ic of different liquidity groups of {factor}')
        # # HTML(bar3.to_html())
        # bar3.show()
        res = round(pd.concat(res).unstack().droplevel(0,axis=1),4)
        res.index = res.index.set_names(['group','set'])
        return res.reset_index()

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


def ic_analysis(ic_seires,forward_period=1):
    """return mean ic and ic t-stats"""
    result = {}
    result['ic.mean'] = ic_seires.mean()
    result['ic.t-stats'] = ic_seires.mean() / ic_seires.std() * ic_seires.count() ** 0.5 / forward_period**0.5
    return pd.concat(result).unstack()

def _group_return_analysis(group_returns,freq=1):
    
    wide_return = group_returns.unstack().T
    nv = (1+wide_return.fillna(0)).cumprod()
    perfs = performance_indicator(nv,language='en',freq=freq)    
    return perfs



