# -*- coding: utf-8 -*-
"""
收益评价相关函数
"""
from __future__ import annotations

import pandas as pd
import numpy as np

__frequency_dict__ = {
    'Y':1,
    'M':12,
    'D':252,
    'W':52,
    'Q':3
}

def performance_indicator(nvs:pd.DataFrame, method='compound',ret_data=False,riskless_ret=0.0,freq:str|int='D',language='cn',mkt_return=None):
    """给定指定净值或收盘价计算相关投资收益评估指标"""
    if isinstance(freq,str):
        assert freq in __frequency_dict__,f"unsurported frequency {freq}"
        freq = __frequency_dict__[freq]
    nvs = pd.DataFrame(nvs)
    def __performance_indicator(nv:pd.Series, ret_data=ret_data,riskless_ret=riskless_ret):
        #装载累计收益
        if ret_data:
            nv = (1+nv).cumprod()

        nv_df = nv.ffill().dropna()
        nv_df = nv_df.bfill()
        portfolio_ret = nv_df / nv_df.iloc[0]
        #计算每日收益
        portfolio_ret_daily = portfolio_ret.pct_change()
    
        
        days_number = nv.__len__()
        
        #计算各类指标
        ret_cum = portfolio_ret.iloc[-1] - 1
        ret_ann = portfolio_ret_daily.mean()*freq#(ret_cum+1) ** (freq / (len(nv_df)-1)) - 1
        sigma_ann = portfolio_ret_daily.std()*(freq**0.5)

        # ret_yoy = (ret_cum+1) ** (250 / len(nv_df)) - 1
        max_drawdown = ((portfolio_ret.cummax() - portfolio_ret)/portfolio_ret.cummax()).max()
        
        # ann_return1Y = nv_df.iloc[-1] / nv_df.iloc[-min((freq*1+1),nv_df.__len__())] -1
        # ann_return2Y = (nv_df.iloc[-1] / nv_df.iloc[-min((freq*2+1),nv_df.__len__())]) **(1/(min(freq*2,nv_df.__len__()-1)/freq))-1
        # ann_return3Y = (nv_df.iloc[-1] / nv_df.iloc[-min((freq*3+1),nv_df.__len__())]) **(1/(min(freq*3,nv_df.__len__()-1)/freq))-1,
        # ann_return5Y = (nv_df.iloc[-1] / nv_df.iloc[-min((freq*5+1),nv_df.__len__())]) **(1/(min(freq*5,nv_df.__len__()-1)/freq))-1,
        
        if max_drawdown == 0:
            calmar_ratio = 0
        else:
            calmar_ratio = ret_ann/max_drawdown
        
        sharpe_ratio = (ret_ann - riskless_ret) / sigma_ann 

        if language.lower() == 'en':
            _performance_indicator = {'ret_cum'    : ret_cum,
                                    'AnnRet'    : ret_ann,
                                    'sigma_ann'  : sigma_ann,
                                    'maxdd':max_drawdown,
                                    'SR':sharpe_ratio,
                                    'calmar':calmar_ratio,
                                    # 'ann.return 1Y ':ann_return1Y,
                                    # 'ann.return 2Y':ann_return2Y,
                                    # 'ann.return 3Y':ann_return3Y,
                                    # 'ann.return 5Y':ann_return5Y,
                                    }
            if mkt_return is not None:
                excess_return = portfolio_ret_daily - mkt_return
                avg_excess_return_ann = excess_return.mean() * freq
                avg_excess_return_stdev_ann = excess_return.std()*freq**0.5
                IR = avg_excess_return_ann / avg_excess_return_stdev_ann
                _performance_indicator['excess_ann'] = avg_excess_return_ann
                _performance_indicator['excess_stdev_ann'] = avg_excess_return_stdev_ann
                _performance_indicator['IR'] = IR
            # _performance_indicator['n_periods'] = str(int(days_number))
        elif language.lower() == 'cn':
            _performance_indicator = {'累计收益':ret_cum,
                            '年化收益':ret_ann,
                            '年化波动率':sigma_ann,
                            '最大回撤':max_drawdown,
                            f'夏普比({riskless_ret})':sharpe_ratio,
                            '年化收益/最大回撤':calmar_ratio,
                            '累计交易周期':days_number,
                            # 'ann.return 1Y ':ann_return1Y,
                            # 'ann.return 2Y':ann_return2Y,
                            # 'ann.return 3Y':ann_return3Y,
                            # 'ann.return 5Y':ann_return5Y,
                            }
    
        _performance_indicator = pd.DataFrame(_performance_indicator,index= ['performance_indicator'])
        _performance_indicator = _performance_indicator.astype(float)
        return _performance_indicator.T
    
    all_df = pd.concat([__performance_indicator(nvs[i],riskless_ret) for i in nvs.columns],axis=1)
    all_df.columns = nvs.columns
    return round(all_df,4)

def cal_turnover_ratio(orders,net_asset_value,price_col='price_filled_avg',volume_col='volume_entrust'):
    """根据订单和每日净资产估算每日换手率
    """
    order_amount = orders[price_col] * orders[volume_col]
    amount_total = order_amount.groupby(level=0).sum()
    amount_buy = order_amount[orders['order_side'].isin([0])].groupby(level=0).sum()
    amount_sell = order_amount[orders['order_side'].isin([1])].groupby(level=0).sum()
    
    turnover_ratio_total = (amount_total / net_asset_value).fillna(0)
    turnover_ratio_buy = (amount_buy / net_asset_value).fillna(0)
    turnover_ratio_sell = (amount_sell / net_asset_value).fillna(0)
    
    turnover_ratio = pd.concat([turnover_ratio_total,turnover_ratio_buy,turnover_ratio_sell],axis=1).astype(np.float)
    turnover_ratio.columns = ['turnover_ratio_total','turnover_ratio_buy','turnover_ratio_sell']
    return turnover_ratio
    
    
