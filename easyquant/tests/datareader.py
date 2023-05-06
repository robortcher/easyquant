#%% data reader

import sys
from pathlib import Path
current_file=Path(r'D:\projects\easyquant\easyquant\tests\datareader.ipynb')
sys.path.append(current_file.parent.parent.parent.__str__())

from easyquant.easydata.config.url import pgsql_local
from easyquant.easydata.reader.base import BaseReader
from easyquant.easyutils.file import load_pkl

from easyquant.easyprocessor.nmprocessor import bar_resample

#%% 从数据库获取数据并处理
my = BaseReader(config=pgsql_local,symbol_col='code',date_col='date')
get_valuation = BaseReader(config=my['finance']['valuation'].config,symbol_col='code',date_col='date')
get_price = BaseReader(config=my['finance']['price_post'].config,symbol_col='code',date_col='date')
test_factor = get_valuation(start_date='2015-01-01',end_date='2022-12-31')
test_price = get_price(start_date='2015-01-01',end_date='2022-12-31')

# test_factor = bar_resample(test_factor,frequency='M')
# test_price = bar_resample(test_price,frequency='M')




#%% get factor ic
if __name__ == '__main__':
    #%% 从pickle文件读取
    # test_factor = load_pkl(r'D:\projects\easyquant\tests\data\test_factor.pkl')
    # test_price = load_pkl(r'D:\projects\easyquant\tests\data\test_price.pkl')
    # test_factor['mom12x3'] = test_price['close'].groupby(level='code').apply(lambda x:x.shift(3) / x.shift(12) -1)

    #%% data processed
    from easyprocessor.csprocessor import CsCapweighted
    cs_cap = CsCapweighted(weight_data=test_factor['market_cap'])
    market_pb_ratio = cs_cap(test_factor[['pe_ratio','pb_ratio']])



    #%% factor testing
    from easyalpha.factortest import FactorTesting
    ft = FactorTesting(price_data=test_price,factor_data=test_factor)
    y = ft.calculate_forward_return(excess_return=True,benchmark_weights=test_factor['market_cap'])


    #%% factor ic
    from easyprocessor.csprocessor import CsCorr
    cs_corr = CsCorr(method='normal')
    cs_corr(test_factor,y)
    
    universe = test_factor['pe_ratio'][test_factor['pe_ratio']>0]
    
    ics = ft.get_factor_ic(lags=list(range(-6,12)),cumulative=False)
    ics2 = ft.get_factor_ic(universe=universe,lags=list(range(1,2)),excess_return=True,benchmark_weights=test_factor['market_cap'])

    #%% get factor group
    group_data  = ft.get_factor_group_return(forward_period=1,excess_return=True,benchmark_weights=test_factor['market_cap'])
    print