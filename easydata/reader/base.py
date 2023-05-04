
from  __future__ import annotations 

from typing import Callable
from ..config.base import BaseConfig
from sqlalchemy import create_engine
import pandas as pd




class BaseReader():
    """
    base sql reader class from database which can be usually filtered by date col and symbol col
    """
    def __init__(
        self,
        config:BaseConfig,
        symbol_col:str|None=None,
        date_col:str|None=None,
        date_format:str|None=None,
        cols_handler:Callable|dict|None=None,
        describe:str|None=None
    ):
        self.config =config
        self.symbol_col = symbol_col
        self.date_col = date_col
        self.engine = create_engine(config.url)
        self.date_format=date_format
        self.cols_handler =cols_handler
        self.table_name = getattr(config, 'table_name',None)
        self.describe = describe
    
    def get_all_fields(self):
        """return all tables the database includes or columns the table includes."""
        return self.config.struct.get_all_fields()
    
    
    def __sql_format(self,**kwargs):
        """
        """
        fields = kwargs.get("fields",None)
        universe = kwargs.get("universe",None)
        start_date = kwargs.get('start_date',None)
        end_date = kwargs.get('end_date',None)
        table_name = kwargs.get('table_name',None)
        assert table_name is not None, f"you must input table_name in query function or init reader with table_name"
        
        
        # symbol filter
        if universe is None:
            symbol_filter = 'True'
        else:
            assert getattr(self,'symbol_col',None) is not None,"please set column of symbol if you want to filter data by universe"
            if isinstance(universe,str):
                universe=[universe]
            universe =["\'" + i.strip() + "\'" for i in universe]
            universe_str = ",".join(universe)
            symbol_filter = f"(`{self.symbol_col}`in ({universe_str}))"

        # date fiter -- start
        if start_date is None:
            start_filter = 'True'
        else:
            getattr(self,'date_col',None) is not None, "please set column of date if you want to filter data by universe"
            if self.date_format is not None:
                start_date = pd.to_datetime(start_date).strftime(self.date_format)
            start_filter = f"(`{self.date_col}` >= \'{start_date}\')"
            
        # date filter -- end
        if end_date is None:
            start_filter = 'True'
        else:
            getattr(self,'date_col',None) is not None, "please set column of date if you want to filter data by universe"
            if self.date_format is not None:
                end_date = pd.to_datetime(end_date).strftime(self.date_format)
            end_filter = f"(`{self.date_col}` <= \'{end_date}\')"


        # fields filter
        if fields is None:
            fields_filter = '*'
        
        else:
            cur_fields = [self.date_col,self.symbol_col]+fields
            cur_fields = [_c for _c in cur_fields if _c is not None]
            fields_filter = ','.join(["\""+i+"\"" for i in cur_fields])
        sql = f"select {fields_filter} from {table_name} where ({start_filter}) and ({end_filter}) and ({symbol_filter})"
        if self.config.manager == 'postgresql':
            sql = sql.replace('`','\"')
        return sql
    
    def __getitem__(self,key):
        if self.config.struct.name == 'database':
            return BaseReader(config=self.config(table_name=key),symbol_col=self.symbol_col,date_col=self.date_col,date_format=self.date_format,cols_handler=self.cols_handler)
        elif self.config.struct.name == 'server':
            return BaseReader(config=self.config(database=key),symbol_col=self.symbol_col,date_col=self.date_col,date_format=self.date_format,cols_handler=self.cols_handler)

    def query(
            self,
            universe:list|None=None,
            fields:list|None=None,
            start_date:str|None=None,
            end_date:str|None=None,
            trade_date:str|None=None,
            table_name:str|None=None
    ):
        """
        """
        if trade_date is not None:
            start_date = trade_date
            end_date = trade_date
        
        if table_name is None:
            table_name = getattr(self.config,'table_name')
        
        sql = self.__sql_format(universe=universe,start_date=start_date,end_date=end_date,fields=fields,table_name=table_name)
        _df = pd.read_sql(sql=sql,con=self.engine)

        final_index=[]
        if self.date_col is not None:
            _df[self.date_col] = pd.to_datetime(_df[self.date_col])
            _df = _df.rename(columns={self.date_col:'date'})
            final_index.append('date')
        
        if self.symbol_col is not None:
            _df = _df.rename(columns={self.symbol_col:'code'})
            final_index.append('code')
        
        if final_index.__len__()>0:
            _df = _df.set_index(final_index)
            _df = _df[~_df.index.duplicated()]
        
        #### cols_handler
        if isinstance(self.cols_handler,dict):
            _df = _df.renmae(columns=self.cols_handler)

        elif isinstance(self.cols_handler,Callable):
            _df.columns = _df.columns.map(self.cols_handler)
        
        return _df
    

    def __call__(
            self,
            universe=None,
            fields=None,
            start_date=None,
            end_date=None,
            trade_date=None,
            table_name=None
    ):
        return self.query(
            universe=universe,
            fields=fields,
            start_date=start_date,
            end_date=end_date,
            trade_date=trade_date,
            table_name=table_name
        )
