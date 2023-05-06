

from __future__ import annotations



import pandas as pd
from copy import deepcopy
from typing import Callable
from dataclasses import dataclass
from sqlalchemy import create_engine


def __check_url_level__(config:BaseConfig)->BaseStruct:
    """检查当前config等级，并返回相应类对象"""
    if config.database is None:
        return Server
    else:
        if config.table_name is None:
            return Database
        else:
            return Table


@dataclass
class BaseConfig():
    host:str
    manager:str
    user:str
    password:str
    port:int
    database:str|None=None
    table_name:str|None=None
    name:str|None=None
    describe:str|None=None


    @property
    def url(self)->str:
        if self.database is None:
            url = f"{self.manager}://{self.user}:{self.password}@{self.host}:{self.port}"
        else:
            url = f"{self.manager}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        return url
    
    @property
    def server(self):
        return self.host
    
    def __call__(self, **kwargs):
        new_obj = deepcopy(self)
        for _k,_v in kwargs.items():
            setattr(new_obj,_k,_v)
        return new_obj
    
    @property
    def struct(self)->BaseStruct:
        struct = __check_url_level__(self)
        return struct(self)
    
    def get_all_fields(self):
        return self.struct.get_all_fields()
    
class BaseStruct():
    def __init__(self,config:BaseConfig)->None:
        self.config=config
        self.engine=create_engine(config.url)
    
    @property
    def url(self):
        return self.config.url
    
    @property
    def table_name(self):
        return self.config.table_name
    
    @property
    def database(self):
        return self.config.database
    
    @property
    def server(self):
        return self.config.server

    def base_query(self,sql:str)->pd.DataFrame:
        return pd.read_sql(sql=sql,con=self.engine)

class Server(BaseStruct):
    name = 'server'
    def get_all_fields(self):
        # if self.config .
        if self.config.manager=='postgresql':
            sql = "select datname from pg_database where datistemplate=false"
        elif self.config.manager=='mysql':
            sql = 'show databases'
        return self.base_query(sql)

    def __getitem__(self,database):
        assert database.lower() in self.get_all_fields().values, F"Server:{self.config.server} has no database named{database}"
        return Database(self.config(database=database))

class Database(BaseStruct):
    name = 'database'

    def get_all_fields(self):
        if self.config.manager == 'mysql':
            sql = "select table_name as table_name from information_schema.tables where table_schema='{self.config.database}'"
        elif self.config.manager == 'postgresql':
            sql = f"SELECT table_name FROM information_schema.tables WHERE table_schema='public'" 
        return self.base_query(sql)['table_name']

    def __getitem__(self,table_name):
        assert table_name.lower() in self.get_all_fields().values, F"database:{self.config.database} has no table_name named{table_name}"
        return Database(self.config(table_name=table_name))

class Table(BaseStruct):
    name = 'table'
    def get_all_fields(self):
        if self.config.manager == 'mysql':
            sql = f"select column_name as field from information_schema.columns where table_schema='{self.config.database}' and table_name='{self.config.table_name}'"
        
        elif self.config.manager == 'postgresql':
            sql = f"select column_name as field from information_schema.columns where table_schema='public' and table_name='{self.table_name}'"
        return self.base_query(sql)['field']
