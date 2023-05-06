from .base import BaseConfig

pgsql_local:BaseConfig=BaseConfig(host='127.0.0.1',manager='postgresql',user='postgres',password='postgres',port='5432')
finance_local:BaseConfig=pgsql_local(database='finance')
price_local:BaseConfig=finance_local(table_name='price_post')