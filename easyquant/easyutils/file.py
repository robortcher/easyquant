import pickle
import os
import pandas as pd

def load_pkl(filename):
    """载入pickle文件"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data 

def save_pkl(context, filename):
    """存储文件到pickle"""
    with open(filename, 'wb') as f:
        data = pickle.dump(context, f)
    return data


def load_pkls(dire):
    paths = [dire + '/' + path for path in os.listdir(dire)]
    dfs = [load_pkl(path) for path in paths]
    return pd.concat(dfs)