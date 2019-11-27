#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yangfenglong
# @Date: 2019-05-20
"""
set of useful functions
usage: import useful_functions as uf
       uf.create_dir(dir)
"""

import os
import hashlib
from datetime import datetime
import logging
import traceback
from functools import wraps
from sklearn.base import BaseEstimator,TransformerMixin

def create_dir(dir):
    if not os.path.exists(dir):
        assert not os.system('mkdir {}'.format(dir))

def get_md5(file):
    md5file = open(file,'rb')
    md5 = hashlib.md5(md5file.read()).hexdigest()
    md5file.close()
    return md5

def now():
    fmt = "%Y%m%d%H%M%S"
    return datetime.now().strftime(fmt)
    
def create_logger(logFile_name):
    """同时输出log到文件和屏幕"""
    logger = logging.getLogger()  # 不加名称设置root logger
    logger.setLevel(logging.INFO)

    sh = logging.StreamHandler() # 使用StreamHandler输出到屏幕
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(message)s', #- %(filename)s:%(lineno)s - %(name)s - 
        datefmt='%Y-%m-%d %H:%M:%S')    
    sh.setFormatter(formatter)
    logger.addHandler(sh) # 添加screen Handler

    fh = logging.FileHandler(logFile_name) # 使用FileHandler输出到文件
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh) #添加file Handle
    return logger


def tryExceptLog(logger):
    """
    
    A decorator that wraps the passed logger in function and logs 
    exceptions should one occur
 
    @param logger: The logging object created by "logger = create_logger(logname)"
    """
     
    def decorator(func):
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.info('called: function:{}, args:{}, kargs:{}'.format(func.__name__, args, kargs))
                return func(*args, **kwargs)
            
            except:
                # log the exception
                err = "There was an exception in  "
                err += func.__name__
                logger.exception(err)
              # re-raise the exception
            raise
        return wrapper
    return decorator

def trace_func(func):  
    ''''' 
    A decorate function to track all function invoke information with DEBUG level 
    Usage: 
    @trace_func 
    def any_function(any parametet) 
    '''  
    def tmp(*args, **kargs):  
        log.info('called: function:{}, args:{}, kargs:{}'.format(func.__name__, args, kargs))
        result = func(*args, **kargs)
        log.info('finish called: function:{}, args:{}, kargs:{}'.format(func.__name__, args, kargs))
        return result
    
    return tmp  

def tryExcept(actual_do):
    """try_except 装饰器"""
    #print('running decorate', actual_do)
    @wraps(actual_do) #把原函数的元信息拷贝到装饰器函数中 可以打印actual_do.__doc__之类的元信息
    def add_tryExcept(*args, **keyargs):
        try:
            return actual_do(*args, **keyargs)
        except:
            print ('Error execute: {}'.format(actual_do.__name__))
            traceback.print_exc()
    return add_tryExcept

def report_to_df(report):
    """sklearn report to pd.DataFrame"""
    from io import StringIO
    import re
    import pandas as pd
    report = re.sub(r" +", " ", report).replace("avg / total", "avg/total").replace("\n ", "\n")
    report_df = pd.read_csv(StringIO("Classes" + report), sep=' ', index_col=0)        
    return report_df

class NonScaler(BaseEstimator,TransformerMixin):
    """sklearn pipeline did nothing transform"""
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X ,y=None):
        return X
