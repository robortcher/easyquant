# -*- coding: utf-8 -*-
# --- author: ---

from __future__ import unicode_literals, absolute_import

import logging
import logging.config
import logging.handlers
from datetime import datetime
# from commons.settings import LogConfig
import os, sys
import json
import time



root_dir = 'D:\database' + '/' + '.singletrader'
log_file = root_dir + '/' + 'logs'


def check_and_mkdir(path):
    # logger.info("开始初始化路径...")
    if not os.path.exists(path):
        os.makedirs(path)
        # logger.info(f"开始创建路径{path}")
check_and_mkdir(log_file)
log_path = log_file+'/'+'singletrader.log'

# class _InfoFilter(logging.Filter):
#     def filter(self, record):
#         if logging.INFO <= record.levelno <= logging.ERROR:
#             return super().filter(record)
#         else:
#             return 0


def _get_filename(*, basename='.log', log_level='info'):
    date_str = datetime.today().strftime('%Y%m%d')
    pidstr = str(os.getpid())
    # parent_path = os.path.dirname(os.path.abspath('__file__'))
    # LOG_PATH = os.path.join(parent_path, 'logs')
    return ''.join((log_path,
        date_str, '-', pidstr, '-', log_level, '-', basename,))



# def _get_filename2(*, suffix='.log', log_level='info'):
#     file = getattr(sys.modules['__main__'], '__file__', None)
#     date_str = datetime.today().strftime('%Y%m%d')
#     basepath = LogConfig.PATH
#     return ''.join((
#         basepath, date_str, '-', os.path.splitext(os.path.basename(file))[0], suffix, ))


class LogFactory:
    _SINGLE_FILE_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 每个日志文件，使用 2GB
    _BACKUP_COUNT = 10  # 轮转数量是 10 个

    _LOG_CONFIG_DICT = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            # 开发环境
            'dev': {
                'class': 'logging.Formatter',
                'format': ('{"level": "%(levelname)s", '
                           '"time": "%(asctime)s", '
                           f'"timeMillis": {int(time.time()*1000)},'
                           '"module": "%(name)s", '
                           '"method": "python-service", '
                           '"addition": "[%(filename)s %(lineno)s %(funcName)s]", '
                           '"message": "%(message)s"}')
            },
        },

        # 'filters': {
        #     'info_filter': {
        #         '()': _InfoFilter,
        #     }
        # },

        'handlers': {
            'console': {
                'formatter': 'dev',
                'class': 'logging.StreamHandler',
            },

            'file': {
                'formatter': 'dev',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': log_path,#get_filename(log_level='info'),
                'maxBytes': _SINGLE_FILE_MAX_BYTES,
                'encoding': 'UTF-8',
                'backupCount': _BACKUP_COUNT,
                'delay': True,
                # 'filters': ['info_filter', ]
            },

            # 'file_error': {
            #     'level': 'ERROR',
            #     'formatter': 'dev',
            #     'class': 'logging.handlers.RotatingFileHandler',
            #     'filename': _get_filename2(log_level='error'),
            #     'maxBytes': _SINGLE_FILE_MAX_BYTES,
            #     'encoding': 'UTF-8',
            #     'backupCount': _BACKUP_COUNT,
            #     'delay': True,
            # },
        },

        'loggers': {
            '': {  # root logger
                'level': 'INFO',
                'handlers': ['console', 'file', ],
            },
            # 'proxy_trading': {  # for server
            #     'level': 'DEBUG',
            #     'handlers': ['console', 'file', ],
            #     'propagate': False
            # }
        },
    }

    logging.config.dictConfig(_LOG_CONFIG_DICT)

    @classmethod
    def get_logger(cls, logger_name="singletrader-service"):
        return logging.getLogger(logger_name)

logger = LogFactory.get_logger()
# class Logger():
#     def info(self, msg):
#         print(msg)
#     def error(self, msg):
#         print(msg)
# logger = Logger()
if __name__ == '__main__':
    logging.info('error')