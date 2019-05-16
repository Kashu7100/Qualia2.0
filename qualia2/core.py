# -*- coding: utf-8 -*- 
import os
import time
import configparser
import subprocess
from logging import getLogger, Formatter, FileHandler, StreamHandler

path = os.path.dirname(os.path.abspath(__file__))

inifile = configparser.ConfigParser()
inifile.read(path+'/config.ini', 'UTF-8')

gpu = True if inifile.get('settings', 'dtype') is 'enable' else False
dtype = inifile.get('settings', 'dtype')
level = int(inifile.get('logging', 'level'))
fmt = inifile.get('logging', 'fmt', raw=True)
datefmt = inifile.get('logging', 'datefmt', raw=True)

if not os.path.exists(path + '/logs/'):
    os.makedirs(path + '/logs/') 

logger = getLogger('QualiaLogger')
logger.setLevel(level)
filehandler = FileHandler(filename=path+'/logs/{}.log'.format(time.strftime('%Y%m%d-%H%M%S')), mode='a')
streamhandler = StreamHandler()
filehandler.setLevel(level)
streamhandler.setLevel(level)
formatter = Formatter(fmt, datefmt)
filehandler.setFormatter(formatter)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

if gpu:
    import cupy as np
    if not np.cuda.is_available():
        logger.error('[*] CUDA device is not available.')
        raise Exception
    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)

    logger.info('[*] GPU acceleration enabled.')
    logger.info('-'*71)
    nvcc = subprocess.check_output('nvcc --version', shell=True)
    for line in nvcc.split(b'\n'):
        logger.info(line.decode("utf-8"))
    logger.info('-'*71)

    def to_cpu(obj):
        return np.asnumpy(obj)

    def to_gpu(obj):
        return np.asarray(obj)
        
else:
    import numpy as np 

    def to_cpu(obj):
        return obj

    def to_gpu(obj):
        logger.error('[*] GPU acceleration is disabled.')
        raise Exception
