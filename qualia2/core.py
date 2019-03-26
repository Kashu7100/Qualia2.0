# -*- coding: utf-8 -*- 
import os
import time
from .config import *
import subprocess
from logging import getLogger, Formatter, FileHandler, StreamHandler

path = os.path.dirname(os.path.abspath(__file__))

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
    np.add.at = np.scatter_add

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
