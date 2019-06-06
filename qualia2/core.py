# -*- coding: utf-8 -*- 
import os
import time
import subprocess
from logging import getLogger, Formatter, FileHandler, StreamHandler

path = os.path.dirname(os.path.abspath(__file__))

gpu = True
dtype = 'float64'
level = 10
fmt = '%(asctime)s - %(name)-20s: %(levelname)-8s %(message)s'
datefmt = '%Y-%m-%d %H:%M:%S'

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
        
def cpu():
    global gpu, to_cpu, to_gpu, np
    gpu = False
    import numpy as np 

    def to_cpu(obj):
        return obj

    def to_gpu(obj):
        logger.error('[*] GPU acceleration is disabled.')
        raise Exception
    
def gpu():
    global gpu, to_cpu, to_gpu, np
    gpu = True
    import cupy as np
    if not np.cuda.is_available():
        logger.error('[*] CUDA device is not available.')
        raise Exception
    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)
    
    def to_cpu(obj):
        return np.asnumpy(obj)

    def to_gpu(obj):
        return np.asarray(obj)

def change_dtype(type):
    global dtype
    dtype = type
