# -*- coding: utf-8 -*- 
import os
from .config import *

if gpu:
    import cupy as np
    if not np.cuda.is_available():
        raise Exception('[*] CUDA device is not available.')
    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)
    np.add.at = np.scatter_add

    print('[*] GPU acceleration enabled.')
    print('-'*71)
    os.system('nvcc --version')
    print('-'*71)
    
    def to_cpu(obj):
        return np.asnumpy(obj)
else:
    import numpy as np 
