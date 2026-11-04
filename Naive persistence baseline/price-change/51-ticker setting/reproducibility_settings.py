# reproducibility_settings.py
import os
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42, framework: str = 'torch') -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
