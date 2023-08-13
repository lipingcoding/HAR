import numpy as np
import torch
import sys


class Logger(object):
    """
    这个类的目的是尽可能不改变原始代码的情况下, 使得程序的输出同时打印在控制台和保存在文件中
    用法: 只需在程序中加入一行 `sys.stdout = Logger(log_file_path)` 即可
    """
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def set_seed(seed):
    import numpy as np
    import random
    import torch
    import dgl
    
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-gpu
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    dgl.seed(seed)

def mkdir_if_not_exist(file_name):
    import os
    import shutil

    dir_name = os.path.dirname(file_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def print_total_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total params: {total_params}, trainable_total_params: {trainable_total_params}')



def nowdt():
    """
    get string representation of date and time of now()
    """
    from datetime import datetime

    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")