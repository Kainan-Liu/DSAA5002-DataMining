# -*- coding: UTF-8 -*-
'''
@Project : DATA-MINING-PROJECT
@File    : utils.py
@Author  : kliu
@Date    : 2023/11/05
'''

import pandas as pd
import torch
import torch.nn as nn
import random
import numpy as np


def save_checkpoints(model: nn.Module, optimizer: nn.Module, pth: str):
    #print("==> Saving Checkpoints")
    checkpoints = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoints, pth)


def load_checkpoints(pth: str):
    #print("==> Loading Checkpoints")
    checkpoints = torch.load(pth)
    return checkpoints


def seed_everything(random_state: int):
    """
    Make the results be reproducible
    :param random_state:用作种子
    :return: None
    """
    np.random.seed(random_state)
    random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
