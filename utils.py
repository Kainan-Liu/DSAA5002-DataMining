# -*- coding: UTF-8 -*-
'''
@Project : DATA-MINING-PROJECT
@File    : utils.py
@Author  : kliu
@Date    : 2023/11/05
'''

import pandas as pd
import math
import torch
import torch.nn as nn
import random
import numpy as np
from model import BertWrapper
from data import ProcessedData


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

def Task1_output(dataset: ProcessedData, output_label: np.array):
    num_sample = len(dataset.data)
    df = dataset.data
    df["label"] = pd.DataFrame(output_label)
    drop_features = ["NewsSource", "text", "Title"]
    return df.drop(labels=drop_features)


class process:
    def __init__(self) -> None:
        pass

    def getEmbedding(self, text: list = None, model_name: str = "bert-base-chinese"):
        '''
        :text: list of text to be processed
        :params: model_name = "bert-base-chinese", pretrained model available on huggingface Hub https://huggingface.co/bert-base-chinese
        '''
        bert = BertWrapper(model_name=model_name)
        hidden_states = bert(text)
        return hidden_states
    
    def cosine_similarity(self, q: torch.Tensor, k: torch.Tensor):
        '''
        :params q: refers to the hidden_states of news text--> num_news x d_model
        :params k: refers to the hidden_states of company name--> d_model x num_company
        actually we compute the cosine similarity here and normalize by the d_model dimension
        returns: a probability matrix where Mij represents the similarity between i news and j company
        '''
        d_model = q.shape[-1]
        score = torch.mm(q, k.T) / math.sqrt(d_model)
        similarity_weight_matrix = nn.Softmax(score)
        return similarity_weight_matrix
    
    def filter(self, matrix, threshold):
        indices = np.where(np.any(matrix > threshold, axis=1))
        row_indices = indices[0]
        column_indices = indices[1]
        return row_indices, column_indices
