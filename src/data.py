import os
import json
import pandas as pd
import numpy as np
import operator
from tqdm import tqdm
from model import BertWrapper


class AShareData:
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir

        if os.path.exists(data_dir):
            self.A_share_data = json.load(open(file=self.data_dir, mode="r"))
            self.A_share_df = pd.read_json(self.data_dir)
        else:
            raise FileNotFoundError(f"{self.data_dir} not exists")

    def getNameList(self):
        '''
        get dictionary of Each A-share company's short name  
        '''
        return self.A_share_df["name"].to_list()
    
    def setInfoDF(self):
        '''
        drop location and time, only maintain name, code, full_name
        '''
        other_features = ["location", "time"]
        self._Info_df = self.A_share_df.drop(other_features)

    def getInfoDF(self):
        if type(self._Info_df):
            return self._Info_df
        else:
            raise ValueError("Info_df not exists, please run setInfoDF first")


class NewsData:
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir

        if os.path.exists(data_dir):
            self.data = pd.read_excel(data_dir)
            self.text()
        else:
            raise FileNotFoundError(f"{self.data_dir} not exists")
        
    def text(self, features=["Title", "NewsContent"]):
        '''
        We can combine the title and newsContent to create a new feature
        And get a list of values
        '''
        self.data.loc[:, 'text'] = self.data.loc[:, features].apply(lambda row: '.'.join(row.values.astype(str)), axis=1)
        self.texts = list(self.data["text"].values)
        return self.texts
    
    def force_denoise(self, name_list: list):
        '''
         A brute force search
        '''
        grid_search_matrix = np.ones((len(self.texts), len(name_list)), dtype=bool)
        for i, text in enumerate(self.texts):
            loop = tqdm(enumerate(name_list), leave=False, total=len(self.texts))
            for j, name in loop:
                grid_search_matrix[i, j] = operator.contains(text, name)
        
        row_indices = np.where(np.any(grid_search_matrix, axis=1))[0] # get the remain row indices
        other_row_indices = np.where(~np.any(grid_search_matrix, axis=1))[0]
        clean_data = self.data.iloc[row_indices, :]
        noise_data = self.data.iloc[other_row_indices, :]
        return clean_data, noise_data
    

class Cleaner:
    def __init__(self, noise_df: pd.DataFrame) -> None:
        self.noise_data = noise_df

    def getEmbedding(self, text: list = None, model_name: str = "bert-base-chinese"):
        '''
        :text: list of text to be processed
        :params: model_name = "bert-base-chinese", pretrained model available on huggingface Hub https://huggingface.co/bert-base-chinese
        '''
        if text is None:
            text = list(self.noise_data["text"].values)
        else:
            assert isinstance(text, list)
        bert = BertWrapper(model_name=model_name)
        hidden_state = bert(text)
        return hidden_state