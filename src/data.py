import os
import json
import pandas as pd
import numpy as np
import operator
from tqdm import tqdm
from torch.utils.data import Dataset


class AShareData:
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir

        if os.path.exists(data_dir):
            self.A_share_data = json.load(open(file=self.data_dir, mode="r", encoding="utf-8"))
            self.A_share_df = pd.read_json(self.data_dir)
        else:
            raise FileNotFoundError(f"{self.data_dir} not exists")

    @property
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

    @property
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
        self.grid_search_matrix = np.ones((len(self.texts), len(name_list)), dtype=bool)
        for i, text in tqdm(enumerate(self.texts), leave=False, total=len(self.texts)):
            for j, name in enumerate(name_list):
                self.grid_search_matrix[i, j] = operator.contains(text, name)
        
        row_indices, column_indices = np.where(self.grid_search_matrix) # get the remain row indices and column indices
        self.row_indices = row_indices
        self.column_indices = column_indices
        other_row_indices = np.where(~np.any(self.grid_search_matrix, axis=1))[0]
        clean_data = self.data.iloc[row_indices, :]
        noise_data = self.data.iloc[other_row_indices, :]
        return clean_data, noise_data
    
    def Explicit_Company_list(self, name_list: list):
        name_dict = dict(zip(range(len(name_list)), name_list))
        indices = {}
        for key, value in zip(self.row_indices, self.column_indices):
            indices.setdefault(key, []).append(value)
        
        company_dict = {}
        for key, value in indices.items():
            company_dict[key] = [name_dict[k] for k in value]
        
        self.data["Explicit_Company"] = pd.Series(company_dict)
        
        return company_dict


class ProcessedData(Dataset):
    def __init__(self, *, dataframe = None, file_path: str = None):
        super().__init__()
        self.file_path = file_path

        if file_path is not None:
            if os.path.exists(file_path):
                self.data = pd.read_csv(file_path)
            else:
                raise FileNotFoundError("File Not exists! Please process first")
        else:
            if dataframe is not None:
                self.data = dataframe
            else:
                raise NotImplementedError("please assign file_path or dataframe first")
        
    @property
    def getNewsID(self):
        return self.data.loc[:, "NewsID"]

    @property
    def getNewsContent(self):
        return self.data.loc[:, "NewsContent"]

    @property
    def getExplicit_Company(self):
        return self.data.loc[:, "Explicit_Company"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index, -1] # -1: text consists of newscontent and title, which insert in the last column
        return text

class retrainData(Dataset):
    def __init__(self, *, dataframe = None):
        super().__init__()
        if dataframe is not None:
            self.data = dataframe
        else:
            raise NotImplementedError("please assign dataframe first")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data.loc[index, "NewsContent"]
        label = self.data.loc[index, "label"]
        return text, label

