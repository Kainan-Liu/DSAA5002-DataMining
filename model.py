import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import config
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from data import ProcessedData
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

class BertWrapper(nn.Module):
    '''
    leverage the power of fine-tuned large language model:
    which is SFT on the Chinese financial news for sentiment analysis
    The model's architecture is:
    BertForSequenceClassification(
            (bert): BertModel(
                (embeddings): BertEmbeddings(
                (word_embeddings): Embedding(21128, 768, padding_idx=0)
                (position_embeddings): Embedding(512, 768)
                (token_type_embeddings): Embedding(2, 768)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
                (encoder): BertEncoder(
                (layer): ModuleList(
                    (0-11): 12 x BertLayer(
                    (attention): BertAttention(
                        (self): BertSelfAttention(
                        (query): Linear(in_features=768, out_features=768, bias=True)
                        (key): Linear(in_features=768, out_features=768, bias=True)
                        (value): Linear(in_features=768, out_features=768, bias=True)
                        (dropout): Dropout(p=0.1, inplace=False)
                        )
                        (output): BertSelfOutput(
                        (dense): Linear(in_features=768, out_features=768, bias=True)
                        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                        (dropout): Dropout(p=0.1, inplace=False)
                        )
                    )
                    (intermediate): BertIntermediate(
                        (dense): Linear(in_features=768, out_features=3072, bias=True)
                        (intermediate_act_fn): GELUActivation()
                    )
                    (output): BertOutput(
                        (dense): Linear(in_features=3072, out_features=768, bias=True)
                        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                        (dropout): Dropout(p=0.1, inplace=False)
                    )
                    )
                )
                )
                (pooler): BertPooler(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (activation): Tanh()
                )
            )
            (dropout): Dropout(p=0.1, inplace=False)
            (classifier): Linear(in_features=768, out_features=3, bias=True)
        )
    '''
    def __init__(self, model_name = "hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)

    def _collate_fn(self, batch):
        inputs = self.tokenizer(batch, max_length=64, padding="max_length", truncation=True, return_tensors="pt") # tokenizer返回的是字典
        inputs = inputs.to(device = config.DEVICE)
        return inputs

    def predict(self, batch_size: int, *, dataPath: str = None, dataset: pd.DataFrame = None, neutral = False):
        print("Begin Training!")
        assert dataPath or dataset, "One of (dataPath or dataset) should not None"
        if dataPath is not None:
            dataset = ProcessedData(file_path=dataPath)
        else:
            dataset = ProcessedData(dataframe=dataset)
        
        dataloader = DataLoader(dataset=dataset,
                                batch_size=16,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=self._collate_fn)
        outputs = torch.empty(size = (0, 3) if neutral else (0, 2))
        self.model.eval()
        
        loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
        for idx, inputs in loop:
            output = self.model(**inputs)[0]
            output = output if neutral else output[:, [0, 2]]

            outputs = torch.concat((outputs, output), dim=0)

        return outputs.detach()
    
    def convert_label(self, outputs: torch.Tensor):
        dim = outputs.shape[1]
        logits = F.softmax(input=outputs, dim=1)
        if dim == 3:
            print("========> label: Negative: 0; Neutral: 1; Postive: 2")
            column_indices = torch.argmax(input=logits, dim=1)
        elif dim == 2:
            print("========> label: Negative: 0; Postive: 1")
            column_indices = torch.argmax(input=logits, dim=1)
        else:
            raise ValueError("Outputs dimension should be 2(negative/postive) or 3(negative/neutral/postive)")
        
        return column_indices.cpu().numpy()
    
    def select_by_certainty(self, outputs: torch.Tensor, threshold: float):
        num_sample = outputs.shape[0]
        dim = outputs.shape[1]

        logits = F.softmax(outputs, dim=1)
        unsure_row_indices = torch.where(torch.abs(logits[:, 0] - logits[:, 1]) < threshold)[0]
        sure_row_indices = torch.where(~torch.abs(logits[:, 0] - logits[:, 1]) < threshold)[0]
        return sure_row_indices, unsure_row_indices
    

class LogisticRegressor:
    def __init__(self) -> None:
        self.model = LogisticRegression(random_state=config.RANDOM_STATE)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        train_data = X.cpu().detach().numpy()
        train_label = y.cpu().detach().numpy()
        self.model.fit(X=train_data, y=train_label)

    def predict(self, test: torch.Tensor):
        test_data = test.cpu().detach().numpy()
        return self.model.predict(X=test)