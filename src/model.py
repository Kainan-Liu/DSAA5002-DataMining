import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import config
import numpy as np
from utils import seed_everything
from transformers import BertTokenizer, BertForSequenceClassification
from data import ProcessedData, retrainData
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        with torch.no_grad():
            for idx, inputs in loop:
                inputs = inputs.to(device = config.DEVICE)
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
    

class LogisticRegressor(nn.Module):
    def __init__(self, num_labels: int = 1, model_name = "../checkpoints") -> None:
        super().__init__()
        self.num_labels = num_labels
        self.model = nn.Sequential(
            nn.Embedding(num_embeddings=21128, embedding_dim=768),
            nn.Linear(in_features=768, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=16),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=num_labels),
            nn.Sigmoid()
        )
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def _collate_fn(self, batch):
        texts, labels = [], []
        for item in batch:
            texts.append(item[0])
            labels.append(item[0])
        inputs = self.tokenizer(texts, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        return inputs, labels

    def forward(self, x):
        return self.model(x)
    
    def train(self, dataframe: str, epochs: int, batch_size: int):
        print("====>Begin Training!")
        # initialization
        self.model.to(device=config.DEVICE)
        dataset = retrainData(dataframe=dataframe)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=self._collate_fn)
        optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.1)
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
            for idx, inputs, label in loop:
            # 1. data
                inputs = inputs.to(device=config.DEVICE)
                label = label.to(device=config.DEVICE)
                data = inputs.input_ids

            for layer in self.model.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

            seed_everything(42)

            loss_all = []
            # 3.1 forward
            output = self.model(data)
            loss = criterion(output, label)

            # 3.2 backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

            # 3.3 early stopping
            loss_all.append(loss.item())
        print("====>Stop Training")

    def predict(self, test_dataframe: pd.DataFrame, batch_size = 64):
        self.model.eval()
        dataset = retrainData(dataframe=test_dataframe)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=self._collate_fn)
        labels = []
        with torch.no_grad():
            for idx, data, _ in enumerate(dataloader):
                output = self.model(data)
                label = (output > 0.5).long()
                labels.append(output.items())
            
        return labels
