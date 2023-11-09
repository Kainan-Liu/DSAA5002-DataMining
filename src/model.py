import torch
import torch.nn as nn
import config
from transformers import BertTokenizer, BertModel


class BertWrapper(nn.Module):
    def __init__(self, model_name = "bert-base-chinese"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        
    def forward(self, text_input, batch_size: int = 50):
        self.model = self.model.to(device=config.DEVICE)
        self.model.eval()
        last_hidden_states = torch.empty((0, 768), device=config.DEVICE)
        for batch_id in range(len(text_input) // batch_size - 1):
            inputs = self.tokenizer(text_input[batch_id * batch_size:(batch_id + 1) * batch_size], padding="max_length", max_length=30, truncation=True, return_tensors="pt") # not sure to squeeze(0) or not
            inputs = inputs.to(device=config.DEVICE)
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state[:, 0]
            last_hidden_states = torch.concatenate((last_hidden_states, last_hidden_state), dim=0)
        return last_hidden_states.detach()