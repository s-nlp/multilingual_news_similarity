import torch
import torch.nn as nn

class BERT_for_AS(nn.Module):
    def __init__(self, pretrained_model, linear_layer_size):
        super(BERT_for_AS, self).__init__()
        self.model = pretrained_model
        self.fc = nn.Linear(linear_layer_size, 2)

    def forward(self, input_ids, attention_mask):
        #we have to take only [CLS], it's first, from the output layer
        x = self.model(input_ids, attention_mask)['last_hidden_state'][:,0]
        logits = self.fc(x)
        return logits