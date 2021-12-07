import torch
import torch.nn as nn


class BERT_for_AS_cossim(nn.Module):
    def __init__(self, pretrained_model, linear_layer_size):
        super(BERT_for_AS_cossim, self).__init__()
        self.model = pretrained_model
        self.fc = nn.Linear(linear_layer_size, linear_layer_size)

    def forward(self, input_ids1, input_ids2, attention_mask1, attention_mask2):
        embs = []
        for input_ids, attention_mask in zip([input_ids1, input_ids2], [attention_mask1, attention_mask2]):
            x = self.model(input_ids, attention_mask)[0]
            x = x.mean(dim=1)
            x = self.fc(x)
            xn = torch.norm(x, p=2, dim=1, keepdim=True)
            emb = x.div(xn)
            embs.append(emb)
        similarity = 1 - torch.cosine_similarity(embs[0], embs[1])
        return similarity