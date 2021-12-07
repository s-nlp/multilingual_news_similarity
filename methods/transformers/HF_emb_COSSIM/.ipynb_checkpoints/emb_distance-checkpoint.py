import transformers
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd




def emb_distance(model_name, df, device = "cpu"):
    dfc = df.copy()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    tokens_info1 = tokenizer(list(dfc["content1"].values), padding=True, return_tensors="pt", truncation=True)
    tokens_info2 = tokenizer(list(dfc["content2"].values), padding=True, return_tensors="pt", truncation=True)

    text1_embedding = []
    text2_embedding = []
    for i in range(tokens_info1["input_ids"].shape[0]):
        if i % 500 == 0:
            print(f"{i} iterations are behind")
        text1_embedding.append(model(tokens_info1["input_ids"][i].unsqueeze(0).to(device), tokens_info1["attention_mask"][i].unsqueeze(0).to(device))["last_hidden_state"].squeeze(0).cpu().detach().tolist())
        text2_embedding.append(model(tokens_info2["input_ids"][i].unsqueeze(0).to(device), tokens_info2["attention_mask"][i].unsqueeze(0).to(device))["last_hidden_state"].squeeze(0).cpu().detach().tolist())
    text1_embedding = torch.tensor(np.mean(text1_embedding, axis=1))
    text2_embedding = torch.tensor(np.mean(text2_embedding, axis=1))
    dfc["prediction"] = 1 - torch.cosine_similarity(text1_embedding, text2_embedding).numpy()
    return {"correlation" : dfc["Overall"].corr(dfc["prediction"], method='pearson'), "similarity": dfc["prediction"].values }