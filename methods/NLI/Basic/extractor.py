import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np

class UniquePairsDataset(Dataset):
    def __init__(self, df, tokenizer, columns=['title1', 'title2']):
        self.tokenizer = tokenizer
        self.df = df
        self.columns = columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        text = self.df.iloc[index][self.columns].values.tolist()
        text[0] = 'News "'+ text[0] + '" is legit.'
        text[1] = 'This example is ' + text[1] + '.'
        tokenized_input_seq_pair = self.tokenizer.encode_plus(text[0], 
                                                         text[1],
                                                         return_token_type_ids=True, 
                                                         padding='longest',
                                                         truncation_strategy='only_first',
                                                         truncation=True)

        ids = tokenized_input_seq_pair['input_ids']
        mask = tokenized_input_seq_pair['attention_mask']
        token_type_ids = tokenized_input_seq_pair["token_type_ids"]

        return {
            'ids': torch.Tensor(ids).long().unsqueeze(0),
            'mask': torch.Tensor(mask).long().unsqueeze(0),
            'token_type_ids': torch.Tensor(token_type_ids).long().unsqueeze(0),
        }
    
    
def extract_nli(loader, model, device): # TODO: rename function
    model.eval()
    probs = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(loader)):
            ids = data['ids'][0].to(device, dtype = torch.long)
            mask = data['mask'][0].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'][0].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids, labels=None)[0]
            predicted_probability = torch.softmax(outputs[0], dim=0).tolist()
            probs.append(predicted_probability)

    return np.array(probs)


def nli_extractor(dataset, model, tokenizer, params, device):  
    
    df_all = dataset.copy()
    
    news_set = UniquePairsDataset(df_all, tokenizer, columns=['content1', 'content2'])
    news_loader = DataLoader(news_set, **params)
    
    # extracting nli features
    results = extract_nli(news_loader, model, device)
        
    df_all['entailment'] = results[:, 0]
    df_all['neutral'] = results[:, 1]
    df_all['contradiction'] = results[:, 2]
    
    return df_all