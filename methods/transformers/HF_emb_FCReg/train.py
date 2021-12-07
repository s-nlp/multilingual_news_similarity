import pandas as pd
import torch
import matplotlib.pyplot as plt

import transformers
from transformers import AutoTokenizer, AutoModel, pipeline
from transformers import  AdamW, BertModel
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
from torch.nn import MSELoss
from sklearn.model_selection import train_test_split

from tqdm import trange
from IPython.display import clear_output
import numpy as np
from tqdm import tqdm
import pickle
import gc

from methods.transformers.HF_emb_FCReg.model import BERT_for_AS


class STLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_mul, ratio, steps_per_cycle, decay=1, last_epoch=-1):
        self.max_mul = max_mul - 1
        self.turning_point = steps_per_cycle // (ratio + 1)
        self.steps_per_cycle = steps_per_cycle
        self.decay = decay
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        residual = self.last_epoch % self.steps_per_cycle
        multiplier = self.decay ** (self.last_epoch // self.steps_per_cycle)
        if residual <= self.turning_point:
            multiplier *= self.max_mul * (residual / self.turning_point)
        else:
            multiplier *= self.max_mul * (
                (self.steps_per_cycle - residual) /
                (self.steps_per_cycle - self.turning_point))
        return [lr * (1 + multiplier) for lr in self.base_lrs]
    
    
    
def train_eval(model_name, data, batch_size, batch_size_val, linear_layer_size, num_epochs, result_dict,
               train = True, checkpoints_path = "./checkpoints", figs_path = "./figs"):
    tr_x, val_x, tr_y, val_y, tr_i, val_i = train_test_split(data['content'].values, data['label'].values, data.index.values,
                                random_state=42, test_size=0.25, shuffle=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    if train:
        train_labels = torch.tensor(tr_y).float()
        val_labels = torch.tensor(val_y).float()


        tokens_info = tokenizer(list(tr_x), padding=True, truncation=True, return_tensors="pt")
        train_data = TensorDataset(tokens_info['input_ids'], tokens_info['attention_mask'], train_labels)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        tokens_info = tokenizer(list(val_x), padding=True, truncation=True, return_tensors="pt")
        valid_data = TensorDataset(tokens_info['input_ids'], tokens_info['attention_mask'], val_labels)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size_val)

        embedder =  BERT_for_AS(model, linear_layer_size)
        optimizer = AdamW(embedder.parameters(), lr=1e-5)
        scheduler = STLR(optimizer, 2, 32, 500, 0.38)
        criterion = MSELoss()

        torch.cuda.empty_cache()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embedder.to(device)
        max_grad_norm = 1.0
        loss_values, validation_loss_values = [], []
        prev_metric = 0
        best_r, best_p = 0, 0
        print(f"Start training {model_name}")
        try:
            for ep in trange(num_epochs, desc="Epoch"):

                embedder.train()
                total_loss = 0

                for step, batch in enumerate(train_dataloader):

                    batch = tuple(t.to(device) for t in batch)
            
                    input_ids, attention_mask, labels  = batch

                    optimizer.zero_grad()
                    logits = embedder(input_ids, attention_mask)
                    prediction = torch.sigmoid(logits).T[0]
                    loss = criterion(prediction, labels)
                    loss.backward()
                    total_loss += loss.item()

                    # Clipping the norm of the gradient to help prevent the "exploding gradients" 
                    torch.nn.utils.clip_grad_norm_(parameters=embedder.parameters(), max_norm=max_grad_norm)

                    optimizer.step()
                    scheduler.step()

                avg_train_loss = total_loss / len(train_dataloader)
                clear_output(wait=True)
                print("\nAverage train loss: {}".format(avg_train_loss))

                loss_values.append(avg_train_loss)


                embedder.eval()
                eval_loss, eval_accuracy = 0, 0
                all_predictions = []
                all_indices = []
                all_labels = []
                for batch in valid_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, attention_mask, labels = batch

                    with torch.no_grad():
                        logits = embedder(input_ids, attention_mask)

                    prediction = torch.sigmoid(logits).T[0]
                    eval_loss += criterion(prediction, labels)

                    pred = prediction.detach().cpu().numpy()
                    all_predictions = all_predictions + list(pred)

                eval_loss = eval_loss / len(valid_dataloader)

                validation_loss_values.append(eval_loss)
                print("Validation loss: {}".format(eval_loss))

                df_val = data.loc[val_i]
                df_val['prediction'] = all_predictions
                corr = df_val['Overall'].corr(df_val['prediction'], method='pearson')
                print(f"Pearson correlation for epoch {ep}: {corr}")

                if np.abs(corr) > prev_metric:
                    torch.save(embedder.state_dict(), f"{checkpoints_path}/model_best_{model_name}_{num_epochs}.pth")
                    print("Best model saved at epoch {}".format(ep))
                    prev_metric = np.abs(corr)

                print('Best correlation: ', prev_metric)
                torch.save(embedder.state_dict(), f"{checkpoints_path}/model_curr_{model_name}_{num_epochs}.pth")

                plt.figure(figsize=(10,8))
                plt.title(f"Train: {model_name}; Best MSE: {prev_metric}")
                plt.plot(loss_values, label='Train loss')
                plt.plot(validation_loss_values, label='Validation loss')
                plt.xlabel('#epoch')
                plt.ylabel('Loss value')
                plt.grid()
                plt.legend()
                plt.savefig(f"{figs_path}/train_plot_{model_name}.png")
                plt.show()
        except KeyboardInterrupt:
            pass
    
    
    print(f"Start evaluation for {model_name}")
    torch.cuda.empty_cache()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedder = BERT_for_AS(model, linear_layer_size)
    embedder.to(device)
    embedder.load_state_dict(torch.load(f'{checkpoints_path}/model_best_{model_name}_{num_epochs}.pth'))
    criterion = MSELoss()

    x = list(val_x)
    y = list(val_y)
    # val_labels = torch.tensor([[1.,0.] if el == 1 else [0.,1.] for el in y])
    val_labels = torch.tensor(y)
    tokens_info = tokenizer(x, padding=True, truncation=True, return_tensors="pt")
    valid_data = TensorDataset(tokens_info['input_ids'], tokens_info['attention_mask'], val_labels)

    valid_dataloader = DataLoader(valid_data, batch_size=batch_size_val, shuffle=False)



    embedder.eval()
    eval_loss, eval_accuracy = 0, 0
    all_predictions = []
    all_indices = []
    all_labels = []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch

        with torch.no_grad():
            logits = embedder(input_ids, attention_mask)

        prediction = torch.sigmoid(logits).T[0]
        eval_loss += criterion(prediction, labels)
        pred = prediction.detach().cpu().numpy()
        all_predictions = all_predictions + list(pred)


    eval_loss = eval_loss / len(valid_dataloader)


    print("Validation loss: {}".format(eval_loss))
    df_val = data.loc[val_i]
    df_val['prediction'] = all_predictions
    corr = df_val['Overall'].corr(df_val['prediction'], method='pearson')
    result_dict[model_name] = corr
    with open("./results/result_HF_emb_FCReg_2parts.pickle", "wb") as f:
        pickle.dump(result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Pearson correlation for {model_name}: {corr}")
    del embedder
    gc.collect()
    torch.cuda.empty_cache()