import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle


def preprocess(document, method = "HF_emb_FCL2NORM"):
    if method == "HF_emb_FCL2NORM":
        words = document.split(" ") # Tokenize
        words = words[:512]
        return " ".join(words)
    
    return document



def load_and_prepare_data(labels, preprocess_func = None):
    dataframes = []
    for label in labels:
        df = pd.read_csv(label, index_col='Unnamed: 0')
        dataframes.append(df)
    df_all = pd.concat(dataframes)
    df_all.reset_index(drop = True, inplace = True)
    df_all.fillna(value = "", inplace=True)
    df_all["content1"] = df_all[['title1', 'text1']].apply(lambda x: x[0] + ' ' + x[1], axis=1)
    df_all["content2"] = df_all[['title2', 'text2']].apply(lambda x: x[0] + ' ' + x[1], axis=1)
    if preprocess_func is not None:
        df_all["content1"] = df_all["content1"].apply(preprocess_func)
        df_all["content2"] = df_all["content2"].apply(preprocess_func)
    return df_all


def prepare_for_training(df, method = "HF_emb_FCL2NORM"):
    if method == "HF_emb_FCL2NORM":
        df['label'] = df['Overall'].apply(lambda x: (x - 1)/3)
        df["content"] = df[["content1", "content2"]].apply(lambda x: [x[0],x[1]], axis=1)
    if method == "HF_emb_FCReg":
        df['label'] = df['Overall'].apply(lambda x: (x - 1)/3)
        df['content1'] = df['content1'].apply(lambda x: x[:500])
        df['content2'] = df['content2'].apply(lambda x: x[:500])
        df['content'] = df[['content1', 'content2']].apply(lambda x: ' '.join([x[0], '[SEP]', x[1]]), axis=1)
        df.reset_index(drop = True, inplace = True)
    if method == "NLI":
        df['content1'] = df['content1'].apply(lambda x: x[:200])
        df['content2'] = df['content2'].apply(lambda x: x[:200])
    return df