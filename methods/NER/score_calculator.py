import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
import pandas as pd
import torch
tqdm.pandas()
import fasttext
import transformers
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForTokenClassification


def create_vocab(text_dict, vocabs):
    for key in vocabs:
        for word in text_dict[key]:
            if word not in vocabs[key]:
                vocabs[key][word] = len(vocabs[key])

class ScoreCounter:
    def __init__(self, needVocab = False, data = None, loadTransformers = False,
                 hf_model_name = "bert-base-multilingual-uncased", loadFastText = False, ft_models_path = './'):
        if needVocab:
            print("Creating vocabulary...")
            self.vocabs = {"LOC" : {}, "PER" : {}, "ORG" : {}}
            try:
                _ = data["ner1"].progress_apply(lambda x: create_vocab(x, self.vocabs))
                _ = data["ner2"].progress_apply(lambda x: create_vocab(x, self.vocabs))
                self.data = data
                print("Created")
            except Exception:
                print("Failed to create a vocabulary")
                
        if loadTransformers:
            print("Start loading transformer model")
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            self.model = AutoModel.from_pretrained(hf_model_name)
            self.device = "cuda" if torch.cuda.is_available else "cpu"
            self.model.to(self.device)
            print("Model loaded")
            
        if loadFastText:
            print("Loading fasttext models")
            langs = ['en', 'es', 'de', 'pl', 'tr', 'ar', 'fr']
            self.ft_dict = {}
            try:
                for lang in tqdm(langs):
                    if lang not in self.ft_dict:
                        self.ft_dict[lang] = fasttext.load_model(ft_models_path + 'cc.' + lang + '.300.bin')
            except Exception:
                print("Some file are missing")
            print("Loaded")
            
        
        self.vocabs_doc_freq = {}
    
    
    def tf_idf_vectorizer(self, doc, set_len, vocab, vocab_doc_freq):
        vector = np.zeros(len(vocab))
        for word in doc:
            tf = (np.array(doc) == word).sum() / len(doc)
            idf = np.log(set_len / vocab_doc_freq[word])
            vector[vocab[word]] = tf * idf + 1
        return vector
    
    def count_docs_with_words(self, texts, vocab, key):
        result = {}
        for word in tqdm(vocab):
            count_docs = np.array(list(map(lambda x: word in x[key], texts))).sum()
            result[word] = count_docs
        return result
    
    def count_vocabs_doc_freq(self):
        texts = np.hstack((self.data["ner1"].values, self.data["ner2"].values))
        for key in self.vocabs:
            self.vocabs_doc_freq[key] = self.count_docs_with_words(texts, self.vocabs[key], key)
            
    def vectores_to_scores(self, vec1, vec2, vocabs):
        max_d = max([len(vocabs[key]) for key in vocabs])
        return (cosine(vec1, vec2)) * len(vec1) / max_d
    
    def tf_idf_scores(self, doc1, doc2, key):
        if self.vocabs_doc_freq == {}:
            print("Counting document frequencies...")
            self.count_vocabs_doc_freq()
        vector1 = self.tf_idf_vectorizer(doc1, self.data.shape[0] * 2, self.vocabs[key], self.vocabs_doc_freq[key])
        vector2 = self.tf_idf_vectorizer(doc2, self.data.shape[0] * 2, self.vocabs[key], self.vocabs_doc_freq[key])
        return self.vectores_to_scores(vector1, vector2, self.vocabs)
    
    def words_bag_vectorizer(self, doc, vocab):
        vector = np.zeros(len(vocab))
        for word in doc:
            vector[vocab[word]] = (np.array(doc) == word).sum()
        return vector
    
    def BOW_vectorizer(self, doc1, doc2, key):
        vector1 = self.words_bag_vectorizer(doc1, self.vocabs[key])
        vector2 = self.words_bag_vectorizer(doc2, self.vocabs[key])
        return self.vectores_to_scores(vector1, vector2, self.vocabs)
    
    
    def transformers_scores(self, doc1, doc2):
        if doc1 == [] or doc2 == []:
            return 0.5
        tokens_info1 = self.tokenizer(doc1, padding=True, return_tensors="pt", truncation=True)
        tokens_info2 = self.tokenizer(doc2, padding=True, return_tensors="pt", truncation=True)
        text1_embedding = []
        text2_embedding = []
        text1_embedding = self.model(tokens_info1["input_ids"].to(self.device),
                 tokens_info1["attention_mask"].to(self.device))["last_hidden_state"].squeeze(0).cpu().detach().mean(dim = 1).mean(dim = 0).numpy()
        text2_embedding = self.model(tokens_info2["input_ids"].to(self.device),
                 tokens_info2["attention_mask"].to(self.device))["last_hidden_state"].squeeze(0).cpu().detach().mean(dim = 1).mean(dim = 0).numpy()
        return cosine(text1_embedding, text2_embedding)
    
    def transformers_scores_CLS(self, doc1, doc2):
        if doc1 == [] or doc2 == []:
            return 0.5
        tokens_info1 = self.tokenizer(doc1, padding=True, return_tensors="pt", truncation=True)
        tokens_info2 = self.tokenizer(doc2, padding=True, return_tensors="pt", truncation=True)
        text1_embedding = []
        text2_embedding = []
        text1_embedding = self.model(tokens_info1["input_ids"].to(device),
                 tokens_info1["attention_mask"].to(device))["last_hidden_state"][:,0].cpu().detach().mean(dim = 0).numpy()
        text2_embedding = self.model(tokens_info2["input_ids"].to(device),
                 tokens_info2["attention_mask"].to(device))["last_hidden_state"][:,0].cpu().detach().mean(dim = 0).numpy()
        return cosine(text1_embedding, text2_embedding)
    
    
    def fasttext_scores(self, doc1, doc2, lang1, lang2):
        doc1_a = list(map(lambda x: x.split(" "), doc1))
        doc2_a = list(map(lambda x: x.split(" "), doc2))
        summ = np.zeros(300)
        for ne in doc1_a:
            emb1 = np.mean([self.ft_dict[lang1].get_word_vector(word) for word in ne], axis = 0)
            summ +=  emb1
        emb1 = summ / len(doc1_a)

        summ = np.zeros(300)
        for ne in doc2_a:
            emb2 = np.mean([self.ft_dict[lang2].get_word_vector(word) for word in ne], axis = 0)
            summ +=  emb2
        emb2 = summ / len(doc2_a)   
        return cosine(emb1, emb2)
    
    def word_overlap(self, doc1, doc2):
        s1 = set(doc1)
        s2 = set(doc2)
        overlap_count = len(s1.intersection(s2))
        return overlap_count