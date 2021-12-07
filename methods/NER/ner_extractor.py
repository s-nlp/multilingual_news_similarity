import transformers
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
from transformers import AutoModelForTokenClassification
from polyglot.text import Text
import spacy
import numpy as np

from tqdm import tqdm,trange


class NerExtractor:
    def __init__(self, method, hf_model_name = "Davlan/bert-base-multilingual-cased-ner-hrl", spacy_model_name = "xx_ent_wiki_sm"):
        self.method = method
        if self.method == "Huggingface":
            tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            model = AutoModelForTokenClassification.from_pretrained(hf_model_name)
            self.nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=0)
        if self.method == "Spacy":
            self.nlp = spacy.load(spacy_model_name)
        
    
    def extract3ner(self, text, text_lang = "null"):
        result = {}
        for key in ["LOC", "PER", "ORG"]:
            if self.method == "Huggingface":
                ner_results = self.nlp(text, grouped_entities=True)
                result[key] = list(map(lambda x: x['word'] if x['entity_group'] == key else '', ner_results))
                result[key] = list(filter(lambda x: x != "", result[key]))  
            if self.method == "Polyglot":
                try:
                    ner_results = Text(text, hint_language_code = text_lang)
                except Exception:
                    ner_results = Text(text)
                result[key] = []
                try:
                    for word in ner_results.entities:
                        if word.tag[-3:] == key:
                            result[key].append(" ".join(word))
                except UnboundLocalError:
                    print(f"Text is skipped for {key}")
                    continue

            if self.method == "Spacy":
                doc = self.nlp(text)
                words = [[Y, X.ent_type_] for Y in doc.ents for X in Y]
                result[key] = []
                for word in words:
                    if word[1] == key:
                        result[key].append(str(word[0]))
        return result
    
    
    def extract1ner(text, text_lang = "null"):
        if mself.method == "Huggingface":
            ner_results = self.nlp(text, grouped_entities=True)
            result = list(map(lambda x: x['word'], ner_results)) 

        if self.method == "Polyglot":
            try:
                ner_results = Text(text, hint_language_code = text_lang)
            except Exception:
                ner_results = Text(text)
            result = []
            try:
                for word in ner_results.entities:
                    result.append(" ".join(word))
            except UnboundLocalError:
                print(f"Text is skipped for {key}")

        if self.method == "Spacy":
            doc = self.nlp(text)
            words = [[Y, X.ent_type_] for Y in doc.ents for X in Y]
            result = []
            for word in words:
                result.append(str(word[0]))

        return result