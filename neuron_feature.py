import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import py_vncorenlp
import os

class NeuronFeature:
    # Tạo và sinh đặc trưng
    def __init__(self) -> None:
        print('init')
        py_vncorenlp.download_model(save_dir='models/vncorenlp')
        self.rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=os.path.join(os.path.dirname(__file__),'models/vncorenlp'))
        os.chdir(os.path.dirname(__file__))
        
    def segment_word(self, text:str):
        senteces = self.rdrsegmenter.word_segment(text)
        return ' '.join(senteces)
    
    def get_embedding_from_text(self, text, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='models')
        model = AutoModel.from_pretrained(model_name, cache_dir='models')
        with torch.no_grad():
            input_ids = tokenizer(text, return_tensors='pt')['input_ids']
            outputs = model(input_ids)
            word_embeddings = outputs.last_hidden_state[0].numpy()
            text_embedding = outputs.last_hidden_state.mean(dim=1)[0].numpy()
        return pd.Series([word_embeddings, text_embedding])  
        
    def get_featured(self, path_file_csv:str='datasets/datasets.csv', feature_name:str='vinai/bartpho-word', path_vector_save:str=None):
        df = pd.read_csv(path_file_csv)
        df['text'] = df['text'].apply(self.segment_word)
        df[['word_vector', 'sentence_vector']] = df['text'].apply(lambda x: self.get_embedding_from_text(x, feature_name))
        # if feature_name == 'BARTpho':
        #     df['text'] = df['text'].apply(self.segment_word)
        #     tokenizer = AutoTokenizer.from_pretrained('models/bartpho-word')
        #     model = AutoModel.from_pretrained('models/bartpho-word')            
        #     def get_embedding_from_bart(text):
        #         with torch.no_grad():
        #             input_ids = tokenizer(text, return_tensors='pt')['input_ids']
        #             outputs = model(input_ids)
        #             word_embeddings = outputs.last_hidden_state[0].numpy()
        #             text_embedding = outputs.last_hidden_state.mean(dim=1)[0].numpy()
        #         return pd.Series([word_embeddings, text_embedding])
        #     df[['word_vector', 'sentence_vector']] = df['text'].apply(get_embedding_from_bart)
            
        # else:
        #     df['text'] = df['text'].apply(self.segment_word)
        #     tokenizer = AutoTokenizer.from_pretrained('models/phobert-base')
        #     model = AutoModel.from_pretrained('models/phobert-base')            
        #     def get_embedding_from_bart(text):
        #         with torch.no_grad():
        #             input_ids = tokenizer(text, return_tensors='pt')['input_ids']
        #             outputs = model(input_ids)
        #             word_embeddings = outputs.last_hidden_state[0].numpy()
        #             text_embedding = outputs.last_hidden_state.mean(dim=1)[0].numpy()
        #         return pd.Series([word_embeddings, text_embedding])
        #     df[['word_vector', 'sentence_vector']] = df['text'].apply(get_embedding_from_bart)
        
        if path_vector_save is None:
            path_vector_save = f"{path_file_csv.split('.')[0]}_{feature_name}.pkl"
        df.to_pickle(path_vector_save)
        return df

if __name__ == "__main__":
    neuronfeature = NeuronFeature()
    neuronfeature.get_featured()