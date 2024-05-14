import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
# import py_vncorenlp
import os
import json
import numpy as np
from utils import visual_embedding, array2string
from tqdm import tqdm

class NeuronFeatures:
    """## Thực hiện trích xuất các đặc trưng sử dụng mạng Nơ-ron học sâu
    """
    def __init__(self) -> None:
        # py_vncorenlp.download_model(save_dir='models/vncorenlp')
        # self.rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=os.path.join(os.path.dirname(__file__),'models/vncorenlp'))
        os.chdir(os.path.dirname(__file__))
        
    def segment_word(self, text:str):
        """## Tokenizer câu thành các từ

        ### Args:
            - `text (str)`: Đoạn text cần token

        ### Returns:
            - `text`: Đoạn text sau khi token
        """
        # senteces = self.rdrsegmenter.word_segment(text)
        # return ' '.join(senteces)
        return text
    
    def get_embedding_from_text(self, text, tokenizer, model, number_words_per_sample_logs):
        """## Trích xuất đặc trưng 

        ### Args:
            - `text (_type_)`: Văn bản cần trích xuất đặc trưng
            - `tokenizer (_type_)`: Tokenzier sử dụng
            - `model (_type_)`: Mô hình sử dụng

        ### Returns:
            - `[word_embeddings, text_embedding]`: Trả về đồng thời 2 loại đặc trưng [Đặc trưng biểu diễn các từ, Đặc trưng biểu diễn văn bản]
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
            tokens = np.array(tokenizer.convert_ids_to_tokens(input_ids[0]))[:number_words_per_sample_logs]
            outputs = model(input_ids)
            word_embeddings = outputs.last_hidden_state[0].cpu().numpy()[:number_words_per_sample_logs]
            text_embedding = outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy()
        return pd.Series([word_embeddings, text_embedding, tokens])  
        
    def get_features(self, path_file_csv:str='datasets/train.csv', feature_name:str='vinai/bartpho-word', path_vector_save:str=None,
                      number_samples_logs:int=2, number_words_per_sample_logs:int=5):
        """## Trích xuất đặc trưng văn bản sử dụng mạng nơ-ron

        ### Args:
            - `path_file_csv (str, optional)`: Đường dẫn tới kho dữ liệu cần trích xuất. Defaults to 'datasets/datasets.csv'.
            - `feature_name (str, optional)`: mô hình sử dụng để trich xuất. 
            Ở đây có thể là các mô hình thuộc kiến trúc Bart  như 'vinai/bartpho-word', Roberta như 'vinai/phobert-base'. Defaults to 'vinai/bartpho-word'.
            - `path_vector_save (str, optional)`: Đường dẫn file lưu trữ véc-tơ biểu diễn của văn bản. 
            Không chỉ định sẽ được tạo tự động với với tên được tạo từ (path_file_csv, feature_name) Defaults to None. Defaults to None.

        ### Returns:
            - ``df`: Một dataframe có 3 cột (text, word_vector, sentence_vector) tương ứng với: văn bản, đặc trưng biểu diễn các từ và đặc trưng biểu diễn văn bản tương ứng'
        """
        df = pd.read_csv(path_file_csv)
        df['text'] = df['text'].apply(self.segment_word)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(feature_name, cache_dir='models')
        model = AutoModel.from_pretrained(feature_name, cache_dir='models').to(device)
        tqdm.pandas()
        df[['words_vector', 'sentence_vector', 'tokens']] = df['text'].progress_apply(lambda x: self.get_embedding_from_text(x, tokenizer, model, number_words_per_sample_logs))
        if path_vector_save is None:
            path_vector_save = f"{path_file_csv.split('.')[0]}_{feature_name.replace('/','_')}.pkl"
        df.to_pickle(path_vector_save)

        # Visual sentence embeddings
        embedings = df["sentence_vector"].to_numpy()
        words = [None] * len(embedings)
        embedings = np.stack(embedings)
        path_file_sentences_visual = visual_embedding(embedings, words)
        
        # Convert Array to String for show
        _df = df[:number_samples_logs]
        _df['sentence_vector'] = _df['sentence_vector'].apply(array2string)
        _df['words_vector'] = _df['words_vector'].apply(lambda x: array2string(x, True))
        return json.loads(_df.to_json(orient="records")), path_file_sentences_visual
    
        # # Visual token embeddings
        # words_embedings = df["word_vector"].to_list()
        # words_embedings = np.concatenate(words_embedings)

        # words = df['tokens'].to_list()
        # words = np.concatenate(words)

        # # embedings = np.stack(embedings)
        # path_file_sentences_visual = visual_embedding(words_embedings, words)
        # return json.loads(df.to_json(orient="records")), path_file_sentences_visual, path_file_sentences_visual

# if __name__ == "__main__":
#     neuronfeature = NeuronFeatures()
#     result, path = neuronfeature.get_features()
#     print(result)