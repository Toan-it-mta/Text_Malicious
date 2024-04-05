import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import py_vncorenlp
import os

class NeuronFeatures:
    """## Thực hiện trích xuất các đặc trưng sử dụng mạng Nơ-ron học sâu
    """
    def __init__(self) -> None:
        py_vncorenlp.download_model(save_dir='models/vncorenlp')
        self.rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=os.path.join(os.path.dirname(__file__),'models/vncorenlp'))
        os.chdir(os.path.dirname(__file__))
        
    def segment_word(self, text:str):
        """## Tokenizer câu thành các từ

        ### Args:
            - `text (str)`: Đoạn text cần token

        ### Returns:
            - `text`: Đoạn text sau khi token
        """
        senteces = self.rdrsegmenter.word_segment(text)
        return ' '.join(senteces)
    
    def get_embedding_from_text(self, text, tokenizer, model):
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
            outputs = model(input_ids)
            word_embeddings = outputs.last_hidden_state[0].cpu().numpy()
            text_embedding = outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy()
        return pd.Series([word_embeddings, text_embedding])  
        
    def get_features(self, path_file_csv:str='datasets/datasets.csv', feature_name:str='vinai/bartpho-word', path_vector_save:str=None):
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
        df[['word_vector', 'sentence_vector']] = df['text'].apply(lambda x: self.get_embedding_from_text(x, tokenizer, model))
        if path_vector_save is None:
            path_vector_save = f"{path_file_csv.split('.')[0]}_{feature_name.replace('/','_')}.pkl"
        df.to_pickle(path_vector_save)
        return df

if __name__ == "__main__":
    neuronfeature = NeuronFeatures()
    neuronfeature.get_featured()