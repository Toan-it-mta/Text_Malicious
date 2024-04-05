from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
import os

class TraditionalFeatures:
    """
    Lớp thực hiện tạo bộ từ điển và trích xuất các đặc trưng truyền thống theo từ điển
    """
    def __init__(self, path_file_stopwords:str='stopwords.txt'):
        """
            Khởi tạo các tham số và load bộ stopwords
        ### Args:
            - `path_file_stopwords (str, optional)`: Đường dẫn tới bộ stopwords. Defaults to 'stopwords.txt'.
        """
        self.vocab = None
        with open(path_file_stopwords, encoding='utf-8') as f:
            self.stop_wors = [word.strip() for word in f.readlines()]

    def create_vocab_from_corpus(self, path_file_corpus:str='datasets/train.csv', min_frequency:float=1, max_frequency:float=1.0, 
                                lower_case:bool=True, max_features=None, path_vocab_file:str = 'vocabs/vocab.txt'):
        """## Sinh bộ từ điển từ một corpus cụ thể

        ### Args:
            - `path_file_corpus (str, optional)`: Đường dẫn tới Corpus. Defaults to 'datasets/train.csv'.
            - `min_frequency (float, optional)`: Tần suất xuất hiện tối thiểu của từ để được cho vào từ điển. Defaults to 1.
            - `max_frequency (float, optional)`: Tần suất xuất hiện tối đa của từ để được cho vào từ điển. Defaults to 1.0.
            - `lower_case (bool, optional)`: Các từ có được xử lý lower_case. Defaults to True.
            - `max_features (_type_, optional)`: Kích thước tối đa của bộ từ điển. Defaults to None.
            - `path_vocab_file (str, optional)`: Đường dẫn bộ từ điển sẽ được lưu lại. Defaults to './vocabs/vocab.txt'.

        ### Returns:
            - `vocab`: Bộ từ điển được sinh ra
        """
        corpus = pd.read_csv(path_file_corpus)
        vectorizer = CountVectorizer(min_df=min_frequency, max_df=max_frequency, lowercase=lower_case, stop_words=self.stop_wors, max_features=max_features)
        vectorizer.fit_transform(corpus['text'].to_list())
        self.vocab = vectorizer.get_feature_names_out().tolist()
        self.write_vocab_to_file(self.vocab, path_vocab_file)
        return self.vocab
    
    def write_vocab_to_file(self, path_vocab_file='vocabs/vocab.txt'):
        """## Lưu trữ bộ từ điển dưới dạng file 

        ### Args:
            - `path_vocab_file (str, optional)`: Đường dẫn file lưu trữ. Defaults to 'vocabs/vocab.txt'.
        """
        try:
            with open(path_vocab_file, 'w', encoding='utf-8') as f:
                for word in self.vocab:
                    f.write(word+'\n')
        except Exception as e:
            print("Error: ", e)
    
    def load_vocab_from_file(self, path_vocab_file='vocabs/vocab.txt'):
        """## Load bộ từ điển sẵn có

        ### Args:
            - `path_vocab_file (str, optional)`: Đường dẫn tới bộ từ điển. Defaults to 'vocabs/vocab.txt'.

        ### Returns:
            - `vocab`: Bộ từ điển được lưu trữ dưới dạng List
        """
        try:
            with open(path_vocab_file, 'r', encoding='utf-8') as f:
                self.vocab = [word.strip() for word in f.readlines()]
            vocab = self.vocab
            return vocab
        except Exception as e:
            print("Error: ", e)
            
    def add_word_to_vocab(self, word: str):
        """## Thêm từ mới vào bộ từ điển

        ### Args:
            - `word (str)`: Từ cần thêm
        """
        if word not in self.vocab:
            self.vocab.append(word)

    def remove_word_from_vocab(self, word: str):
        """## Xóa một từ khỏi bộ từ điển

        ### Args:
            - `word (str)`: Từ cần xóa
        """
        if word in self.vocab:
            self.vocab.reomve(word)
    
    def get_features( self, path_file_csv:str='datasets/train.csv', feature_name:str='count-vectorizing',
                    path_vector_save:str=None, path_vectorizer_save:str=None):
        """## Trích xuất đặc trưng truyền thống. Cần thực hiện load Vocab trước nếu không sẽ tự động sinh Vocab theo kho dữ liệu

        ### Args:
            - `path_file_csv (str, optional)`: Đường dẫn tới kho dữ liệu. Defaults to 'datasets/train.csv'.
            - `feature_name (str, optional)`: Tên của đặc trưng cần trích xuất có thể là 'count-vectorizing' và 'tf-idf'. Defaults to 'count-vectorizing'.
            - `path_vector_save (str, optional)`: Đường dẫn file lưu trữ véc-tơ biểu diễn của văn bản. 
            Không chỉ định sẽ được tạo tự động với với tên được tạo từ (path_file_csv, feature_name) Defaults to None.
            - `path_vectorizer_save (str, optional)`: Đường dẫn lưu trữ công cụ trích xuất đặc trưng tương ứng. Defaults to None.

        ### Returns:
            - `df`: Một dataframe có 2 cột (text, feature_name) tương ứng với văn bản và đặc trưng tương ứng
        """
        try:
            df = pd.read_csv(path_file_csv)
            corpus = df['text'].to_list()
            if path_vectorizer_save is None:
                path_vectorizer_save = os.path.join(os.path.sep.join(path_file_csv.split(os.path.sep)[:-1]),f'vectorizers/{feature_name}-vectorizer.pkl')
            
            if os.path.exists(path_vectorizer_save):
                with open(path_vectorizer_save, 'rb') as f:
                    vectorizer_ = pickle.load(f)
                    
            elif feature_name == 'count-vectorizing':
                vectorizer_ = CountVectorizer(vocabulary=self.vocab)
                vectorizer_.fit(corpus)
                with open(path_vectorizer_save, 'wb') as f:
                    pickle.dump(vectorizer_, f)
                              
            else:
                vectorizer_ = TfidfVectorizer(vocabulary=self.vocab)
                vectorizer_.fit(corpus)
                with open(path_vectorizer_save, 'wb') as f:
                    pickle.dump(vectorizer_, f)
                    
            def vectorizer_text(text):
                X = vectorizer_.transform([text]).toarray()
                return X[0]
            
            df[feature_name] = df['text'].apply(vectorizer_text)
            if path_vector_save is None:
                path_vector_save = f"{path_file_csv.split('.')[0]}_{feature_name}.pkl"
                
            df.to_pickle(path_vector_save)
            return df
        except Exception as e:
            print("Error: ", e)

# if __name__ == "__main__":
#     #Khởi tạo phương thức trích xuất đặc trưng văn bản theo bộ từ điển đã có
#     traditionalfeature = TraditionalFeatures(path_file_stopwords='stopwords.txt')
#     traditionalfeature.load_vocab_from_file(path_vocab_file='vocabs/vocab.txt')
#     #Thực hiện trích xuất đặc trưng
#     df = traditionalfeature.get_features(path_file_csv='datasets/train.csv')
    
