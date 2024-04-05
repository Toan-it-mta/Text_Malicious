from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
import os

class TraditionalFeatured:
    def __init__(self, path_file_stopwords:str='stopwords.txt'):
        self.vocab = None
        with open(path_file_stopwords, encoding='utf-8') as f:
            self.stop_wors = [word.strip() for word in f.readlines()]
        
    #Tạo và sinh vocab
    def create_vocab_from_corpus(self, path_file_corpus:str='datasets/train.csv', min_frequency:float=1, max_frequency:float=1.0, lower_case:bool=True, max_features=None, path_vocab_file:str = './vocabs/vocab.txt'):
        corpus = pd.read_csv(path_file_corpus)
        vectorizer = CountVectorizer(min_df=min_frequency, max_df=max_frequency, lowercase=lower_case, stop_words=self.stop_wors, max_features=max_features)
        vectorizer.fit_transform(corpus['text'].to_list())
        self.vocab = vectorizer.get_feature_names_out().tolist()
        self.write_vocab_to_file(self.vocab, path_vocab_file)
        return self.vocab
    
    def write_vocab_to_file(self, vocab, path_vocab_file='vocabs/vocab.txt'):
        try:
            with open(path_vocab_file, 'w', encoding='utf-8') as f:
                for word in vocab:
                    f.write(word+'\n')
        except Exception as e:
            print("Error: ", e)
    
    def load_vocab_from_file(self, path_vocab_file='vocabs/vocab.txt'):
        try:
            with open(path_vocab_file, 'r', encoding='utf-8') as f:
                self.vocab = [word.strip() for word in f.readlines()]
            vocab = self.vocab
            return vocab
        except Exception as e:
            print("Error: ", e)
            
    def add_word_to_vocab(self, word: str):
        if word not in self.vocab:
            self.vocab.append(word)

    def remove_word_from_vocab(self, word: str):
        if word in self.vocab:
            self.vocab.reomve(word)
    
    # Tạo và sinh đặc trưng
    def get_featured( self, path_file_csv:str='datasets/train.csv', feature_name:str='count-vectorizing',
                    path_vector_save:str=None, path_vectorizer_save:str=None):
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

if __name__ == "__main__":
    traditionalfeature = TraditionalFeatured(path_file_stopwords='stopwords.txt')
    traditionalfeature.create_vocab_from_corpus()