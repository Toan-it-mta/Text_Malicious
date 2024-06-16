from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from traditional_features import TraditionalFeatures
from neuron_features import NeuronFeatures
from sklearn.model_selection import train_test_split
from utils import compute_metrics
import os
import pickle
import torch
from transformers import AutoModel, AutoTokenizer
import json
import pandas as pd

def create_svm_model(**kargs):
    """## Khởi tạo mô hình SVM
    """
    classifier = SVC(**kargs)
    return classifier

def create_navie_bayes_model(**kargs):
    """## Khởi tạo mô hình Navie Bayes
    """
    classifier = GaussianNB(**kargs)
    return classifier

def create_knn_model(n_neighbors=5, **kargs):
    """## Khởi tạo mô hình K-NN
    """
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, **kargs)
    return classifier
    
async def train(model_name:str='k-nn', feature_name:str='tf-idf', path_file_train_csv:str='datasets/train.csv',
                path_file_stop_words:str="stopwords.txt", val_size:float = 0.1,
                path_file_vocab:str=None,  path_model_save:str=None,
                path_vectorizer_save:str=None, **kargs):
    """## Huấn luyện các mô hình truyền thống với các đặc trưng cụ thể

    ### Args:
        - `model_name (str, optional)`: Mô hình sử dụng. Các giá trị có thể nhận là: 'k-nn', 'svm', 'navie-bayes'. Defaults to 'k-nn'.
        - `feature_name (str, optional)`: Loại đặc trưng cần trích xuất. Các giá trị có thể nhận là: 'tf-idf', 'count-vectorizing', 'vinai/phobert-base', 'vinai/bartpho-word' . Defaults to 'tf-idf'.
        - `path_file_train_csv (str, optional)`: Đường dần tới bộ dữ liệu train. Defaults to 'datasets/train.csv'.
        - `path_file_stop_words (str, optional)`: Đường dẫn tới file stopwords. Defaults to "stopwords.txt".
        - `val_size (float, optional)`: Tỷ lệ chia bộ Validation . Defaults to 0.1.
        - `path_file_vocab (str, optional)`: Đường dẫn tới vocab sử dụng áp dụng cho các phương pháp trích xuất đặc trưng truyền thống. Defaults to None.
        - `path_model_save (str, optional)`: Đương dẫn model sẽ được lưu. Defaults to None. Không bổ sung sẽ được tạo tự động trong folder: models/log-train
        - `path_vectorizer_save (str, optional)`: Đường dẫn mô hình trích xuất đặc trưng truyền thống. Defaults to None. Sẽ được tạo tự động trong folder datasets/vectorizers

    ### Returns:
        - `Valid_score`: Thang đo độ chính xác của tập Valid
    """
    if feature_name=='tf-idf' or feature_name=='count-vectorizing':
        traditionalfeature = TraditionalFeatures(path_file_stopwords=path_file_stop_words)
        if path_file_vocab is not None:
            traditionalfeature.load_vocab_from_file(path_file_vocab)
        result, _, _ = traditionalfeature.get_features(path_file_csv=path_file_train_csv, feature_name=feature_name, 
                                                   path_vectorizer_save=path_vectorizer_save)
        str_result = json.dumps(result)
        df_train = pd.read_json(path_or_buf=str_result, orient='records')
        X, y = df_train[feature_name].to_list(), df_train['label'].to_list()
        print(len(X))
    else:
        neuronfeature = NeuronFeatures()
        result, _, _ = neuronfeature.get_features(path_file_csv=path_file_train_csv, feature_name=feature_name)
        str_result = json.dumps(result)
        df_train = pd.read_json(path_or_buf=str_result, orient='records')
        X, y = df_train["arr_sentence_vector"].to_list(), df_train['label'].to_list()
        print(len(X))
        
    X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=val_size, random_state=42)
    
    if model_name == 'k-nn':
        classifier = create_knn_model(**kargs)
    elif model_name == 'navie-bayes':
        classifier = create_navie_bayes_model(**kargs)
    else:
        classifier = create_svm_model(**kargs) 
    
    classifier.fit(X_train, y_train)
    
    if path_model_save is None:
        model_name_ = f"model_{feature_name.replace('/','_')}_{model_name}.pkl"
        path_model_save = os.path.join('models/log-train', model_name_)
    
    with open(path_model_save, 'wb') as f:
        pickle.dump(classifier, f)
        
    val_predicts = classifier.predict(X_val).tolist()
    score = compute_metrics(val_predicts, y_val)
    
    return {
        'val_acc': score['accuracy'],
        'val_f1': score['f1_score']
    }
    

async def test(path_model:str=None, feature_name:str='tf-idf', path_file_test_csv:str='datasets/test.csv', path_vectorizer_save:str=None):
    """## Test các mô hình truyền thống với các đặc trưng cụ thể

    ### Args:
        - `path_model (str, optional)`: Đường dẫn tới mô hình. Defaults to None.
        - `feature_name (str, optional)`: Loại đặc trưng cần trích xuất. Các giá trị có thể nhận là: 'tf-idf', 'count-vectorizing', 'vinai/phobert-base', 'vinai/bartpho-word' . Defaults to 'tf-idf'.
        - `path_file_test_csv (str, optional)`: Đường dần tới bộ dữ liệu test. Defaults to 'datasets/test.csv'.
        - `path_vectorizer_save (str, optional)`: Đường dẫn mô hình trích xuất đặc trưng truyền thống. Defaults to None.

    ### Yields:
        - - `Test_score`: Thang đo độ chính xác của tập Test
    """
    if feature_name=='tf-idf' or feature_name=='count-vectorizing':
        traditionalfeature = TraditionalFeatures()
        if path_vectorizer_save is None:
            path_vectorizer_save = os.path.join(os.path.sep.join(path_file_test_csv.split(os.path.sep)[:-1]),
                                                f'vectorizers/{feature_name}-vectorizer.pkl')
        
        result, _, _ = traditionalfeature.get_features(path_file_csv=path_file_test_csv, feature_name=feature_name, 
                                                   path_vectorizer_save=path_vectorizer_save)
        str_result = json.dumps(result)
        df_train = pd.read_json(path_or_buf=str_result, orient='records')
        X, y = df_train[feature_name].to_list(), df_train['label'].to_list()
        
    else:
        neuronfeature = NeuronFeatures()
        result, _, _ = neuronfeature.get_features(path_file_csv=path_file_test_csv, feature_name=feature_name)
        str_result = json.dumps(result)
        df_train = pd.read_json(path_or_buf=str_result, orient='records')
        X, y = df_train["arr_sentence_vector"].to_list(), df_train['label'].to_list()
        
    with open(path_model, 'rb') as f:
        classifier = pickle.load(f)
    
    val_predicts = classifier.predict(X).tolist()
    score = compute_metrics(val_predicts, y)
    
    return {
        'test_acc': score['accuracy'],
        'test_f1': score['f1_score']
    }

async def infer(path_model:str=None, feature_name:str='tf-idf', text:str='', path_vectorizer_save:str=None):
    """## Infer mô hình với đoạn text cụ thể

    ### Args:
        - `path_model (str, optional)`: Đường dẫn tới mô hình. Defaults to None.
        - `feature_name (str, optional)`: Loại đặc trưng cần trích xuất. Các giá trị có thể nhận là: 'tf-idf', 'count-vectorizing', 'vinai/phobert-base', 'vinai/bartpho-word' . Defaults to 'tf-idf'.
        - `text (str, optional)`: Đoạn văn bản cần test.
        - `path_vectorizer_save (str, optional)`: Đường dẫn mô hình trích xuất đặc trưng truyền thống. Bắt buộc đối với các đặc trưng truyền thống

    ### Yields:
        - - `label`:  Nhãn của đoạn text {0: "normal", 1: "malicious"}
    """
    id2label = {0: "normal", 1: "malicious"}
    
    if feature_name=='tf-idf' or feature_name=='count-vectorizing':
        with open(path_vectorizer_save, 'rb') as f:
            vectorizer_ = pickle.load(f)
        embedding = vectorizer_.transform([text]).toarray()
          
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(feature_name, cache_dir='models')
        model = AutoModel.from_pretrained(feature_name, cache_dir='models').to(device)
        neuronfeature = NeuronFeatures()
        embedding = neuronfeature.get_embedding_from_text(text=text, tokenizer=tokenizer, model=model, number_words_per_sample_logs= 1)[1].reshape(1, -1)
    
    with open(path_model, 'rb') as f:
        classifier = pickle.load(f)
    
    id = classifier.predict(embedding).tolist()[0]
    return id2label[id]

# if __name__ == "__main__":
    # print(train(model_name='navie-bayes', feature_name='tf-idf', path_file_train_csv="datasets/_train.csv"))
    # print(test(path_model="models/log-train/model_vinai_phobert-base_navie-bayes.pkl", feature_name='vinai/phobert-base', path_file_test_csv='datasets/test.csv'))
    # print(infer('models/log-train/model_vinai_phobert-base_navie-bayes.pkl', feature_name='vinai/phobert-base', 
    #                   path_vectorizer_save='datasets/vectorizers/tf-idf-vectorizer.pkl', text='"""chào bạn"""'))