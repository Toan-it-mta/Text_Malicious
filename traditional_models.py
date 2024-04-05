from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from traditional_feature import TraditionalFeatured
from neuron_feature import NeuronFeature
from sklearn.model_selection import train_test_split
from utils import compute_metrics
import os
import pickle

def create_svm_model():
    classifier = SVC()
    return classifier

def create_navie_bayes_model():
    classifier = GaussianNB()
    return classifier

def create_knn_model(n_neighbors=5):
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    return classifier
    
def train_model(model_name:str='k-nn', feature_name:str='tf-idf', path_file_train_csv:str='datasets/train.csv', 
            path_vocab_file:str=None, val_size:float = 0.1, path_model_save:str=None,
            path_vectorizer_save:str=None, **kargs):
    
    if feature_name=='tf-idf' or feature_name=='count-vectorizing':
        traditionalfeature = TraditionalFeatured()
        if path_vocab_file is not None:
            traditionalfeature.load_vocab_from_file(path_vocab_file)
        df_train = traditionalfeature.get_featured(path_file_csv=path_file_train_csv, feature_name=feature_name, 
                                                   path_vectorizer_save=path_vectorizer_save)
        
        X, y = df_train[feature_name].to_list(), df_train['label'].to_list()
    else:
        neuronfeature = NeuronFeature()
        df_train = neuronfeature.get_featured(path_file_csv=path_file_train_csv, feature_name=feature_name)
        X, y = df_train["sentence_vector"].to_list(), df_train['label'].to_list()
        
    X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=val_size, random_state=42)
    
    if model_name == 'k-nn':
        classifier = create_knn_model(**kargs)
    elif model_name == 'navie-bayes':
        classifier = create_navie_bayes_model(**kargs)
    else:
        classifier = create_svm_model(**kargs) 
    
    classifier.fit(X_train, y_train)
    
    if path_model_save is None:
        model_name_ = f'model_{feature_name}_{model_name}.pkl'
        path_model_save = os.path.join('models/log-train', model_name_)
    
    with open(path_model_save, 'wb') as f:
        pickle.dump(classifier, f)
        
    val_predicts = classifier.predict(X_val).tolist()
    score = compute_metrics(val_predicts, y_val)
    
    return {
        'val_acc': score['accuracy'],
        'val_f1': score['f1_score']
    }
    

def test_model(path_model:str=None, feature_name:str='tf-idf', path_file_test_csv:str='datasets/test.csv', path_vectorizer_save:str=None):
    if feature_name=='tf-idf' or feature_name=='count-vectorizing':
        traditionalfeature = TraditionalFeatured()
        if path_vectorizer_save is None:
            path_vectorizer_save = os.path.join(os.path.sep.join(path_file_test_csv.split(os.path.sep)[:-1]),
                                                f'vectorizers/{feature_name}-vectorizer.pkl')
        
        df_train = traditionalfeature.get_featured(path_file_csv=path_file_test_csv, feature_name=feature_name, 
                                                   path_vectorizer_save=path_vectorizer_save)
        X, y = df_train[feature_name].to_list(), df_train['label'].to_list()
        
    else:
        neuronfeature = NeuronFeature()
        df_train = neuronfeature.get_featured(path_file_csv=path_file_test_csv, feature_name=feature_name)
        X, y = df_train["sentence_vector"].to_list(), df_train['label'].to_list()
        
    with open(path_model, 'rb') as f:
        classifier = pickle.load(f)
    
    val_predicts = classifier.predict(X).tolist()
    score = compute_metrics(val_predicts, y)
    
    return {
        'test_acc': score['accuracy'],
        'test_f1': score['f1_score']
    }

def infer_model(path_model:str=None, feature_name:str='tf-idf', text:str='', path_vectorizer_save:str=None):
    id2label = {0: "normal", 1: "malicious"}
    
    if feature_name=='tf-idf' or feature_name=='count-vectorizing':
        with open(path_vectorizer_save, 'rb') as f:
            vectorizer_ = pickle.load(f)
        embedding = vectorizer_.transform([text]).toarray()
          
    else:
        neuronfeature = NeuronFeature()
        embedding = neuronfeature.get_embedding_from_text(model_name=feature_name, text=text)[1]
    
    with open(path_model, 'rb') as f:
        classifier = pickle.load(f)
    
    id = classifier.predict(embedding).tolist()[0]
    return id2label[id]

if __name__ == "__main__":
    print(train_model(model_name='svm', feature_name='vinai/phobert-base'))
    # print(test_model('models/log-train/model_tf-idf_svm.pkl', feature_name='tf-idf', path_file_test_csv='datasets/test.csv'))
    # print(infer_model('models/log-train/model_tf-idf_svm.pkl', feature_name='tf-idf', 
    #                   path_vectorizer_save='datasets/vectorizers/tf-idf-vectorizer.pkl', text='địt'))