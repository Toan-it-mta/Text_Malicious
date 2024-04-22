import re
from sklearn.metrics import f1_score, accuracy_score



def compute_metrics_acc_score(predictions, labels):
    score = accuracy_score(labels, predictions)
    return {'accuracy': float(score)}

def compute_metrics_f_score(predictions, labels):
    score = f1_score(labels,predictions, average='macro')
    return {"f1_score": float(score)}

def compute_metrics(predictions, labels):
    """## Tính toán các thang đo

    ### Args:
        - `predictions (_type_)`: Đầu ra mô hình dự đoán
        - `labels (_type_)`: Nhãn chính xác

    ### Returns:
        - 'score': Hai điểm accuracy và f-1_score
    """
    metric_results = {}
    for metric_function in [compute_metrics_acc_score, compute_metrics_f_score]:
        metric_results.update(metric_function(predictions, labels))
    return metric_results

def preprocessing_text(text):
    text = re.sub("\r", "\n", text)
    text = re.sub("\n{2,}", "\n", text)
    text = re.sub("…", ".", text)
    text = re.sub("/.{2,}", ".", text)
    text = text.strip()
    text = text.lower()
    return text