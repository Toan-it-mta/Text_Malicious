import re
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import datetime
import os

def array2string(array, words=False):
    if words:
        words_embedding = []
        for i in array:
            output = np.array2string(i, separator=',', max_line_width=np.inf)
            clean_str = output.strip("'[]'")
            # Tách chuỗi thành các phần tử
            elements = clean_str.split(',')
            # Chuyển các phần tử thành số thực và lưu vào danh sách
            result = [element.strip() for element in elements]
            words_embedding.append(result)
        return words_embedding
    
    output = np.array2string(array, separator=',', max_line_width=np.inf)
    # Loại bỏ dấu ngoặc và dấu nháy đơn
    clean_str = output.strip("'[]'")
    # Tách chuỗi thành các phần tử
    elements = clean_str.split(',')
    # Chuyển các phần tử thành số thực và lưu vào danh sách
    result = [element.strip() for element in elements]
    return result


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

def plot_matrix(embeddings, texts, filename):
    # _, ax = plt.subplots()
    # scatter = ax.scatter(data[:, 0], data[:, 1], c=range(len(data)))
    # legend_labels = [f'Data point {i+1}' for i in range(len(data))]

    # plt.figure(figsize=(8, 8))
    for i, text in enumerate(texts):
        x, y = embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(text, (x, y), xytext=(5, 2), textcoords="offset points", ha="right", va="bottom")
    # plt.show()
    # _ = ax.legend(
    #     scatter.legend_elements()[0],
    #     legend_labels
    # )
    plt.savefig(filename)
    plt.close()

def visual_embedding(embedings, texts):

    # Giảm chiều dữ liệu về 2 chiều bằng PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(embedings)

    # Vẽ biểu đồ ma trận của dữ liệu giảm chiều
    basename = "image"
    suffix = datetime.now().strftime("%y%m%d_%H%M%S")
    file_name = "_".join([basename, suffix,'.png'])
    file_path = os.path.join('images',file_name)
    plot_matrix(reduced_data, texts, file_path)
    return file_path


# if __name__ == "__main__":
    # data = np.random.rand(100, 10)
    # words = [None]*10
    # visual_embedding(data, words)

    # Tạo các mảng mẫu
