"""
Modul Metrik Evaluasi
=====================
Implementasi metrik evaluasi untuk mengukur performa model:
- Accuracy
- Precision (per class & macro average)
- Recall (per class & macro average)
- F1-Score (per class & macro average)
- Confusion Matrix
"""

import numpy as np


def accuracy(y_pred, y_true):
    """
    Menghitung akurasi klasifikasi.
    
    Parameters:
        y_pred (np.ndarray): Prediksi model (one-hot atau probabilitas)
        y_true (np.ndarray): Label sebenarnya (one-hot encoded)
        
    Returns:
        float: Akurasi (0.0 - 1.0)
    """
    if y_pred.ndim > 1:
        pred_classes = np.argmax(y_pred, axis=1)
    else:
        pred_classes = (y_pred >= 0.5).astype(int)
    
    if y_true.ndim > 1:
        true_classes = np.argmax(y_true, axis=1)
    else:
        true_classes = y_true.astype(int)
    
    return np.mean(pred_classes == true_classes)


def confusion_matrix(y_pred, y_true, num_classes=None):
    """
    Menghitung Confusion Matrix.
    
    Parameters:
        y_pred (np.ndarray): Prediksi model (one-hot atau probabilitas)
        y_true (np.ndarray): Label sebenarnya (one-hot encoded)
        num_classes (int): Jumlah kelas (opsional, auto-detected jika None)
        
    Returns:
        np.ndarray: Confusion matrix (shape: num_classes x num_classes)
                    Baris = true class, Kolom = predicted class
    """
    if y_pred.ndim > 1:
        pred_classes = np.argmax(y_pred, axis=1)
    else:
        pred_classes = (y_pred >= 0.5).astype(int)
    
    if y_true.ndim > 1:
        true_classes = np.argmax(y_true, axis=1)
    else:
        true_classes = y_true.astype(int)
    
    if num_classes is None:
        num_classes = max(np.max(pred_classes), np.max(true_classes)) + 1
    
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_classes, pred_classes):
        cm[t, p] += 1
    
    return cm


def precision_recall_f1(y_pred, y_true, num_classes=None):
    """
    Menghitung Precision, Recall, dan F1-Score per kelas dan macro average.
    
    Parameters:
        y_pred (np.ndarray): Prediksi model
        y_true (np.ndarray): Label sebenarnya
        num_classes (int): Jumlah kelas
        
    Returns:
        dict: Dictionary berisi:
            - 'precision_per_class': Precision per kelas
            - 'recall_per_class': Recall per kelas
            - 'f1_per_class': F1-score per kelas
            - 'macro_precision': Macro average precision
            - 'macro_recall': Macro average recall
            - 'macro_f1': Macro average F1-score
    """
    cm = confusion_matrix(y_pred, y_true, num_classes)
    num_classes = cm.shape[0]
    
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp  # Kolom c minus diagonal
        fn = np.sum(cm[c, :]) - tp  # Baris c minus diagonal
        
        # Precision = TP / (TP + FP)
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall = TP / (TP + FN)
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        if precision[c] + recall[c] > 0:
            f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])
        else:
            f1[c] = 0.0
    
    return {
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'macro_precision': np.mean(precision),
        'macro_recall': np.mean(recall),
        'macro_f1': np.mean(f1)
    }


def classification_report(y_pred, y_true, class_names=None):
    """
    Membuat classification report yang mirip dengan sklearn.
    
    Parameters:
        y_pred (np.ndarray): Prediksi model
        y_true (np.ndarray): Label sebenarnya
        class_names (list): Nama-nama kelas (opsional)
        
    Returns:
        str: Classification report dalam format tabel
    """
    results = precision_recall_f1(y_pred, y_true)
    num_classes = len(results['precision_per_class'])
    
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    
    # Hitung support (jumlah sampel per kelas)
    if y_true.ndim > 1:
        true_classes = np.argmax(y_true, axis=1)
    else:
        true_classes = y_true.astype(int)
    
    support = np.zeros(num_classes, dtype=int)
    for c in range(num_classes):
        support[c] = np.sum(true_classes == c)
    
    # Format report
    header = f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}"
    separator = "-" * len(header)
    
    report = [header, separator]
    
    for i in range(num_classes):
        name = class_names[i] if i < len(class_names) else str(i)
        line = (f"{name:<15} {results['precision_per_class'][i]:>10.4f} "
                f"{results['recall_per_class'][i]:>10.4f} "
                f"{results['f1_per_class'][i]:>10.4f} "
                f"{support[i]:>10d}")
        report.append(line)
    
    report.append(separator)
    
    # Macro average
    total_support = np.sum(support)
    macro_line = (f"{'Macro Avg':<15} {results['macro_precision']:>10.4f} "
                  f"{results['macro_recall']:>10.4f} "
                  f"{results['macro_f1']:>10.4f} "
                  f"{total_support:>10d}")
    report.append(macro_line)
    
    # Accuracy
    acc = accuracy(y_pred, y_true)
    acc_line = (f"{'Accuracy':<15} {'':>10} {'':>10} "
                f"{acc:>10.4f} {total_support:>10d}")
    report.append(acc_line)
    
    return "\n".join(report)
