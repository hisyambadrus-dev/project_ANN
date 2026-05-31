"""
Modul Visualisasi
=================
Fungsi-fungsi untuk membuat visualisasi hasil eksperimen:
- Grafik Loss vs Epoch
- Grafik Accuracy vs Epoch
- Confusion Matrix Heatmap
- Perbandingan eksperimen
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interaktif
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


# Konfigurasi default plot
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 100,
})


def plot_training_history(history, title="Training History", save_path=None):
    """
    Plot grafik loss dan accuracy selama training.
    
    Parameters:
        history (dict): Dictionary dengan keys 'train_loss', 'train_accuracy',
                       'test_loss', 'test_accuracy'
        title (str): Judul grafik
        save_path (str): Path untuk menyimpan gambar (opsional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if history.get('test_loss'):
        ax1.plot(epochs, history['test_loss'], 'r-', label='Testing Loss', linewidth=2)
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy
    ax2.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    if history.get('test_accuracy'):
        ax2.plot(epochs, history['test_accuracy'], 'r-', label='Testing Accuracy', linewidth=2)
    ax2.set_title(f'{title} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Grafik disimpan ke: {save_path}")
    
    plt.close()


def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix", 
                          save_path=None):
    """
    Plot confusion matrix sebagai heatmap.
    
    Parameters:
        cm (np.ndarray): Confusion matrix
        class_names (list): Nama-nama kelas
        title (str): Judul grafik
        save_path (str): Path untuk menyimpan gambar
    """
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Label
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Anotasi nilai
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrix disimpan ke: {save_path}")
    
    plt.close()


def plot_comparison(results_dict, metric='test_accuracy', 
                    title="Perbandingan Eksperimen", save_path=None):
    """
    Plot perbandingan beberapa eksperimen dalam bar chart.
    
    Parameters:
        results_dict (dict): {nama_eksperimen: nilai_metrik}
        metric (str): Nama metrik yang dibandingkan
        title (str): Judul grafik
        save_path (str): Path untuk menyimpan gambar
    """
    names = list(results_dict.keys())
    values = list(results_dict.values())
    
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=0.5)
    
    # Anotasi nilai di atas bar
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_title(title)
    ax.set_xlabel('Konfigurasi')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Perbandingan disimpan ke: {save_path}")
    
    plt.close()


def plot_multiple_histories(histories_dict, metric='loss', 
                           title="Perbandingan Training", save_path=None):
    """
    Plot beberapa training history dalam satu grafik.
    
    Parameters:
        histories_dict (dict): {nama: history_dict}
        metric (str): 'loss' atau 'accuracy'
        title (str): Judul grafik
        save_path (str): Path untuk menyimpan gambar
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories_dict)))
    
    for (name, history), color in zip(histories_dict.items(), colors):
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Training metrics
        ax1.plot(epochs, history['train_loss'], label=f'{name}', 
                color=color, linewidth=1.5)
        
        ax2.plot(epochs, history['train_accuracy'], label=f'{name}', 
                color=color, linewidth=1.5)
    
    ax1.set_title(f'{title} - Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title(f'{title} - Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Grafik perbandingan disimpan ke: {save_path}")
    
    plt.close()
