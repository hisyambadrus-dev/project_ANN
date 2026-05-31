"""
Script Eksperimen Neural Network From Scratch
===============================================
Menjalankan semua eksperimen yang diwajibkan dalam tugas besar:

1. Pengaruh Arsitektur (depth & width)
2. Pengaruh Inisialisasi Bobot
3. Pengaruh Fungsi Aktivasi
4. Pengaruh Learning Rate
5. Pengaruh Metode Optimasi
6. Pengaruh Batch Size
7. Pengaruh Jumlah Epoch

Setiap eksperimen menggunakan pendekatan controlled experiment:
mengubah satu parameter, parameter lainnya konstan.
"""

import numpy as np
import sys
import os
import time

# Tambahkan src ke path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_network import NeuralNetwork, build_model
from data_loader import (load_dataset, normalize, train_test_split, 
                         CLASS_NAMES, download_dataset)
from metrics import accuracy, confusion_matrix, precision_recall_f1, classification_report
from visualizations import (plot_training_history, plot_confusion_matrix, 
                           plot_comparison, plot_multiple_histories)
from utils import set_seed, save_results, Timer


# ==================== KONFIGURASI DEFAULT ====================
DEFAULT_CONFIG = {
    'seed': 42,
    'test_size': 0.2,
    'normalization': 'standard',
    'layer_sizes': [16, 64, 32, 7],      # Default: 2 hidden layers
    'activation': 'relu',
    'output_activation': 'softmax',
    'init_method': 'he',
    'loss': 'categorical_crossentropy',
    'optimizer': 'minibatch_gd',
    'learning_rate': 0.01,
    'batch_size': 32,
    'epochs': 100,
}

# Direktori output
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          'results', 'scratch')


def load_and_prepare_data(data_path=None):
    """
    Memuat dan mempersiapkan dataset.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if data_path is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'data')
        
        # Cari file dataset
        possible_files = [
            os.path.join(data_dir, 'Dry_Bean_Dataset.csv'),
            os.path.join(data_dir, 'DryBeanDataset', 'Dry_Bean_Dataset.csv'),
            os.path.join(data_dir, 'Dry_Bean_Dataset.xlsx'),
            os.path.join(data_dir, 'DryBeanDataset', 'Dry_Bean_Dataset.xlsx'),
            os.path.join(data_dir, 'dry+bean+dataset', 'DryBeanDataset', 
                        'Dry_Bean_Dataset.xlsx'),
        ]
        
        data_path = None
        for f in possible_files:
            if os.path.exists(f):
                data_path = f
                break
        
        if data_path is None:
            print("Dataset tidak ditemukan. Mencoba mengunduh...")
            data_path = download_dataset(data_dir)
            if data_path is None:
                raise FileNotFoundError(
                    "Tidak dapat menemukan dataset. Silakan unduh Dry Bean Dataset "
                    "dari https://archive.ics.uci.edu/dataset/602/dry+bean+dataset "
                    "dan simpan di folder 'data/'"
                )
    
    # Load dataset
    X, y_encoded, y_onehot = load_dataset(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=DEFAULT_CONFIG['test_size'], 
        seed=DEFAULT_CONFIG['seed']
    )
    
    # Normalisasi (fit pada training, apply pada testing)
    X_train, norm_params = normalize(X_train, method=DEFAULT_CONFIG['normalization'])
    from data_loader import apply_normalization
    X_test = apply_normalization(X_test, norm_params)
    
    print(f"\nData Training: {X_train.shape[0]} sampel")
    print(f"Data Testing:  {X_test.shape[0]} sampel")
    
    return X_train, X_test, y_train, y_test


def run_single_experiment(X_train, X_test, y_train, y_test, config, exp_name=""):
    """
    Menjalankan satu eksperimen dengan konfigurasi tertentu.
    
    Parameters:
        config (dict): Konfigurasi eksperimen
        exp_name (str): Nama eksperimen
        
    Returns:
        dict: Hasil eksperimen (history, metrics, config)
    """
    print(f"\n{'='*60}")
    print(f"EKSPERIMEN: {exp_name}")
    print(f"{'='*60}")
    
    set_seed(config.get('seed', 42))
    
    # Build model
    model = build_model(
        layer_sizes=config['layer_sizes'],
        activations=config['activation'],
        output_activation=config.get('output_activation', 'softmax'),
        init_method=config['init_method'],
        loss=config['loss'],
        optimizer=config['optimizer'],
        learning_rate=config['learning_rate'],
        seed=config.get('seed', 42)
    )
    
    model.summary()
    
    # Training
    with Timer("Training"):
        history = model.train(
            X_train, y_train, X_test, y_test,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            verbose=True,
            seed=config.get('seed', 42)
        )
    
    # Evaluasi
    eval_results = model.evaluate(X_test, y_test, class_names=CLASS_NAMES)
    
    return {
        'history': history,
        'eval': eval_results,
        'config': config,
        'name': exp_name
    }


# ==================== EKSPERIMEN 1: ARSITEKTUR ====================
def experiment_architecture(X_train, X_test, y_train, y_test):
    """
    Eksperimen pengaruh kedalaman (depth) dan lebar (width) jaringan.
    
    Pengujian:
    - Variasi jumlah hidden layer: 1, 2, 3
    - Variasi jumlah neuron: 16, 32, 64, 128
    """
    print("\n" + "=" * 70)
    print("EKSPERIMEN 1: PENGARUH ARSITEKTUR JARINGAN (DEPTH & WIDTH)")
    print("=" * 70)
    
    results = {}
    histories = {}
    
    # === 1a. Variasi Depth (Jumlah Hidden Layer) ===
    depth_configs = {
        '1 Hidden Layer [64]': [16, 64, 7],
        '2 Hidden Layer [64,32]': [16, 64, 32, 7],
        '3 Hidden Layer [128,64,32]': [16, 128, 64, 32, 7],
    }
    
    for name, layer_sizes in depth_configs.items():
        config = {**DEFAULT_CONFIG, 'layer_sizes': layer_sizes}
        result = run_single_experiment(X_train, X_test, y_train, y_test, 
                                       config, f"Depth - {name}")
        results[f"depth_{name}"] = result
        histories[name] = result['history']
    
    # Visualisasi perbandingan depth
    plot_multiple_histories(histories, title="Pengaruh Depth",
                          save_path=os.path.join(OUTPUT_DIR, 'exp1_depth_comparison.png'))
    
    depth_acc = {k: v['eval']['accuracy'] for k, v in results.items() if 'depth' in k}
    plot_comparison(depth_acc, metric='Accuracy',
                   title="Perbandingan Depth - Test Accuracy",
                   save_path=os.path.join(OUTPUT_DIR, 'exp1_depth_accuracy.png'))
    
    # === 1b. Variasi Width (Jumlah Neuron per Layer) ===
    width_configs = {
        'Width 16': [16, 16, 7],
        'Width 32': [16, 32, 7],
        'Width 64': [16, 64, 7],
        'Width 128': [16, 128, 7],
    }
    
    histories_width = {}
    for name, layer_sizes in width_configs.items():
        config = {**DEFAULT_CONFIG, 'layer_sizes': layer_sizes}
        result = run_single_experiment(X_train, X_test, y_train, y_test, 
                                       config, f"Width - {name}")
        results[f"width_{name}"] = result
        histories_width[name] = result['history']
    
    plot_multiple_histories(histories_width, title="Pengaruh Width",
                          save_path=os.path.join(OUTPUT_DIR, 'exp1_width_comparison.png'))
    
    width_acc = {k: v['eval']['accuracy'] for k, v in results.items() if 'width' in k}
    plot_comparison(width_acc, metric='Accuracy',
                   title="Perbandingan Width - Test Accuracy",
                   save_path=os.path.join(OUTPUT_DIR, 'exp1_width_accuracy.png'))
    
    return results


# ==================== EKSPERIMEN 2: INISIALISASI BOBOT ====================
def experiment_weight_init(X_train, X_test, y_train, y_test):
    """
    Eksperimen pengaruh metode inisialisasi bobot.
    
    Pengujian: zero, random_uniform, random_normal, xavier, he
    """
    print("\n" + "=" * 70)
    print("EKSPERIMEN 2: PENGARUH INISIALISASI BOBOT")
    print("=" * 70)
    
    init_methods = ['zero', 'random_uniform', 'random_normal', 'xavier', 'he']
    
    results = {}
    histories = {}
    
    for init_method in init_methods:
        config = {**DEFAULT_CONFIG, 'init_method': init_method}
        name = f"Init: {init_method}"
        result = run_single_experiment(X_train, X_test, y_train, y_test, 
                                       config, name)
        results[init_method] = result
        histories[init_method] = result['history']
    
    # Visualisasi
    plot_multiple_histories(histories, title="Pengaruh Inisialisasi Bobot",
                          save_path=os.path.join(OUTPUT_DIR, 'exp2_init_comparison.png'))
    
    init_acc = {k: v['eval']['accuracy'] for k, v in results.items()}
    plot_comparison(init_acc, metric='Accuracy',
                   title="Perbandingan Inisialisasi - Test Accuracy",
                   save_path=os.path.join(OUTPUT_DIR, 'exp2_init_accuracy.png'))
    
    return results


# ==================== EKSPERIMEN 3: FUNGSI AKTIVASI ====================
def experiment_activation(X_train, X_test, y_train, y_test):
    """
    Eksperimen pengaruh fungsi aktivasi pada hidden layers.
    
    Pengujian: sigmoid, relu, tanh
    (Output layer selalu menggunakan softmax)
    """
    print("\n" + "=" * 70)
    print("EKSPERIMEN 3: PENGARUH FUNGSI AKTIVASI")
    print("=" * 70)
    
    activation_funcs = ['sigmoid', 'relu', 'tanh']
    
    results = {}
    histories = {}
    
    for act in activation_funcs:
        # Gunakan Xavier untuk sigmoid/tanh, He untuk ReLU
        init = 'he' if act == 'relu' else 'xavier'
        config = {**DEFAULT_CONFIG, 'activation': act, 'init_method': init}
        name = f"Activation: {act}"
        result = run_single_experiment(X_train, X_test, y_train, y_test, 
                                       config, name)
        results[act] = result
        histories[act] = result['history']
    
    # Visualisasi
    plot_multiple_histories(histories, title="Pengaruh Fungsi Aktivasi",
                          save_path=os.path.join(OUTPUT_DIR, 'exp3_activation_comparison.png'))
    
    act_acc = {k: v['eval']['accuracy'] for k, v in results.items()}
    plot_comparison(act_acc, metric='Accuracy',
                   title="Perbandingan Fungsi Aktivasi - Test Accuracy",
                   save_path=os.path.join(OUTPUT_DIR, 'exp3_activation_accuracy.png'))
    
    # Confusion matrix untuk setiap aktivasi
    for act, result in results.items():
        cm = result['eval']['confusion_matrix']
        plot_confusion_matrix(cm, CLASS_NAMES, title=f"Confusion Matrix - {act}",
                            save_path=os.path.join(OUTPUT_DIR, f'exp3_cm_{act}.png'))
    
    return results


# ==================== EKSPERIMEN 4: LEARNING RATE ====================
def experiment_learning_rate(X_train, X_test, y_train, y_test):
    """
    Eksperimen pengaruh learning rate.
    
    Pengujian: 0.1, 0.01, 0.001
    """
    print("\n" + "=" * 70)
    print("EKSPERIMEN 4: PENGARUH LEARNING RATE")
    print("=" * 70)
    
    learning_rates = [0.1, 0.01, 0.001]
    
    results = {}
    histories = {}
    
    for lr in learning_rates:
        config = {**DEFAULT_CONFIG, 'learning_rate': lr}
        name = f"LR: {lr}"
        result = run_single_experiment(X_train, X_test, y_train, y_test, 
                                       config, name)
        results[f"lr_{lr}"] = result
        histories[f"lr={lr}"] = result['history']
    
    # Visualisasi
    plot_multiple_histories(histories, title="Pengaruh Learning Rate",
                          save_path=os.path.join(OUTPUT_DIR, 'exp4_lr_comparison.png'))
    
    lr_acc = {k: v['eval']['accuracy'] for k, v in results.items()}
    plot_comparison(lr_acc, metric='Accuracy',
                   title="Perbandingan Learning Rate - Test Accuracy",
                   save_path=os.path.join(OUTPUT_DIR, 'exp4_lr_accuracy.png'))
    
    return results


# ==================== EKSPERIMEN 5: METODE OPTIMASI ====================
def experiment_optimizer(X_train, X_test, y_train, y_test):
    """
    Eksperimen pengaruh metode optimasi.
    
    Pengujian: Mini-batch GD, Momentum, Adam
    """
    print("\n" + "=" * 70)
    print("EKSPERIMEN 5: PENGARUH METODE OPTIMASI")
    print("=" * 70)
    
    optimizer_configs = {
        'Mini-batch GD': {'optimizer': 'minibatch_gd', 'learning_rate': 0.01},
        'Momentum': {'optimizer': 'momentum', 'learning_rate': 0.01},
        'Adam': {'optimizer': 'adam', 'learning_rate': 0.001},
    }
    
    results = {}
    histories = {}
    
    for name, opt_config in optimizer_configs.items():
        config = {**DEFAULT_CONFIG, **opt_config}
        result = run_single_experiment(X_train, X_test, y_train, y_test, 
                                       config, f"Optimizer: {name}")
        results[name] = result
        histories[name] = result['history']
    
    # Visualisasi
    plot_multiple_histories(histories, title="Pengaruh Metode Optimasi",
                          save_path=os.path.join(OUTPUT_DIR, 'exp5_optimizer_comparison.png'))
    
    opt_acc = {k: v['eval']['accuracy'] for k, v in results.items()}
    plot_comparison(opt_acc, metric='Accuracy',
                   title="Perbandingan Optimizer - Test Accuracy",
                   save_path=os.path.join(OUTPUT_DIR, 'exp5_optimizer_accuracy.png'))
    
    return results


# ==================== EKSPERIMEN 6: BATCH SIZE ====================
def experiment_batch_size(X_train, X_test, y_train, y_test):
    """
    Eksperimen pengaruh ukuran batch.
    
    Pengujian: 16, 32, 64
    """
    print("\n" + "=" * 70)
    print("EKSPERIMEN 6: PENGARUH BATCH SIZE")
    print("=" * 70)
    
    batch_sizes = [16, 32, 64]
    
    results = {}
    histories = {}
    
    for bs in batch_sizes:
        config = {**DEFAULT_CONFIG, 'batch_size': bs}
        name = f"Batch Size: {bs}"
        result = run_single_experiment(X_train, X_test, y_train, y_test, 
                                       config, name)
        results[f"bs_{bs}"] = result
        histories[f"batch={bs}"] = result['history']
    
    # Visualisasi
    plot_multiple_histories(histories, title="Pengaruh Batch Size",
                          save_path=os.path.join(OUTPUT_DIR, 'exp6_batchsize_comparison.png'))
    
    bs_acc = {k: v['eval']['accuracy'] for k, v in results.items()}
    plot_comparison(bs_acc, metric='Accuracy',
                   title="Perbandingan Batch Size - Test Accuracy",
                   save_path=os.path.join(OUTPUT_DIR, 'exp6_batchsize_accuracy.png'))
    
    return results


# ==================== EKSPERIMEN 7: JUMLAH EPOCH ====================
def experiment_epochs(X_train, X_test, y_train, y_test):
    """
    Eksperimen pengaruh jumlah epoch.
    
    Pengujian: 50, 100, 200
    Analisis: Overfitting, waktu training
    """
    print("\n" + "=" * 70)
    print("EKSPERIMEN 7: PENGARUH JUMLAH EPOCH")
    print("=" * 70)
    
    epoch_values = [50, 100, 200]
    
    results = {}
    histories = {}
    
    for ep in epoch_values:
        config = {**DEFAULT_CONFIG, 'epochs': ep}
        name = f"Epochs: {ep}"
        result = run_single_experiment(X_train, X_test, y_train, y_test, 
                                       config, name)
        results[f"ep_{ep}"] = result
        histories[f"epochs={ep}"] = result['history']
    
    # Visualisasi
    plot_multiple_histories(histories, title="Pengaruh Jumlah Epoch",
                          save_path=os.path.join(OUTPUT_DIR, 'exp7_epochs_comparison.png'))
    
    ep_acc = {k: v['eval']['accuracy'] for k, v in results.items()}
    plot_comparison(ep_acc, metric='Accuracy',
                   title="Perbandingan Jumlah Epoch - Test Accuracy",
                   save_path=os.path.join(OUTPUT_DIR, 'exp7_epochs_accuracy.png'))
    
    return results


# ==================== EKSPERIMEN 8: FUNGSI LOSS ====================
def experiment_loss_function(X_train, X_test, y_train, y_test):
    """
    Eksperimen pengaruh fungsi loss.
    
    Pengujian: MSE vs Categorical Cross Entropy
    """
    print("\n" + "=" * 70)
    print("EKSPERIMEN 8: PENGARUH FUNGSI LOSS")
    print("=" * 70)
    
    loss_configs = {
        'MSE': {'loss': 'mse', 'output_activation': 'sigmoid'},
        'Categorical CE': {'loss': 'categorical_crossentropy', 
                          'output_activation': 'softmax'},
    }
    
    results = {}
    histories = {}
    
    for name, loss_config in loss_configs.items():
        config = {**DEFAULT_CONFIG, **loss_config}
        result = run_single_experiment(X_train, X_test, y_train, y_test, 
                                       config, f"Loss: {name}")
        results[name] = result
        histories[name] = result['history']
    
    # Visualisasi
    plot_multiple_histories(histories, title="Pengaruh Fungsi Loss",
                          save_path=os.path.join(OUTPUT_DIR, 'exp8_loss_comparison.png'))
    
    loss_acc = {k: v['eval']['accuracy'] for k, v in results.items()}
    plot_comparison(loss_acc, metric='Accuracy',
                   title="Perbandingan Fungsi Loss - Test Accuracy",
                   save_path=os.path.join(OUTPUT_DIR, 'exp8_loss_accuracy.png'))
    
    return results


# ==================== MAIN ====================
def main():
    """
    Menjalankan seluruh eksperimen from scratch.
    """
    print("=" * 70)
    print("TUGAS BESAR: NEURAL NETWORK FROM SCRATCH")
    print("Mata Kuliah Kecerdasan Buatan - Universitas Darunnajah 2025/2026")
    print("=" * 70)
    
    # Buat direktori output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    all_results = {}
    
    # Jalankan semua eksperimen
    print("\n\n>>> MEMULAI EKSPERIMEN <<<\n")
    
    with Timer("Eksperimen 1 - Arsitektur"):
        all_results['architecture'] = experiment_architecture(
            X_train, X_test, y_train, y_test)
    
    with Timer("Eksperimen 2 - Inisialisasi Bobot"):
        all_results['weight_init'] = experiment_weight_init(
            X_train, X_test, y_train, y_test)
    
    with Timer("Eksperimen 3 - Fungsi Aktivasi"):
        all_results['activation'] = experiment_activation(
            X_train, X_test, y_train, y_test)
    
    with Timer("Eksperimen 4 - Learning Rate"):
        all_results['learning_rate'] = experiment_learning_rate(
            X_train, X_test, y_train, y_test)
    
    with Timer("Eksperimen 5 - Metode Optimasi"):
        all_results['optimizer'] = experiment_optimizer(
            X_train, X_test, y_train, y_test)
    
    with Timer("Eksperimen 6 - Batch Size"):
        all_results['batch_size'] = experiment_batch_size(
            X_train, X_test, y_train, y_test)
    
    with Timer("Eksperimen 7 - Jumlah Epoch"):
        all_results['epochs'] = experiment_epochs(
            X_train, X_test, y_train, y_test)
    
    with Timer("Eksperimen 8 - Fungsi Loss"):
        all_results['loss_function'] = experiment_loss_function(
            X_train, X_test, y_train, y_test)
    
    # ==================== RINGKASAN HASIL ====================
    print("\n\n" + "=" * 70)
    print("RINGKASAN HASIL EKSPERIMEN FROM SCRATCH")
    print("=" * 70)
    
    for exp_name, exp_results in all_results.items():
        print(f"\n--- {exp_name.upper()} ---")
        for sub_name, result in exp_results.items():
            acc = result['eval']['accuracy']
            loss = result['eval']['loss']
            print(f"  {sub_name}: Accuracy={acc:.4f}, Loss={loss:.4f}")
    
    print(f"\nSemua hasil visualisasi disimpan di: {OUTPUT_DIR}")
    print("Eksperimen selesai!")


if __name__ == '__main__':
    main()
