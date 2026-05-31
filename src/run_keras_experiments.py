"""
Script Eksperimen Neural Network Menggunakan Keras
====================================================
Menjalankan eksperimen yang sama menggunakan Keras (TensorFlow)
untuk perbandingan dengan implementasi from scratch.
"""

import numpy as np
import sys
import os
import time

# Tambahkan src ke path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras_model import (build_keras_model, train_keras_model, 
                         evaluate_keras_model, keras_history_to_dict)
from data_loader import (load_dataset, normalize, train_test_split, 
                         CLASS_NAMES, download_dataset, apply_normalization)
from visualizations import (plot_training_history, plot_confusion_matrix, 
                           plot_comparison, plot_multiple_histories)
from utils import set_seed, Timer


# ==================== KONFIGURASI DEFAULT ====================
DEFAULT_CONFIG = {
    'seed': 42,
    'test_size': 0.2,
    'normalization': 'standard',
    'layer_sizes': [16, 64, 32, 7],
    'activation': 'relu',
    'output_activation': 'softmax',
    'init_method': 'he_normal',
    'loss': 'categorical_crossentropy',
    'optimizer': 'sgd',
    'learning_rate': 0.01,
    'batch_size': 32,
    'epochs': 100,
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          'results', 'keras')


def load_and_prepare_data(data_path=None):
    """Memuat dan mempersiapkan dataset (sama dengan scratch)."""
    if data_path is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'data')
        
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
                    "Tidak dapat menemukan dataset."
                )
    
    X, y_encoded, y_onehot = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=DEFAULT_CONFIG['test_size'], 
        seed=DEFAULT_CONFIG['seed']
    )
    
    X_train, norm_params = normalize(X_train, method=DEFAULT_CONFIG['normalization'])
    X_test = apply_normalization(X_test, norm_params)
    
    print(f"\nData Training: {X_train.shape[0]} sampel")
    print(f"Data Testing:  {X_test.shape[0]} sampel")
    
    return X_train, X_test, y_train, y_test


def run_keras_experiment(X_train, X_test, y_train, y_test, config, exp_name=""):
    """
    Menjalankan satu eksperimen Keras.
    """
    print(f"\n{'='*60}")
    print(f"EKSPERIMEN KERAS: {exp_name}")
    print(f"{'='*60}")
    
    import tensorflow as tf
    tf.random.set_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))
    
    model = build_keras_model(
        layer_sizes=config['layer_sizes'],
        activations=config['activation'],
        output_activation=config.get('output_activation', 'softmax'),
        init_method=config['init_method'],
        loss=config['loss'],
        optimizer=config['optimizer'],
        learning_rate=config['learning_rate']
    )
    
    model.summary()
    
    keras_history = train_keras_model(
        model, X_train, y_train, X_test, y_test,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=1
    )
    
    eval_results = evaluate_keras_model(model, X_test, y_test, CLASS_NAMES)
    history = keras_history_to_dict(keras_history)
    
    return {
        'history': history,
        'eval': eval_results,
        'config': config,
        'name': exp_name
    }


# ==================== EKSPERIMEN KERAS ====================

def experiment_architecture_keras(X_train, X_test, y_train, y_test):
    """Eksperimen arsitektur menggunakan Keras."""
    print("\n" + "=" * 70)
    print("EKSPERIMEN KERAS 1: PENGARUH ARSITEKTUR")
    print("=" * 70)
    
    depth_configs = {
        '1 Hidden [64]': [16, 64, 7],
        '2 Hidden [64,32]': [16, 64, 32, 7],
        '3 Hidden [128,64,32]': [16, 128, 64, 32, 7],
    }
    
    results = {}
    histories = {}
    
    for name, layer_sizes in depth_configs.items():
        config = {**DEFAULT_CONFIG, 'layer_sizes': layer_sizes}
        result = run_keras_experiment(X_train, X_test, y_train, y_test, 
                                      config, name)
        results[name] = result
        histories[name] = result['history']
    
    plot_multiple_histories(histories, title="Keras - Pengaruh Arsitektur",
                          save_path=os.path.join(OUTPUT_DIR, 'keras_exp1_arch.png'))
    
    acc = {k: v['eval']['accuracy'] for k, v in results.items()}
    plot_comparison(acc, title="Keras - Arsitektur Accuracy",
                   save_path=os.path.join(OUTPUT_DIR, 'keras_exp1_arch_acc.png'))
    
    return results


def experiment_activation_keras(X_train, X_test, y_train, y_test):
    """Eksperimen fungsi aktivasi menggunakan Keras."""
    print("\n" + "=" * 70)
    print("EKSPERIMEN KERAS 2: PENGARUH FUNGSI AKTIVASI")
    print("=" * 70)
    
    activations = ['sigmoid', 'relu', 'tanh']
    results = {}
    histories = {}
    
    for act in activations:
        init = 'he_normal' if act == 'relu' else 'glorot_uniform'
        config = {**DEFAULT_CONFIG, 'activation': act, 'init_method': init}
        result = run_keras_experiment(X_train, X_test, y_train, y_test, 
                                      config, f"Activation: {act}")
        results[act] = result
        histories[act] = result['history']
    
    plot_multiple_histories(histories, title="Keras - Pengaruh Aktivasi",
                          save_path=os.path.join(OUTPUT_DIR, 'keras_exp2_activation.png'))
    
    acc = {k: v['eval']['accuracy'] for k, v in results.items()}
    plot_comparison(acc, title="Keras - Aktivasi Accuracy",
                   save_path=os.path.join(OUTPUT_DIR, 'keras_exp2_activation_acc.png'))
    
    return results


def experiment_optimizer_keras(X_train, X_test, y_train, y_test):
    """Eksperimen metode optimasi menggunakan Keras."""
    print("\n" + "=" * 70)
    print("EKSPERIMEN KERAS 3: PENGARUH OPTIMIZER")
    print("=" * 70)
    
    optimizer_configs = {
        'SGD': {'optimizer': 'sgd', 'learning_rate': 0.01},
        'Momentum': {'optimizer': 'momentum', 'learning_rate': 0.01},
        'Adam': {'optimizer': 'adam', 'learning_rate': 0.001},
    }
    
    results = {}
    histories = {}
    
    for name, opt_config in optimizer_configs.items():
        config = {**DEFAULT_CONFIG, **opt_config}
        result = run_keras_experiment(X_train, X_test, y_train, y_test, 
                                      config, f"Optimizer: {name}")
        results[name] = result
        histories[name] = result['history']
    
    plot_multiple_histories(histories, title="Keras - Pengaruh Optimizer",
                          save_path=os.path.join(OUTPUT_DIR, 'keras_exp3_optimizer.png'))
    
    acc = {k: v['eval']['accuracy'] for k, v in results.items()}
    plot_comparison(acc, title="Keras - Optimizer Accuracy",
                   save_path=os.path.join(OUTPUT_DIR, 'keras_exp3_optimizer_acc.png'))
    
    return results


def experiment_learning_rate_keras(X_train, X_test, y_train, y_test):
    """Eksperimen learning rate menggunakan Keras."""
    print("\n" + "=" * 70)
    print("EKSPERIMEN KERAS 4: PENGARUH LEARNING RATE")
    print("=" * 70)
    
    learning_rates = [0.1, 0.01, 0.001]
    results = {}
    histories = {}
    
    for lr in learning_rates:
        config = {**DEFAULT_CONFIG, 'learning_rate': lr}
        result = run_keras_experiment(X_train, X_test, y_train, y_test, 
                                      config, f"LR: {lr}")
        results[f"lr_{lr}"] = result
        histories[f"lr={lr}"] = result['history']
    
    plot_multiple_histories(histories, title="Keras - Pengaruh Learning Rate",
                          save_path=os.path.join(OUTPUT_DIR, 'keras_exp4_lr.png'))
    
    acc = {k: v['eval']['accuracy'] for k, v in results.items()}
    plot_comparison(acc, title="Keras - Learning Rate Accuracy",
                   save_path=os.path.join(OUTPUT_DIR, 'keras_exp4_lr_acc.png'))
    
    return results


def main():
    """Menjalankan seluruh eksperimen Keras."""
    print("=" * 70)
    print("TUGAS BESAR: NEURAL NETWORK - IMPLEMENTASI KERAS")
    print("Mata Kuliah Kecerdasan Buatan - Universitas Darunnajah 2025/2026")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    all_results = {}
    
    with Timer("Keras Exp 1 - Arsitektur"):
        all_results['architecture'] = experiment_architecture_keras(
            X_train, X_test, y_train, y_test)
    
    with Timer("Keras Exp 2 - Aktivasi"):
        all_results['activation'] = experiment_activation_keras(
            X_train, X_test, y_train, y_test)
    
    with Timer("Keras Exp 3 - Optimizer"):
        all_results['optimizer'] = experiment_optimizer_keras(
            X_train, X_test, y_train, y_test)
    
    with Timer("Keras Exp 4 - Learning Rate"):
        all_results['learning_rate'] = experiment_learning_rate_keras(
            X_train, X_test, y_train, y_test)
    
    # ==================== RINGKASAN ====================
    print("\n\n" + "=" * 70)
    print("RINGKASAN HASIL EKSPERIMEN KERAS")
    print("=" * 70)
    
    for exp_name, exp_results in all_results.items():
        print(f"\n--- {exp_name.upper()} ---")
        for sub_name, result in exp_results.items():
            acc = result['eval']['accuracy']
            loss = result['eval']['loss']
            print(f"  {sub_name}: Accuracy={acc:.4f}, Loss={loss:.4f}")
    
    print(f"\nSemua hasil visualisasi disimpan di: {OUTPUT_DIR}")
    print("Eksperimen Keras selesai!")


if __name__ == '__main__':
    main()
