"""
Modul Utilitas
==============
Fungsi-fungsi utilitas pendukung untuk proyek Neural Network.
"""

import numpy as np
import os
import json
import time


def set_seed(seed=42):
    """
    Set random seed untuk reproducibility.
    
    Parameters:
        seed (int): Random seed
    """
    np.random.seed(seed)


def save_results(results, filepath):
    """
    Menyimpan hasil eksperimen ke file JSON.
    
    Parameters:
        results (dict): Dictionary hasil eksperimen
        filepath (str): Path file output
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Konversi numpy types ke Python native types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"Hasil disimpan ke: {filepath}")


def load_results(filepath):
    """
    Memuat hasil eksperimen dari file JSON.
    
    Parameters:
        filepath (str): Path file input
        
    Returns:
        dict: Dictionary hasil eksperimen
    """
    with open(filepath, 'r') as f:
        return json.load(f)


class Timer:
    """
    Context manager untuk mengukur waktu eksekusi.
    
    Usage:
        with Timer("Training"):
            model.train(...)
    """
    
    def __init__(self, name=""):
        self.name = name
        self.elapsed = 0
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"[{self.name}] Waktu: {self.elapsed:.2f} detik")


def print_model_summary(model):
    """
    Menampilkan ringkasan arsitektur model.
    
    Parameters:
        model: Instance NeuralNetwork
    """
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    
    total_params = 0
    
    for i, layer in enumerate(model.layers):
        n_params = layer.weights.shape[0] * layer.weights.shape[1] + layer.biases.shape[1]
        total_params += n_params
        
        print(f"Layer {i+1}: {layer.weights.shape[0]} -> {layer.weights.shape[1]} "
              f"| Activation: {layer.activation_name} "
              f"| Params: {n_params}")
    
    print("-" * 60)
    print(f"Total Parameters: {total_params}")
    print("=" * 60)
