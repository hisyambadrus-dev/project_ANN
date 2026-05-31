"""
Modul Inisialisasi Bobot
========================
Implementasi metode inisialisasi bobot untuk Neural Network:
- Zero Initialization
- Random Uniform
- Random Normal
- Xavier Initialization (Glorot)
- He Initialization

Pemilihan metode inisialisasi yang tepat sangat penting untuk
konvergensi training Neural Network.
"""

import numpy as np


def zero_init(shape, seed=None):
    """
    Zero Initialization
    
    Semua bobot diinisialisasi dengan nilai 0.
    
    Catatan: Tidak disarankan untuk hidden layers karena menyebabkan
    symmetry problem — semua neuron akan belajar hal yang sama.
    
    Parameters:
        shape (tuple): Dimensi matriks bobot (input_size, output_size)
        seed (int): Random seed (tidak digunakan, untuk konsistensi interface)
        
    Returns:
        np.ndarray: Matriks bobot berisi nol
    """
    return np.zeros(shape)


def random_uniform_init(shape, seed=42):
    """
    Random Uniform Initialization
    
    Bobot diinisialisasi secara acak dari distribusi uniform [-0.5, 0.5].
    
    Parameters:
        shape (tuple): Dimensi matriks bobot (input_size, output_size)
        seed (int): Random seed untuk reproducibility
        
    Returns:
        np.ndarray: Matriks bobot dari distribusi uniform
    """
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.5, 0.5, size=shape)


def random_normal_init(shape, seed=42):
    """
    Random Normal Initialization
    
    Bobot diinisialisasi secara acak dari distribusi normal (mean=0, std=0.01).
    
    Parameters:
        shape (tuple): Dimensi matriks bobot (input_size, output_size)
        seed (int): Random seed untuk reproducibility
        
    Returns:
        np.ndarray: Matriks bobot dari distribusi normal
    """
    rng = np.random.RandomState(seed)
    return rng.normal(0, 0.01, size=shape)


def xavier_init(shape, seed=42):
    """
    Xavier Initialization (Glorot Initialization)
    
    Formula: W ~ U[-√(6/(n_in + n_out)), √(6/(n_in + n_out))]
    
    Dirancang untuk menjaga variance aktivasi tetap konstan
    di sepanjang forward dan backward pass.
    
    Cocok digunakan dengan: Sigmoid, Tanh
    
    Parameters:
        shape (tuple): Dimensi matriks bobot (n_in, n_out)
        seed (int): Random seed untuk reproducibility
        
    Returns:
        np.ndarray: Matriks bobot dengan Xavier initialization
    """
    rng = np.random.RandomState(seed)
    n_in, n_out = shape
    limit = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-limit, limit, size=shape)


def he_init(shape, seed=42):
    """
    He Initialization (Kaiming Initialization)
    
    Formula: W ~ N(0, √(2/n_in))
    
    Dirancang khusus untuk fungsi aktivasi ReLU yang hanya
    mengaktifkan setengah dari neuron (karena output negatif = 0).
    
    Cocok digunakan dengan: ReLU dan variannya
    
    Parameters:
        shape (tuple): Dimensi matriks bobot (n_in, n_out)
        seed (int): Random seed untuk reproducibility
        
    Returns:
        np.ndarray: Matriks bobot dengan He initialization
    """
    rng = np.random.RandomState(seed)
    n_in = shape[0]
    std = np.sqrt(2.0 / n_in)
    return rng.normal(0, std, size=shape)


def get_initializer(name):
    """
    Factory function untuk mendapatkan fungsi inisialisasi berdasarkan nama.
    
    Parameters:
        name (str): Nama metode inisialisasi
                     ('zero', 'random_uniform', 'random_normal', 'xavier', 'he')
        
    Returns:
        function: Fungsi inisialisasi yang sesuai
        
    Raises:
        ValueError: Jika nama inisialisasi tidak dikenali
    """
    initializers = {
        'zero': zero_init,
        'random_uniform': random_uniform_init,
        'random_normal': random_normal_init,
        'xavier': xavier_init,
        'he': he_init
    }
    
    if name.lower() not in initializers:
        raise ValueError(
            f"Metode inisialisasi '{name}' tidak dikenali. "
            f"Pilihan yang tersedia: {list(initializers.keys())}"
        )
    
    return initializers[name.lower()]
