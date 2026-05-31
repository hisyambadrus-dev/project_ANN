"""
Modul Fungsi Aktivasi dan Turunannya
=====================================
Implementasi fungsi aktivasi yang digunakan dalam Neural Network:
- Sigmoid
- ReLU
- Tanh (Hyperbolic Tangent)
- Softmax

Setiap fungsi aktivasi memiliki method forward (untuk forward propagation)
dan backward (turunan untuk backpropagation).
"""

import numpy as np


class Sigmoid:
    """
    Fungsi Aktivasi Sigmoid
    
    Formula: σ(x) = 1 / (1 + exp(-x))
    Turunan: σ'(x) = σ(x) * (1 - σ(x))
    
    Digunakan untuk: Output layer pada binary classification,
                     hidden layer (jarang digunakan karena vanishing gradient)
    """
    
    @staticmethod
    def forward(z):
        """
        Forward pass sigmoid.
        Menggunakan clip untuk menghindari overflow pada exp.
        
        Parameters:
            z (np.ndarray): Input array
            
        Returns:
            np.ndarray: Output setelah aktivasi sigmoid
        """
        z = np.clip(z, -500, 500)  # Mencegah overflow
        return 1.0 / (1.0 + np.exp(-z))
    
    @staticmethod
    def backward(a):
        """
        Turunan sigmoid: σ'(x) = σ(x) * (1 - σ(x))
        
        Parameters:
            a (np.ndarray): Output dari forward pass (sudah diaktivasi)
            
        Returns:
            np.ndarray: Turunan fungsi aktivasi
        """
        return a * (1 - a)


class ReLU:
    """
    Fungsi Aktivasi ReLU (Rectified Linear Unit)
    
    Formula: f(x) = max(0, x)
    Turunan: f'(x) = 1 jika x > 0, 0 jika x <= 0
    
    Digunakan untuk: Hidden layer (paling populer untuk deep networks)
    Kelebihan: Mengatasi vanishing gradient, komputasi cepat
    """
    
    @staticmethod
    def forward(z):
        """
        Forward pass ReLU.
        
        Parameters:
            z (np.ndarray): Input array
            
        Returns:
            np.ndarray: Output setelah aktivasi ReLU
        """
        return np.maximum(0, z)
    
    @staticmethod
    def backward(z):
        """
        Turunan ReLU: 1 jika z > 0, 0 jika z <= 0
        
        Parameters:
            z (np.ndarray): Input sebelum aktivasi (pre-activation)
            
        Returns:
            np.ndarray: Turunan fungsi aktivasi
        """
        return (z > 0).astype(float)


class Tanh:
    """
    Fungsi Aktivasi Tanh (Hyperbolic Tangent)
    
    Formula: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Turunan: tanh'(x) = 1 - tanh(x)^2
    
    Digunakan untuk: Hidden layer
    Kelebihan: Output centered di 0 (-1 sampai 1)
    """
    
    @staticmethod
    def forward(z):
        """
        Forward pass Tanh.
        
        Parameters:
            z (np.ndarray): Input array
            
        Returns:
            np.ndarray: Output setelah aktivasi Tanh
        """
        return np.tanh(z)
    
    @staticmethod
    def backward(a):
        """
        Turunan Tanh: 1 - tanh(x)^2
        
        Parameters:
            a (np.ndarray): Output dari forward pass (sudah diaktivasi)
            
        Returns:
            np.ndarray: Turunan fungsi aktivasi
        """
        return 1 - a ** 2


class Softmax:
    """
    Fungsi Aktivasi Softmax
    
    Formula: softmax(x_i) = exp(x_i) / Σ exp(x_j)
    
    Digunakan untuk: Output layer pada multi-class classification
    Output: Distribusi probabilitas (semua output berjumlah 1)
    
    Catatan: Turunan softmax digabungkan dengan Categorical Cross Entropy
             untuk efisiensi komputasi pada backpropagation.
    """
    
    @staticmethod
    def forward(z):
        """
        Forward pass Softmax.
        Menggunakan max subtraction untuk numerical stability.
        
        Parameters:
            z (np.ndarray): Input array (shape: batch_size x num_classes)
            
        Returns:
            np.ndarray: Distribusi probabilitas
        """
        # Numerical stability: kurangi max dari setiap baris
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    @staticmethod
    def backward(a):
        """
        Turunan Softmax.
        Catatan: Pada implementasi ini, turunan softmax digabungkan langsung
        dengan loss function (CCE) sehingga gradient = predicted - actual.
        Method ini mengembalikan ones karena gradient sudah dihitung di loss.
        
        Parameters:
            a (np.ndarray): Output dari forward pass
            
        Returns:
            np.ndarray: Ones (gradient dihitung di loss function)
        """
        return np.ones_like(a)


def get_activation(name):
    """
    Factory function untuk mendapatkan objek aktivasi berdasarkan nama.
    
    Parameters:
        name (str): Nama fungsi aktivasi ('sigmoid', 'relu', 'tanh', 'softmax')
        
    Returns:
        object: Instance dari kelas aktivasi yang sesuai
        
    Raises:
        ValueError: Jika nama aktivasi tidak dikenali
    """
    activations = {
        'sigmoid': Sigmoid(),
        'relu': ReLU(),
        'tanh': Tanh(),
        'softmax': Softmax()
    }
    
    if name.lower() not in activations:
        raise ValueError(
            f"Fungsi aktivasi '{name}' tidak dikenali. "
            f"Pilihan yang tersedia: {list(activations.keys())}"
        )
    
    return activations[name.lower()]
