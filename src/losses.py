"""
Modul Fungsi Loss dan Turunannya
=================================
Implementasi fungsi loss yang digunakan dalam Neural Network:
- MSE (Mean Squared Error)
- Binary Cross Entropy
- Categorical Cross Entropy

Setiap fungsi loss memiliki method forward (menghitung loss)
dan backward (menghitung gradient untuk backpropagation).
"""

import numpy as np


class MSE:
    """
    Mean Squared Error (MSE) Loss
    
    Formula: L = (1/n) * Σ (y_pred - y_true)^2
    Turunan: dL/dy_pred = (2/n) * (y_pred - y_true)
    
    Digunakan untuk: Regresi, bisa juga untuk klasifikasi (kurang optimal)
    """
    
    @staticmethod
    def forward(y_pred, y_true):
        """
        Menghitung MSE loss.
        
        Parameters:
            y_pred (np.ndarray): Prediksi model (shape: batch_size x output_size)
            y_true (np.ndarray): Label sebenarnya (shape: batch_size x output_size)
            
        Returns:
            float: Nilai rata-rata MSE loss
        """
        return np.mean((y_pred - y_true) ** 2)
    
    @staticmethod
    def backward(y_pred, y_true):
        """
        Menghitung gradient MSE.
        
        Parameters:
            y_pred (np.ndarray): Prediksi model
            y_true (np.ndarray): Label sebenarnya
            
        Returns:
            np.ndarray: Gradient loss terhadap prediksi
        """
        n = y_true.shape[0]
        return 2 * (y_pred - y_true) / n


class BinaryCrossEntropy:
    """
    Binary Cross Entropy Loss
    
    Formula: L = -(1/n) * Σ [y*log(p) + (1-y)*log(1-p)]
    Turunan: dL/dp = -(y/p - (1-y)/(1-p)) / n
    
    Digunakan untuk: Binary classification (2 kelas)
    """
    
    @staticmethod
    def forward(y_pred, y_true):
        """
        Menghitung Binary Cross Entropy loss.
        Menggunakan clip untuk menghindari log(0).
        
        Parameters:
            y_pred (np.ndarray): Prediksi model (probabilitas 0-1)
            y_true (np.ndarray): Label sebenarnya (0 atau 1)
            
        Returns:
            float: Nilai rata-rata BCE loss
        """
        epsilon = 1e-15  # Untuk numerical stability
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(loss)
    
    @staticmethod
    def backward(y_pred, y_true):
        """
        Menghitung gradient Binary Cross Entropy.
        
        Parameters:
            y_pred (np.ndarray): Prediksi model
            y_true (np.ndarray): Label sebenarnya
            
        Returns:
            np.ndarray: Gradient loss terhadap prediksi
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        n = y_true.shape[0]
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / n


class CategoricalCrossEntropy:
    """
    Categorical Cross Entropy Loss
    
    Formula: L = -(1/n) * Σ Σ y_true * log(y_pred)
    Turunan (dengan Softmax): dL/dz = y_pred - y_true
    
    Digunakan untuk: Multi-class classification (>2 kelas)
    Catatan: Biasanya digunakan bersama softmax pada output layer.
             Gradient digabungkan untuk efisiensi numerik.
    """
    
    @staticmethod
    def forward(y_pred, y_true):
        """
        Menghitung Categorical Cross Entropy loss.
        
        Parameters:
            y_pred (np.ndarray): Prediksi model (distribusi probabilitas)
                                 Shape: (batch_size, num_classes)
            y_true (np.ndarray): Label one-hot encoded
                                 Shape: (batch_size, num_classes)
            
        Returns:
            float: Nilai rata-rata CCE loss
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred), axis=1)
        return np.mean(loss)
    
    @staticmethod
    def backward(y_pred, y_true):
        """
        Menghitung gradient Categorical Cross Entropy (digabung dengan Softmax).
        
        Ketika digunakan bersama softmax, gradient menjadi sederhana:
        dL/dz = y_pred - y_true
        
        Parameters:
            y_pred (np.ndarray): Prediksi model (output softmax)
            y_true (np.ndarray): Label one-hot encoded
            
        Returns:
            np.ndarray: Gradient loss (y_pred - y_true) / batch_size
        """
        n = y_true.shape[0]
        return (y_pred - y_true) / n


def get_loss(name):
    """
    Factory function untuk mendapatkan objek loss berdasarkan nama.
    
    Parameters:
        name (str): Nama fungsi loss ('mse', 'binary_crossentropy', 'categorical_crossentropy')
        
    Returns:
        object: Instance dari kelas loss yang sesuai
        
    Raises:
        ValueError: Jika nama loss tidak dikenali
    """
    losses = {
        'mse': MSE(),
        'binary_crossentropy': BinaryCrossEntropy(),
        'categorical_crossentropy': CategoricalCrossEntropy()
    }
    
    if name.lower() not in losses:
        raise ValueError(
            f"Fungsi loss '{name}' tidak dikenali. "
            f"Pilihan yang tersedia: {list(losses.keys())}"
        )
    
    return losses[name.lower()]
