"""
Modul Optimizer (Algoritma Optimasi)
=====================================
Implementasi algoritma optimasi untuk update bobot Neural Network:
- Mini-batch Gradient Descent (wajib)
- Momentum (bonus)
- Adam (bonus)
"""

import numpy as np


class MiniBatchGD:
    """
    Mini-batch Gradient Descent Optimizer
    
    Update rule: W = W - lr * dW
                 b = b - lr * db
    
    Mini-batch GD membagi dataset menjadi batch-batch kecil dan
    melakukan update bobot setelah setiap batch diproses.
    
    Attributes:
        learning_rate (float): Laju pembelajaran
    """
    
    def __init__(self, learning_rate=0.01):
        """
        Parameters:
            learning_rate (float): Laju pembelajaran (default: 0.01)
        """
        self.learning_rate = learning_rate
    
    def update(self, layers):
        """
        Update bobot dan bias semua layer.
        
        Parameters:
            layers (list): List dari objek Layer yang memiliki atribut
                          weights, biases, dW, dan db
        """
        for layer in layers:
            if hasattr(layer, 'weights') and layer.dW is not None:
                layer.weights -= self.learning_rate * layer.dW
                layer.biases -= self.learning_rate * layer.db


class Momentum:
    """
    Gradient Descent with Momentum (Bonus)
    
    Menambahkan "momentum" untuk mempercepat konvergensi dan mengurangi
    osilasi pada gradient descent.
    
    Update rule:
        v_W = β * v_W + (1-β) * dW
        v_b = β * v_b + (1-β) * db
        W = W - lr * v_W
        b = b - lr * v_b
    
    Attributes:
        learning_rate (float): Laju pembelajaran
        beta (float): Koefisien momentum (default: 0.9)
        velocities (dict): Menyimpan velocity untuk setiap layer
    """
    
    def __init__(self, learning_rate=0.01, beta=0.9):
        """
        Parameters:
            learning_rate (float): Laju pembelajaran
            beta (float): Koefisien momentum (0 sampai 1)
        """
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocities = {}
    
    def update(self, layers):
        """
        Update bobot dan bias menggunakan momentum.
        
        Parameters:
            layers (list): List dari objek Layer
        """
        for i, layer in enumerate(layers):
            if hasattr(layer, 'weights') and layer.dW is not None:
                # Inisialisasi velocity jika belum ada
                if i not in self.velocities:
                    self.velocities[i] = {
                        'vW': np.zeros_like(layer.weights),
                        'vb': np.zeros_like(layer.biases)
                    }
                
                v = self.velocities[i]
                
                # Update velocity
                v['vW'] = self.beta * v['vW'] + (1 - self.beta) * layer.dW
                v['vb'] = self.beta * v['vb'] + (1 - self.beta) * layer.db
                
                # Update weights
                layer.weights -= self.learning_rate * v['vW']
                layer.biases -= self.learning_rate * v['vb']


class Adam:
    """
    Adam Optimizer (Adaptive Moment Estimation) - Bonus
    
    Menggabungkan keunggulan Momentum dan RMSProp.
    Menggunakan first moment (mean) dan second moment (variance)
    dari gradient untuk adaptive learning rate per parameter.
    
    Update rule:
        m = β1 * m + (1-β1) * g          # First moment
        v = β2 * v + (1-β2) * g^2        # Second moment
        m_hat = m / (1 - β1^t)            # Bias correction
        v_hat = v / (1 - β2^t)            # Bias correction
        W = W - lr * m_hat / (√v_hat + ε)
    
    Attributes:
        learning_rate (float): Laju pembelajaran
        beta1 (float): Decay rate untuk first moment (default: 0.9)
        beta2 (float): Decay rate untuk second moment (default: 0.999)
        epsilon (float): Konstanta kecil untuk numerical stability
        t (int): Timestep counter
        moments (dict): Menyimpan first & second moments
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Parameters:
            learning_rate (float): Laju pembelajaran (default: 0.001)
            beta1 (float): Exponential decay rate untuk first moment
            beta2 (float): Exponential decay rate untuk second moment
            epsilon (float): Konstanta untuk numerical stability
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Timestep
        self.moments = {}
    
    def update(self, layers):
        """
        Update bobot dan bias menggunakan Adam optimizer.
        
        Parameters:
            layers (list): List dari objek Layer
        """
        self.t += 1
        
        for i, layer in enumerate(layers):
            if hasattr(layer, 'weights') and layer.dW is not None:
                # Inisialisasi moments jika belum ada
                if i not in self.moments:
                    self.moments[i] = {
                        'mW': np.zeros_like(layer.weights),
                        'mb': np.zeros_like(layer.biases),
                        'vW': np.zeros_like(layer.weights),
                        'vb': np.zeros_like(layer.biases)
                    }
                
                m = self.moments[i]
                
                # Update first moment (mean of gradients)
                m['mW'] = self.beta1 * m['mW'] + (1 - self.beta1) * layer.dW
                m['mb'] = self.beta1 * m['mb'] + (1 - self.beta1) * layer.db
                
                # Update second moment (mean of squared gradients)
                m['vW'] = self.beta2 * m['vW'] + (1 - self.beta2) * (layer.dW ** 2)
                m['vb'] = self.beta2 * m['vb'] + (1 - self.beta2) * (layer.db ** 2)
                
                # Bias correction
                mW_hat = m['mW'] / (1 - self.beta1 ** self.t)
                mb_hat = m['mb'] / (1 - self.beta1 ** self.t)
                vW_hat = m['vW'] / (1 - self.beta2 ** self.t)
                vb_hat = m['vb'] / (1 - self.beta2 ** self.t)
                
                # Update weights
                layer.weights -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
                layer.biases -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + self.epsilon)


def get_optimizer(name, learning_rate=0.01, **kwargs):
    """
    Factory function untuk mendapatkan optimizer berdasarkan nama.
    
    Parameters:
        name (str): Nama optimizer ('minibatch_gd', 'momentum', 'adam')
        learning_rate (float): Laju pembelajaran
        **kwargs: Parameter tambahan untuk optimizer tertentu
        
    Returns:
        object: Instance dari kelas optimizer yang sesuai
        
    Raises:
        ValueError: Jika nama optimizer tidak dikenali
    """
    if name.lower() in ['minibatch_gd', 'sgd', 'mini_batch_gd']:
        return MiniBatchGD(learning_rate=learning_rate)
    elif name.lower() == 'momentum':
        beta = kwargs.get('beta', 0.9)
        return Momentum(learning_rate=learning_rate, beta=beta)
    elif name.lower() == 'adam':
        return Adam(
            learning_rate=learning_rate,
            beta1=kwargs.get('beta1', 0.9),
            beta2=kwargs.get('beta2', 0.999),
            epsilon=kwargs.get('epsilon', 1e-8)
        )
    else:
        raise ValueError(
            f"Optimizer '{name}' tidak dikenali. "
            f"Pilihan yang tersedia: ['minibatch_gd', 'momentum', 'adam']"
        )
