"""
Neural Network From Scratch
============================
Implementasi lengkap Artificial Neural Network (ANN) menggunakan NumPy.

Kelas utama:
- DenseLayer: Fully Connected Layer dengan forward & backward propagation
- NeuralNetwork: Model ANN lengkap dengan training, prediksi, dan evaluasi

Fitur:
- Arsitektur fleksibel (jumlah layer dan neuron dapat dikonfigurasi)
- Berbagai fungsi aktivasi (Sigmoid, ReLU, Tanh, Softmax)
- Berbagai metode inisialisasi bobot (Zero, Random, Xavier, He)
- Berbagai fungsi loss (MSE, BCE, CCE)
- Mini-batch Gradient Descent, Momentum, dan Adam optimizer
- Tracking metrik training dan testing per epoch
"""

import numpy as np
import time

from activations import get_activation, ReLU
from losses import get_loss
from initializers import get_initializer
from optimizers import get_optimizer
from metrics import accuracy, confusion_matrix, precision_recall_f1, classification_report
from data_loader import create_mini_batches


class DenseLayer:
    """
    Fully Connected Layer (Dense Layer)
    
    Layer dasar dalam Neural Network yang menghubungkan setiap neuron
    di input ke setiap neuron di output.
    
    Operasi forward:
        z = X @ W + b          (linear transformation)
        a = activation(z)      (non-linear activation)
    
    Operasi backward:
        dz = da * activation'(z)    (gradient melalui aktivasi)
        dW = input.T @ dz           (gradient bobot)
        db = sum(dz)                (gradient bias)
        dX = dz @ W.T              (gradient input untuk layer sebelumnya)
    
    Attributes:
        weights (np.ndarray): Matriks bobot (input_size, output_size)
        biases (np.ndarray): Vektor bias (1, output_size)
        activation: Objek fungsi aktivasi
        activation_name (str): Nama fungsi aktivasi
        input (np.ndarray): Cache input untuk backpropagation
        z (np.ndarray): Cache pre-activation untuk backpropagation
        a (np.ndarray): Cache output (post-activation) untuk backpropagation
        dW (np.ndarray): Gradient bobot
        db (np.ndarray): Gradient bias
    """
    
    def __init__(self, input_size, output_size, activation='relu', 
                 init_method='he', seed=42):
        """
        Inisialisasi Dense Layer.
        
        Parameters:
            input_size (int): Jumlah neuron input
            output_size (int): Jumlah neuron output
            activation (str): Nama fungsi aktivasi
            init_method (str): Metode inisialisasi bobot
            seed (int): Random seed untuk reproducibility
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        self.activation = get_activation(activation)
        
        # Inisialisasi bobot
        initializer = get_initializer(init_method)
        self.weights = initializer((input_size, output_size), seed=seed)
        self.biases = np.zeros((1, output_size))
        
        # Cache untuk backpropagation
        self.input = None
        self.z = None  # Pre-activation
        self.a = None  # Post-activation
        
        # Gradients
        self.dW = None
        self.db = None
    
    def forward(self, X):
        """
        Forward propagation melalui layer ini.
        
        Langkah:
        1. Linear transformation: z = X @ W + b
        2. Activation: a = activation(z)
        
        Parameters:
            X (np.ndarray): Input data (batch_size, input_size)
            
        Returns:
            np.ndarray: Output setelah aktivasi (batch_size, output_size)
        """
        self.input = X
        self.z = X @ self.weights + self.biases  # Linear transformation
        self.a = self.activation.forward(self.z)  # Non-linear activation
        return self.a
    
    def backward(self, da, is_output_layer=False):
        """
        Backward propagation melalui layer ini.
        
        Untuk hidden layer:
            dz = da * activation'(z atau a)
            
        Untuk output layer dengan Softmax + CCE:
            dz = da  (gradient sudah dihitung di loss function)
        
        Kemudian:
            dW = input.T @ dz
            db = sum(dz, axis=0)
            dX = dz @ W.T
        
        Parameters:
            da (np.ndarray): Gradient dari layer berikutnya atau loss
            is_output_layer (bool): True jika ini output layer dengan Softmax+CCE
            
        Returns:
            np.ndarray: Gradient input (untuk diteruskan ke layer sebelumnya)
        """
        batch_size = da.shape[0]
        
        if is_output_layer and self.activation_name == 'softmax':
            # Untuk Softmax + CCE, gradient sudah digabung: dz = y_pred - y_true
            dz = da
        elif self.activation_name == 'relu':
            # ReLU: turunan berdasarkan z (pre-activation)
            dz = da * self.activation.backward(self.z)
        else:
            # Sigmoid, Tanh: turunan berdasarkan a (post-activation)
            dz = da * self.activation.backward(self.a)
        
        # Gradient bobot: dW = input.T @ dz
        self.dW = self.input.T @ dz
        
        # Gradient bias: db = sum(dz)
        self.db = np.sum(dz, axis=0, keepdims=True)
        
        # Gradient input: dX = dz @ W.T (untuk layer sebelumnya)
        dX = dz @ self.weights.T
        
        return dX
    
    def get_num_params(self):
        """
        Menghitung jumlah parameter di layer ini.
        
        Returns:
            int: Total parameter (weights + biases)
        """
        return self.weights.size + self.biases.size
    
    def __repr__(self):
        return (f"DenseLayer(in={self.input_size}, out={self.output_size}, "
                f"activation={self.activation_name})")


class NeuralNetwork:
    """
    Artificial Neural Network (ANN) - Implementasi From Scratch
    
    Model Neural Network lengkap dengan kemampuan:
    - Konfigurasi arsitektur fleksibel
    - Forward propagation
    - Backpropagation
    - Training dengan mini-batch gradient descent
    - Prediksi dan evaluasi
    - Tracking history training
    
    Attributes:
        layers (list): List dari DenseLayer
        loss_fn: Objek fungsi loss
        optimizer: Objek optimizer
        history (dict): History training (loss, accuracy per epoch)
    """
    
    def __init__(self):
        """Inisialisasi Neural Network kosong."""
        self.layers = []
        self.loss_fn = None
        self.loss_name = None
        self.optimizer = None
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': []
        }
    
    def add_layer(self, input_size, output_size, activation='relu',
                  init_method='he', seed=42):
        """
        Menambahkan Dense Layer ke network.
        
        Parameters:
            input_size (int): Jumlah neuron input
            output_size (int): Jumlah neuron output
            activation (str): Fungsi aktivasi ('sigmoid', 'relu', 'tanh', 'softmax')
            init_method (str): Metode inisialisasi ('zero', 'random_uniform', 
                              'random_normal', 'xavier', 'he')
            seed (int): Random seed
        """
        layer = DenseLayer(input_size, output_size, activation, init_method, seed)
        self.layers.append(layer)
    
    def build(self, layer_sizes, activations, init_method='he', seed=42):
        """
        Membangun arsitektur neural network secara otomatis.
        
        Parameters:
            layer_sizes (list): List ukuran layer [input, hidden1, ..., output]
                               Contoh: [16, 64, 32, 7]
            activations (list): List nama aktivasi untuk setiap layer
                               Contoh: ['relu', 'relu', 'softmax']
            init_method (str): Metode inisialisasi bobot
            seed (int): Random seed
        """
        self.layers = []
        
        for i in range(len(layer_sizes) - 1):
            self.add_layer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i],
                init_method=init_method,
                seed=seed + i  # Seed berbeda untuk setiap layer
            )
    
    def compile(self, loss='categorical_crossentropy', optimizer='minibatch_gd',
                learning_rate=0.01, **optimizer_kwargs):
        """
        Mengkompilasi model dengan loss function dan optimizer.
        
        Parameters:
            loss (str): Nama fungsi loss
            optimizer (str): Nama optimizer
            learning_rate (float): Laju pembelajaran
            **optimizer_kwargs: Parameter tambahan optimizer
        """
        self.loss_fn = get_loss(loss)
        self.loss_name = loss
        self.optimizer = get_optimizer(optimizer, learning_rate, **optimizer_kwargs)
        
        # Reset history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': []
        }
    
    def forward(self, X):
        """
        Forward propagation melalui seluruh network.
        
        Data mengalir dari input layer → hidden layers → output layer.
        Setiap layer melakukan: z = X@W + b, lalu a = activation(z)
        
        Parameters:
            X (np.ndarray): Input data (batch_size, input_features)
            
        Returns:
            np.ndarray: Output prediksi dari layer terakhir
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, y_pred, y_true):
        """
        Backpropagation melalui seluruh network.
        
        Gradient loss dihitung dan dipropagasikan dari output layer
        ke input layer menggunakan chain rule.
        
        Langkah:
        1. Hitung gradient loss terhadap output: dL/da
        2. Untuk setiap layer (dari belakang ke depan):
           - Hitung dz, dW, db
           - Propagasikan gradient ke layer sebelumnya
        
        Parameters:
            y_pred (np.ndarray): Prediksi model
            y_true (np.ndarray): Label sebenarnya
        """
        # Gradient dari loss function
        grad = self.loss_fn.backward(y_pred, y_true)
        
        # Backpropagation dari layer terakhir ke pertama
        for i in reversed(range(len(self.layers))):
            is_output = (i == len(self.layers) - 1)
            grad = self.layers[i].backward(grad, is_output_layer=is_output)
    
    def train(self, X_train, y_train, X_test=None, y_test=None,
              epochs=100, batch_size=32, verbose=True, seed=42):
        """
        Training model menggunakan mini-batch gradient descent.
        
        Proses training per epoch:
        1. Bagi data menjadi mini-batches
        2. Untuk setiap batch:
           a. Forward propagation
           b. Hitung loss
           c. Backpropagation
           d. Update bobot (optimizer)
        3. Hitung metrik epoch (loss & accuracy)
        
        Parameters:
            X_train (np.ndarray): Data training
            y_train (np.ndarray): Label training (one-hot)
            X_test (np.ndarray): Data testing (opsional)
            y_test (np.ndarray): Label testing (opsional)
            epochs (int): Jumlah epoch
            batch_size (int): Ukuran mini-batch
            verbose (bool): Tampilkan progress training
            seed (int): Random seed
            
        Returns:
            dict: Training history
        """
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Buat mini-batches (shuffle setiap epoch)
            mini_batches = create_mini_batches(X_train, y_train, batch_size, 
                                                seed=seed + epoch)
            
            epoch_loss = 0
            num_batches = len(mini_batches)
            
            for X_batch, y_batch in mini_batches:
                # === FORWARD PROPAGATION ===
                y_pred = self.forward(X_batch)
                
                # === HITUNG LOSS ===
                batch_loss = self.loss_fn.forward(y_pred, y_batch)
                epoch_loss += batch_loss
                
                # === BACKPROPAGATION ===
                self.backward(y_pred, y_batch)
                
                # === UPDATE BOBOT ===
                self.optimizer.update(self.layers)
            
            # Rata-rata loss per epoch
            avg_loss = epoch_loss / num_batches
            
            # Hitung accuracy training
            train_pred = self.forward(X_train)
            train_acc = accuracy(train_pred, y_train)
            train_loss = self.loss_fn.forward(train_pred, y_train)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            
            # Hitung metrik testing jika data test tersedia
            if X_test is not None and y_test is not None:
                test_pred = self.forward(X_test)
                test_loss = self.loss_fn.forward(test_pred, y_test)
                test_acc = accuracy(test_pred, y_test)
                
                self.history['test_loss'].append(test_loss)
                self.history['test_accuracy'].append(test_acc)
            
            # Print progress
            if verbose and (epoch + 1) % max(1, epochs // 20) == 0:
                epoch_time = time.time() - epoch_start
                msg = (f"Epoch {epoch+1}/{epochs} | "
                       f"Train Loss: {train_loss:.4f} | "
                       f"Train Acc: {train_acc:.4f}")
                
                if X_test is not None:
                    msg += (f" | Test Loss: {test_loss:.4f} | "
                           f"Test Acc: {test_acc:.4f}")
                
                msg += f" | Time: {epoch_time:.2f}s"
                print(msg)
        
        total_time = time.time() - start_time
        if verbose:
            print(f"\nTraining selesai dalam {total_time:.2f} detik")
        
        return self.history
    
    def predict(self, X):
        """
        Melakukan prediksi pada data baru.
        
        Parameters:
            X (np.ndarray): Data input
            
        Returns:
            np.ndarray: Prediksi (probabilitas untuk setiap kelas)
        """
        return self.forward(X)
    
    def predict_classes(self, X):
        """
        Melakukan prediksi dan mengembalikan indeks kelas.
        
        Parameters:
            X (np.ndarray): Data input
            
        Returns:
            np.ndarray: Indeks kelas prediksi
        """
        probs = self.predict(X)
        return np.argmax(probs, axis=1)
    
    def evaluate(self, X, y, class_names=None):
        """
        Evaluasi model pada data tertentu.
        
        Parameters:
            X (np.ndarray): Data input
            y (np.ndarray): Label sebenarnya (one-hot)
            class_names (list): Nama kelas (opsional)
            
        Returns:
            dict: Dictionary berisi semua metrik evaluasi
        """
        y_pred = self.predict(X)
        
        loss = self.loss_fn.forward(y_pred, y)
        acc = accuracy(y_pred, y)
        cm = confusion_matrix(y_pred, y)
        prf = precision_recall_f1(y_pred, y)
        
        print(f"\n{'='*60}")
        print(f"EVALUASI MODEL")
        print(f"{'='*60}")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"\nClassification Report:")
        print(classification_report(y_pred, y, class_names))
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return {
            'loss': loss,
            'accuracy': acc,
            'confusion_matrix': cm,
            'precision': prf['macro_precision'],
            'recall': prf['macro_recall'],
            'f1_score': prf['macro_f1'],
            'precision_per_class': prf['precision_per_class'],
            'recall_per_class': prf['recall_per_class'],
            'f1_per_class': prf['f1_per_class']
        }
    
    def summary(self):
        """Menampilkan ringkasan arsitektur model."""
        print("=" * 65)
        print("NEURAL NETWORK SUMMARY")
        print("=" * 65)
        print(f"{'Layer':<8} {'Input':<10} {'Output':<10} {'Activation':<12} {'Params':<10}")
        print("-" * 65)
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            n_params = layer.get_num_params()
            total_params += n_params
            layer_type = "Output" if i == len(self.layers) - 1 else f"Hidden {i+1}"
            print(f"{layer_type:<8} {layer.input_size:<10} {layer.output_size:<10} "
                  f"{layer.activation_name:<12} {n_params:<10}")
        
        print("-" * 65)
        print(f"Total Parameters: {total_params}")
        if self.loss_fn:
            print(f"Loss Function: {self.loss_name}")
        if self.optimizer:
            print(f"Optimizer: {self.optimizer.__class__.__name__} "
                  f"(lr={self.optimizer.learning_rate})")
        print("=" * 65)


def build_model(layer_sizes, activations='relu', output_activation='softmax',
                init_method='he', loss='categorical_crossentropy',
                optimizer='minibatch_gd', learning_rate=0.01, seed=42,
                **optimizer_kwargs):
    """
    Helper function untuk membangun, mengkompilasi, dan mengembalikan model.
    
    Parameters:
        layer_sizes (list): Ukuran layer [input, hidden1, ..., output]
        activations (str or list): Aktivasi untuk hidden layers
        output_activation (str): Aktivasi untuk output layer
        init_method (str): Metode inisialisasi bobot
        loss (str): Fungsi loss
        optimizer (str): Nama optimizer
        learning_rate (float): Laju pembelajaran
        seed (int): Random seed
        
    Returns:
        NeuralNetwork: Model yang sudah dikompilasi
        
    Contoh:
        model = build_model(
            layer_sizes=[16, 64, 32, 7],
            activations='relu',
            output_activation='softmax',
            init_method='he',
            loss='categorical_crossentropy',
            optimizer='adam',
            learning_rate=0.001
        )
    """
    model = NeuralNetwork()
    
    # Buat list aktivasi
    num_hidden = len(layer_sizes) - 2
    
    if isinstance(activations, str):
        act_list = [activations] * num_hidden + [output_activation]
    else:
        act_list = list(activations) + [output_activation]
    
    # Build arsitektur
    model.build(layer_sizes, act_list, init_method, seed)
    
    # Compile model
    model.compile(loss=loss, optimizer=optimizer, 
                  learning_rate=learning_rate, **optimizer_kwargs)
    
    return model
