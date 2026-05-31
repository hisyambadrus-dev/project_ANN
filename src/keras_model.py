"""
Implementasi Neural Network Menggunakan Keras (TensorFlow)
==========================================================
Model yang sama dibangun menggunakan framework Keras untuk perbandingan
dengan implementasi from scratch.

Fitur:
- Arsitektur yang fleksibel (sama seperti from scratch)
- Berbagai optimizer, loss function, dan aktivasi
- Training dan evaluasi
- Perbandingan dengan model from scratch
"""

import numpy as np
import os
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, callbacks
from tensorflow.keras.utils import to_categorical


def build_keras_model(layer_sizes, activations='relu', output_activation='softmax',
                      init_method='he_normal', loss='categorical_crossentropy',
                      optimizer='sgd', learning_rate=0.01, metrics=['accuracy']):
    """
    Membangun dan mengkompilasi model Keras.
    
    Parameters:
        layer_sizes (list): Ukuran layer [input, hidden1, ..., output]
        activations (str or list): Aktivasi hidden layers
        output_activation (str): Aktivasi output layer
        init_method (str): Metode inisialisasi (Keras naming convention)
        loss (str): Fungsi loss
        optimizer (str): Nama optimizer
        learning_rate (float): Laju pembelajaran
        metrics (list): Metrik evaluasi
        
    Returns:
        keras.Model: Model yang sudah dikompilasi
    """
    # Mapping inisialisasi ke Keras naming
    init_mapping = {
        'zero': 'zeros',
        'random_uniform': 'random_uniform',
        'random_normal': 'random_normal',
        'xavier': 'glorot_uniform',
        'he': 'he_normal',
        'he_normal': 'he_normal',
        'glorot_uniform': 'glorot_uniform',
        'zeros': 'zeros'
    }
    
    keras_init = init_mapping.get(init_method, init_method)
    
    # Buat model sequential
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))
    
    # Hidden layers
    num_hidden = len(layer_sizes) - 2
    
    if isinstance(activations, str):
        act_list = [activations] * num_hidden
    else:
        act_list = list(activations)
    
    for i in range(num_hidden):
        model.add(layers.Dense(
            layer_sizes[i + 1],
            activation=act_list[i],
            kernel_initializer=keras_init,
            name=f'hidden_{i+1}'
        ))
    
    # Output layer
    model.add(layers.Dense(
        layer_sizes[-1],
        activation=output_activation,
        kernel_initializer=keras_init,
        name='output'
    ))
    
    # Pilih optimizer
    if optimizer.lower() in ['sgd', 'minibatch_gd', 'mini_batch_gd']:
        opt = optimizers.SGD(learning_rate=learning_rate)
    elif optimizer.lower() == 'momentum':
        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer.lower() == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = optimizers.SGD(learning_rate=learning_rate)
    
    # Compile model
    model.compile(
        loss=loss,
        optimizer=opt,
        metrics=metrics
    )
    
    return model


def train_keras_model(model, X_train, y_train, X_test=None, y_test=None,
                      epochs=100, batch_size=32, verbose=1):
    """
    Training model Keras.
    
    Parameters:
        model: Model Keras yang sudah dikompilasi
        X_train, y_train: Data training
        X_test, y_test: Data testing (opsional)
        epochs (int): Jumlah epoch
        batch_size (int): Ukuran batch
        verbose (int): Level verbosity (0=silent, 1=progress bar, 2=one line per epoch)
        
    Returns:
        keras.callbacks.History: History training
    """
    validation_data = None
    if X_test is not None and y_test is not None:
        validation_data = (X_test, y_test)
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        verbose=verbose
    )
    
    elapsed = time.time() - start_time
    print(f"\nTraining Keras selesai dalam {elapsed:.2f} detik")
    
    return history


def evaluate_keras_model(model, X_test, y_test, class_names=None):
    """
    Evaluasi model Keras.
    
    Parameters:
        model: Model Keras
        X_test: Data testing
        y_test: Label testing (one-hot)
        class_names: Nama kelas
        
    Returns:
        dict: Hasil evaluasi
    """
    # Evaluasi loss dan accuracy
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    
    # Prediksi
    y_pred = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Confusion matrix
    num_classes = y_test.shape[1]
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_classes, pred_classes):
        cm[t, p] += 1
    
    print(f"\n{'='*60}")
    print(f"EVALUASI MODEL KERAS")
    print(f"{'='*60}")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return {
        'loss': loss,
        'accuracy': acc,
        'confusion_matrix': cm,
        'predictions': y_pred
    }


def keras_history_to_dict(keras_history):
    """
    Konversi Keras History ke format dictionary yang sama dengan from scratch.
    
    Parameters:
        keras_history: keras.callbacks.History object
        
    Returns:
        dict: History dalam format standar
    """
    history = {
        'train_loss': keras_history.history['loss'],
        'train_accuracy': keras_history.history['accuracy'],
        'test_loss': keras_history.history.get('val_loss', []),
        'test_accuracy': keras_history.history.get('val_accuracy', [])
    }
    return history
