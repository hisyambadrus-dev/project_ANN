"""
Modul Data Loader
=================
Memuat dan memproses Dry Bean Dataset untuk eksperimen Neural Network.

Fitur:
- Loading dataset dari file CSV/Excel
- Preprocessing (normalisasi/standarisasi)
- One-hot encoding untuk label
- Train/Test split
- Mini-batch generator
"""

import numpy as np
import os


# ==================== Nama Kelas ====================
CLASS_NAMES = ['SEKER', 'BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SIRA']

# Nama kelas untuk display (title case)
CLASS_DISPLAY_NAMES = ['Seker', 'Barbunya', 'Bombay', 'Cali', 'Dermason', 'Horoz', 'Sira']

# ==================== Nama Fitur ====================
FEATURE_NAMES = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
    'AspectRatio', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
    'Extent', 'Solidity', 'Roundness', 'Compactness',
    'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
]


def load_dataset(filepath):
    """
    Memuat dataset dari file CSV atau Excel.
    
    Parameters:
        filepath (str): Path ke file dataset
        
    Returns:
        tuple: (X, y_encoded, y_onehot)
            - X (np.ndarray): Fitur input (13611, 16)
            - y_encoded (np.ndarray): Label encoded (13611,)
            - y_onehot (np.ndarray): Label one-hot (13611, 7)
    """
    # Tentukan ekstensi file
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.csv':
        # Baca CSV manual tanpa pandas
        data, header = _read_csv(filepath)
    elif ext in ['.xlsx', '.xls']:
        # Untuk Excel, gunakan openpyxl
        try:
            import openpyxl
            data, header = _read_excel(filepath)
        except ImportError:
            raise ImportError(
                "Untuk membaca file Excel, install openpyxl: pip install openpyxl"
            )
    else:
        # Coba baca sebagai CSV
        data, header = _read_csv(filepath)
    
    # Pisahkan fitur dan label
    # Asumsi: kolom terakhir adalah label (Class)
    X = data[:, :-1].astype(float)
    y_str = data[:, -1]
    
    # Encode label ke integer
    y_encoded = np.array([CLASS_NAMES.index(label.strip()) for label in y_str])
    
    # One-hot encoding
    y_onehot = one_hot_encode(y_encoded, num_classes=7)
    
    print(f"Dataset berhasil dimuat!")
    print(f"  Jumlah sampel: {X.shape[0]}")
    print(f"  Jumlah fitur: {X.shape[1]}")
    print(f"  Jumlah kelas: {len(CLASS_NAMES)}")
    print(f"  Distribusi kelas:")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(y_encoded == i)
        print(f"    {name}: {count} ({count/len(y_encoded)*100:.1f}%)")
    
    return X, y_encoded, y_onehot


def _read_csv(filepath):
    """
    Membaca file CSV secara manual.
    
    Parameters:
        filepath (str): Path ke file CSV
        
    Returns:
        tuple: (data_array, header_list)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Header
    header = lines[0].strip().split(',')
    
    # Data
    data = []
    for line in lines[1:]:
        line = line.strip()
        if line:
            data.append(line.split(','))
    
    return np.array(data), header


def _read_excel(filepath):
    """
    Membaca file Excel menggunakan openpyxl.
    
    Parameters:
        filepath (str): Path ke file Excel
        
    Returns:
        tuple: (data_array, header_list)
    """
    import openpyxl
    wb = openpyxl.load_workbook(filepath, read_only=True)
    ws = wb.active
    
    rows = list(ws.iter_rows(values_only=True))
    header = [str(cell) for cell in rows[0]]
    data = []
    for row in rows[1:]:
        data.append([str(cell) if cell is not None else '' for cell in row])
    
    wb.close()
    return np.array(data), header


def one_hot_encode(y, num_classes):
    """
    Mengubah label integer menjadi one-hot encoding.
    
    Contoh: 
        y = [0, 2, 1], num_classes = 3
        output = [[1,0,0], [0,0,1], [0,1,0]]
    
    Parameters:
        y (np.ndarray): Label integer (shape: n_samples,)
        num_classes (int): Jumlah kelas
        
    Returns:
        np.ndarray: One-hot encoded labels (shape: n_samples x num_classes)
    """
    n = len(y)
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), y.astype(int)] = 1
    return one_hot


def normalize(X, method='standard'):
    """
    Normalisasi/Standarisasi fitur.
    
    Parameters:
        X (np.ndarray): Data fitur (n_samples, n_features)
        method (str): Metode normalisasi
            - 'standard': Z-score standardization (mean=0, std=1)
            - 'minmax': Min-Max normalization (range 0-1)
        
    Returns:
        tuple: (X_normalized, params)
            - X_normalized: Data yang sudah dinormalisasi
            - params: Dictionary berisi parameter normalisasi
                      (digunakan untuk transform data test)
    """
    if method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1  # Hindari pembagian dengan nol
        X_norm = (X - mean) / std
        params = {'method': 'standard', 'mean': mean, 'std': std}
    
    elif method == 'minmax':
        x_min = np.min(X, axis=0)
        x_max = np.max(X, axis=0)
        range_val = x_max - x_min
        range_val[range_val == 0] = 1  # Hindari pembagian dengan nol
        X_norm = (X - x_min) / range_val
        params = {'method': 'minmax', 'min': x_min, 'max': x_max}
    
    else:
        raise ValueError(f"Metode normalisasi '{method}' tidak dikenali.")
    
    return X_norm, params


def apply_normalization(X, params):
    """
    Menerapkan normalisasi menggunakan parameter yang sudah dihitung.
    Digunakan untuk mentransform data test menggunakan parameter dari data training.
    
    Parameters:
        X (np.ndarray): Data yang akan dinormalisasi
        params (dict): Parameter normalisasi dari fungsi normalize()
        
    Returns:
        np.ndarray: Data yang sudah dinormalisasi
    """
    if params['method'] == 'standard':
        return (X - params['mean']) / params['std']
    elif params['method'] == 'minmax':
        range_val = params['max'] - params['min']
        range_val[range_val == 0] = 1
        return (X - params['min']) / range_val


def train_test_split(X, y, test_size=0.2, seed=42):
    """
    Membagi dataset menjadi training set dan testing set.
    
    Parameters:
        X (np.ndarray): Data fitur
        y (np.ndarray): Label (one-hot encoded)
        test_size (float): Proporsi data test (0.0 - 1.0)
        seed (int): Random seed untuk reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    indices = rng.permutation(n)
    
    split_idx = int(n * (1 - test_size))
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def create_mini_batches(X, y, batch_size=32, seed=None):
    """
    Membuat mini-batches dari dataset.
    
    Parameters:
        X (np.ndarray): Data fitur
        y (np.ndarray): Label
        batch_size (int): Ukuran batch
        seed (int): Random seed (opsional)
        
    Returns:
        list: List dari tuple (X_batch, y_batch)
    """
    n = X.shape[0]
    
    if seed is not None:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
    else:
        indices = np.random.permutation(n)
    
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    mini_batches = []
    
    for i in range(0, n, batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        mini_batches.append((X_batch, y_batch))
    
    return mini_batches


def download_dataset(save_dir='data'):
    """
    Mengunduh Dry Bean Dataset dari UCI ML Repository.
    
    Parameters:
        save_dir (str): Direktori untuk menyimpan dataset
        
    Returns:
        str: Path ke file dataset yang sudah diunduh
    """
    import urllib.request
    import zipfile
    
    os.makedirs(save_dir, exist_ok=True)
    
    url = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
    zip_path = os.path.join(save_dir, "dry_bean_dataset.zip")
    
    print(f"Mengunduh dataset dari {url}...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Mengekstrak file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_dir)
    
    # Cari file data
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            if file.endswith(('.csv', '.xlsx', '.xls', '.arff')):
                filepath = os.path.join(root, file)
                print(f"Dataset tersimpan di: {filepath}")
                return filepath
    
    print("File dataset tidak ditemukan setelah ekstrak.")
    return None
