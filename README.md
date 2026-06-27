# Tugas Besar: Neural Network from Scratch
## Mata Kuliah Kecerdasan Buatan | Universitas Darunnajah 2025/2026

### Deskripsi
Proyek klasifikasi jenis kacang kering (Dry Bean Dataset) menggunakan Artificial Neural Network (ANN) dengan dua pendekatan:
1. **Implementasi From Scratch** — Neural Network dibangun menggunakan NumPy tanpa library deep learning
2. **Implementasi Keras** — Neural Network menggunakan framework Keras/TensorFlow

### Dataset
- **Dry Bean Dataset** (UCI Machine Learning Repository)
- 13.611 sampel, 16 fitur numerik, 7 kelas (Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira)

### Struktur Folder
```
Matkul AI/
├── README.md                       # File ini
├── requirements.txt                # Dependencies
├── data/                           # Dataset
│   └── Dry_Bean_Dataset.csv
├── src/                            # Source code
│   ├── neural_network.py           # NN From Scratch (kelas utama)
│   ├── activations.py              # Fungsi aktivasi & turunannya
│   ├── losses.py                   # Fungsi loss & turunannya
│   ├── initializers.py             # Metode inisialisasi bobot
│   ├── optimizers.py               # Optimizer (SGD, Momentum, Adam)
│   ├── data_loader.py              # Loading & preprocessing data
│   ├── metrics.py                  # Metrik evaluasi
│   ├── visualizations.py           # Visualisasi hasil
│   ├── utils.py                    # Fungsi utilitas
│   ├── keras_model.py              # Implementasi Keras
│   ├── run_scratch_experiments.py   # Eksperimen from scratch
│   └── run_keras_experiments.py     # Eksperimen Keras
├── results/                        # Hasil eksperimen & grafik
│   ├── scratch/
│   └── keras/
└── doc/                            # Laporan PDF
```

### Cara Setup dan Menjalankan Program

#### 1. Prerequisites
- Python 3.8 atau lebih baru
- pip (Python package manager)

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Download Dataset
Unduh Dry Bean Dataset dari [UCI ML Repository](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) dan simpan file CSV di folder `data/`.

Atau, program akan otomatis mencoba mengunduh saat pertama kali dijalankan.

#### 4. Menjalankan Eksperimen From Scratch
```bash
cd src
python run_scratch_experiments.py
```

#### 5. Menjalankan Eksperimen Keras
```bash
cd src
python run_keras_experiments.py
```

#### 6. Melihat Hasil
Semua grafik dan visualisasi disimpan di folder `results/`.

### Fitur yang Diimplementasikan

#### From Scratch
| Komponen | Detail |
|----------|--------|
| **Arsitektur** | Fully Connected Layer (fleksibel) |
| **Inisialisasi** | Zero, Random Uniform, Random Normal, Xavier, He |
| **Aktivasi** | Sigmoid, ReLU, Tanh, Softmax |
| **Loss** | MSE, Binary Cross Entropy, Categorical Cross Entropy |
| **Optimizer** | Mini-batch GD, Momentum (bonus), Adam (bonus) |
| **Metrik** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |

#### Eksperimen
1. Pengaruh depth & width arsitektur
2. Pengaruh inisialisasi bobot
3. Pengaruh fungsi aktivasi
4. Pengaruh learning rate (0.1, 0.01, 0.001)
5. Pengaruh metode optimasi (Mini-batch GD, Momentum, Adam)
6. Pengaruh batch size (16, 32, 64)
7. Pengaruh jumlah epoch (50, 100, 200)
8. Pengaruh fungsi loss

### Pembagian Tugas
| Anggota | Tugas |
|---------|-------|
| Muhammad Hakim Farros M | Implementasi From Scratch (NN, Aktivasi, Loss) & Laporan |
| Hisyam Badrus Syafiq | Implementasi Keras, Eksperimen & Presentasi |

### Referensi
1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
2. Koklu, M. & Ozkan, I.A. (2020). Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques. Computers and Electronics in Agriculture.
3. UCI Machine Learning Repository - Dry Bean Dataset
