# 👁️ Glaucoma Detection AI — IDSC 2026
### Mathematics for Hope in Healthcare

> *"Glaucoma silently steals sight — 50% of patients don't know they have it until it's too late. This AI system detects glaucoma from a single retinal photo, before permanent damage occurs."*

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Results](#results)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Model Architecture](#model-architecture)
- [Ethics & Limitations](#ethics--limitations)
- [Citation](#citation)
- [Team](#team)

---

## 🎯 Overview

This project was developed for the **International Data Science Challenge 2026 (IDSC 2026)** under the theme **"Mathematics for Hope in Healthcare"**, organized by Universiti Putra Malaysia (UPM) in partnership with UNAIR, UNMUL, and UB.

We built a complete end-to-end CNN pipeline for **Glaucomatous Optic Neuropathy (GON) detection** using retinal fundus images from the HYGD PhysioNet dataset.

| Metric | Score |
|--------|-------|
| **Accuracy** | **95.85%** (208/217) |
| **AUC-ROC** | **0.9795** |
| **Precision GON+** | 0.98 |
| **Recall GON+** | 0.95 |
| **F1-Score GON+** | 0.97 |
| **F1-Score GON-** | 0.95 |
| **False Positives** | 3 |
| **False Negatives** | 6 |

---

## 📊 Dataset

**Hillel Yaffe Glaucoma Dataset (HYGD)**
- **Source:** [PhysioNet](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/)
- **DOI:** https://doi.org/10.13026/pdxv-m215
- **Total Images:** 747 retinal fundus images
- **After Quality Filter:** 701 images (quality score ≥ 4)
- **Labels:** GON+ (Glaucoma) / GON- (Normal)
- **Label Type:** Gold-standard — full clinical examination (OCT, IOP, Visual Field Tests, 1-year follow-up)
- **Camera:** TOPCON DRI OCT Triton, 45° FOV
- **Patients:** 288 patients, age 36–95 years

> ⚠️ **Note:** Dataset images are NOT included in this repository. Download directly from PhysioNet.

### Download Dataset:
```bash
# Option 1: wget
wget -r -N -c -np https://physionet.org/files/hillel-yaffe-glaucoma-dataset/1.0.0/

# Option 2: AWS CLI
aws s3 sync --no-sign-request s3://physionet-open/hillel-yaffe-glaucoma-dataset/1.0.0/ ./HYGD
```

After downloading, place files as:
```
IDSC2026/
├── Images/       ← all .jpg files here
└── Labels.csv    ← labels file here
```

---

## 📁 Project Structure

```
IDSC2026/
│
├── 📁 Images/                          # Original dataset images (download separately)
├── 📁 Images_resized/                  # Resized 224×224 images (auto-generated)
├── 📁 Images_augmented/                # Augmented images (auto-generated)
│
├── 📓 notebooks/
│   ├── 01_EDA.ipynb                    # Exploratory Data Analysis
│   ├── 02_PREPRO.ipynb                 # Preprocessing & Augmentation
│   ├── 03_TrainTestSplit.ipynb         # Patient-level Train/Val/Test Split
│   ├── 04_Training.ipynb               # CNN Model Training
│   ├── 05_Evaluation.ipynb             # Evaluation, Metrics & ROC Curve
│   ├── 06_Visualization.ipynb          # Grad-CAM Explainability
│   │
│   ├── confusion_matrix.png            # Confusion matrix visualization
│   ├── distribusi_label.png            # Class distribution chart
│   ├── gradcam_visualization.png       # Grad-CAM heatmaps
│   ├── quality_score.png               # Quality score analysis
│   ├── roc_curve.png                   # ROC curve (AUC = 0.9795)
│   ├── sample_images.png               # Sample retinal images
│   └── training_history.png            # Training accuracy/loss curves
│
├── 🌐 app.py                           # Streamlit Dashboard
├── 🌐 app2.py                          # Streamlit Dashboard (alternative)
│
├── 📊 Labels.csv                       # Original dataset labels
├── 📊 labels_filtered.csv              # Filtered labels (quality score ≥ 4)
├── 📊 glaucoma_clean_dataset.csv        # Clean dataset with image paths
├── 📊 glaucoma_augmented_dataset.csv    # Final balanced dataset (1001 images)
├── 📊 train_dataset.csv                 # Training split (619 images)
├── 📊 val_dataset.csv                   # Validation split (165 images)
├── 📊 test_dataset.csv                  # Test split (217 images)
├── 📊 test_predictions.csv              # Model predictions on test set
│
├── 🤖 glaucoma_model.keras              # Trained CNN model (Keras format) ← main
├── 🤖 glaucoma_model.h5                 # Trained CNN model (HDF5 format)
│
├── 📋 requirements.txt                  # Python dependencies
└── 📖 README.md                         # This file
```

---

## 🔄 Pipeline

```
📥 HYGD Dataset (747 images)
         │
         ▼
📊 01_EDA.ipynb
   ├── Load & inspect Labels.csv
   ├── Class distribution: 548 GON+ (73%) / 199 GON- (27%)
   ├── Visualize retinal fundus images
   ├── Quality score analysis (mean: 5.90 ± 1.01)
   └── Filter quality score < 4 → 46 removed → 701 images remain
         │
         ▼
⚙️ 02_PREPRO.ipynb
   ├── Resize all 701 images → 224×224 pixels
   ├── Convert to NumPy arrays
   ├── Normalize pixel values: 0–255 → 0.0–1.0
   ├── Augmentation on GON- ONLY:
   │   ├── Horizontal Flip  → 150 new images
   │   └── 90° Rotation     → 150 new images
   └── Final: 509 GON+ vs 492 GON- = 1001 total
         │
         ▼
✂️ 03_TrainTestSplit.ipynb
   ├── Split by PATIENT ID (prevents data leakage)
   ├── Train:      619 images — 179 patients (61.8%)
   ├── Validation: 165 images —  45 patients (16.5%)
   └── Test:       217 images —  57 patients (21.7%)
         │
         ▼
🧠 04_Training.ipynb
   ├── Architecture: 3 Conv blocks (32→64→128 filters)
   ├── Regularization: Dropout(0.5)
   ├── Callbacks: EarlyStopping(patience=5) + ModelCheckpoint
   ├── Config: 10 epochs, batch_size=32, Adam optimizer
   └── Best model saved: glaucoma_model.keras
         │
         ▼
📈 05_Evaluation.ipynb
   ├── Test Accuracy:   95.85% (208/217 correct)
   ├── AUC-ROC Score:   0.9795
   ├── False Positives: 3
   ├── False Negatives: 6
   └── Confusion Matrix + ROC Curve generated
         │
         ▼
🔍 06_Visualization.ipynb
   ├── Grad-CAM heatmaps on correct & incorrect predictions
   ├── Analysis of false negative cases
   └── Confirms model focuses on optic disc region
         │
         ▼
🌐 app.py — Streamlit Dashboard
   └── Upload retinal image → instant prediction + confidence score
```

---

## 📈 Results

### Confusion Matrix
```
                  Predicted Normal    Predicted Glaucoma
Actual Normal           82                    3
Actual Glaucoma          6                  126
```

### Classification Report
```
                  Precision   Recall   F1-Score   Support
GON- (Normal)       0.93       0.96      0.95        85
GON+ (Glaucoma)     0.98       0.95      0.97       132
─────────────────────────────────────────────────────────
Accuracy                                 0.96       217
AUC-ROC                                  0.9795
```

### Training History
| Epoch | Train Accuracy | Val Accuracy |
|-------|---------------|--------------|
| 1 | ~85% | ~87% |
| 4 | 92.08% | **95.76%** ← Best saved |
| 10 | 95.80% | 93.94% |

### Key Highlights
- ✅ **95.85% accuracy** on 217 unseen test images
- ✅ **AUC-ROC 0.9795** — near-perfect class discrimination
- ✅ **Only 3 false positives** — minimal misdiagnosis of healthy patients
- ✅ **Only 6 false negatives** — very few missed glaucoma cases
- ✅ **Grad-CAM confirms** model focuses on optic disc — clinically valid
- ✅ **No overfitting** — train and val accuracy converge cleanly



## ▶️ How to Run

> ⚠️ **Run notebooks strictly in order** — each depends on outputs from the previous one.

```bash
# Activate virtual environment first
source venv/bin/activate

# Launch Jupyter
jupyter notebook
```

### Notebook Execution Order

| Order | Notebook | Input | Output |
|-------|----------|-------|--------|
| 1 | `01_EDA.ipynb` | `Labels.csv`, `Images/` | `labels_filtered.csv` |
| 2 | `02_PREPRO.ipynb` | `labels_filtered.csv` | `Images_resized/`, `Images_augmented/`, `glaucoma_augmented_dataset.csv` |
| 3 | `03_TrainTestSplit.ipynb` | `glaucoma_augmented_dataset.csv` | `train_dataset.csv`, `val_dataset.csv`, `test_dataset.csv` |
| 4 | `04_Training.ipynb` | `train/val dataset` | `glaucoma_model.keras`, `training_history.png` |
| 5 | `05_Evaluation.ipynb` | `test_dataset.csv`, `glaucoma_model.keras` | `test_predictions.csv`, `confusion_matrix.png`, `roc_curve.png` |
| 6 | `06_Visualization.ipynb` | `test_predictions.csv`, `glaucoma_model.keras` | `gradcam_visualization.png` |

---

## 🌐 Streamlit Dashboard

```bash
# Make sure venv is active
streamlit run app.py
```

Open browser: **http://localhost:8501**

### How to Use
1. Click **Browse files**
2. Upload a retinal fundus image (JPG / PNG)
3. Wait for prediction (~2 seconds)
4. Read result — **Glaucoma** or **Normal** with confidence score

> ⚠️ For educational and research demonstration only. Not a clinical diagnostic tool.

---

## 🔬 Model Architecture

```
Input:  (224, 224, 3)
    │
    ▼
Conv2D(32, 3×3, ReLU)  → MaxPooling2D(2×2)   # Edge & texture detection
    │
    ▼
Conv2D(64, 3×3, ReLU)  → MaxPooling2D(2×2)   # Shape detection
    │
    ▼
Conv2D(128, 3×3, ReLU) → MaxPooling2D(2×2)   # Optic disc features
    │
    ▼
Flatten → Dense(128, ReLU) → Dropout(0.5)
    │
    ▼
Dense(1, Sigmoid)
    │
    ▼
Output: Probability [0.0 – 1.0]
  > 0.5  →  Glaucoma (GON+)
  ≤ 0.5  →  Normal   (GON-)

Total Trainable Parameters: 11,169,089
```

---

## ⚠️ Ethics & Limitations

### Ethical Considerations
- **Explainability:** Grad-CAM provides visual explanation — clinicians can verify model decisions before acting
- **Patient Privacy:** HYGD is fully de-identified — no personal information traceable
- **Screening Only:** Designed as screening aid, not a replacement for specialist diagnosis
- **Transparency:** All preprocessing, augmentation, and training steps are fully documented and reproducible

### Limitations
1. **Device Variability** — Trained on single camera (TOPCON DRI OCT Triton). May degrade on images from different cameras
2. **Binary Classification** — Only Glaucoma vs Normal. Other retinal conditions not addressed
3. **Image Quality Dependency** — Requires quality score ≥ 4. Very poor quality images may yield unreliable predictions
4. **Limited Demographics** — Single geographic location (Israel). Generalizability requires further validation

---

## 🔮 Future Improvements

- [ ] Transfer learning (EfficientNetB0, MobileNetV2)
- [ ] Glaucoma severity staging (multi-class)
- [ ] Deploy on Streamlit Community Cloud
- [ ] Batch prediction support
- [ ] Cross-camera domain adaptation
- [ ] Integration with OCT data

---

## 📦 Requirements

```
tensorflow==2.21.0
pandas
numpy
matplotlib
seaborn
scikit-learn
pillow
opencv-python
streamlit
jupyter
ipykernel
```

```bash
pip freeze > requirements.txt
```

---

## 📚 Citation

```bibtex
@dataset{hygd2025,
  author    = {Abramovich, O. and Pizem, H. and Fhima, J. and others},
  title     = {Hillel Yaffe Glaucoma Dataset (HYGD)},
  year      = {2025},
  publisher = {PhysioNet},
  doi       = {10.13026/pdxv-m215},
  url       = {https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/}
}
```

---

## 👥 Team

**IDSC 2026 — Team [Road To Immo]**

| Name | Institution |
|------|-------------|
| Ahmad Marannuang Tajibu 23031554040 | [Universitas Negeri Surabaya] |
| Lofian Rafi Qurrotul Ain  23031554221 | [Universitas Negeri Surabaya] |

**Competition:** International Data Science Challenge 2026
**Theme:** Mathematics for Hope in Healthcare
**Organized by:** UPM × UNAIR × UNMUL × UB
**Contact:** ahmadhakiim@upm.edu.my

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

Dataset: [Open Data Commons Attribution License v1.0](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/)

---

*"Mathematics is not just numbers on a whiteboard — Mathematics is Hope."* 👁️✨
