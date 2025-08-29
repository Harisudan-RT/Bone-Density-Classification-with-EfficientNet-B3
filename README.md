# Bone density classification from X-ray images

## Overview

This project implements a **deep learning pipeline for classifying bone density images** into three categories:  
- Normal  
- Osteopenia  
- Osteoporosis  

Key features:  
- **EfficientNet-B3 backbone** with transfer learning  
- **Advanced data augmentation** using Albumentations  
- **Test-Time Augmentation (TTA)** for robust predictions  
- **Label smoothing & class-weighted loss** for handling class imbalance  
- **Cosine annealing scheduler** for dynamic learning rate adjustment  

---

## 🔧 Technologies Used

- **Deep Learning**: PyTorch, TIMM  
- **Data Processing & Augmentation**: OpenCV, Albumentations, NumPy, Pandas  
- **Imbalanced Dataset Handling**: imbalanced-learn (RandomOverSampler)  
- **Visualization**: Matplotlib  
- **Google Colab**: GPU acceleration  

---

## 📦 Dependencies

Install via pip:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install opencv-python scikit-learn pandas matplotlib numpy imbalanced-learn
pip install albumentations tqdm timm

🖼 Dataset

The dataset contains bone density images organized in three folders corresponding to the classes:
Normal, Osteopenia, and Osteoporosis.

Images are resized and normalized for EfficientNet input.

Data augmentation ensures robust feature extraction.

🚀 Features

EfficientNet-B3 Backbone: Pretrained on ImageNet, fine-tuned for bone density classification.

Advanced Data Augmentation: Horizontal/vertical flip, rotation, shift/scale/rotate, brightness/contrast, Gaussian blur/noise, and coarse dropout.

Test-Time Augmentation (TTA): Multiple predictions per image averaged for stable results.

Loss Functions: Label smoothing + class-weighted cross-entropy to handle imbalance.

Learning Rate Scheduler: Cosine annealing with warm restarts for smooth training.

Early Stopping: Stops training when validation accuracy does not improve.

Balanced Dataset: Oversampling minority classes for better model generalization.

🏆 Training & Evaluation

Training:

Uses GPU if available.

Batch size and number of epochs configurable.

Mixed-precision training for efficiency.

Evaluation:

Standard validation metrics: Accuracy, F1-score, Confusion Matrix.

TTA-based evaluation for enhanced reliability.


