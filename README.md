# 👤 Face Grouping with Hybrid Embeddings + DBSCAN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muffin-123/FaceGrouping_HybridEmbedder/blob/main/HybridEmbedder.ipynb)

This notebook groups photos by person using an **ensemble of pretrained face models** (ArcFace, FaceNet, SFace, VGG-Face), **MTCNN alignment**, and **DBSCAN** clustering over a cosine distance matrix. It’s order-independent, robust to pose/lighting, and easy to tune.

## ✨ Features
- Face detection & alignment: **MTCNN**
- Hybrid embeddings: **ArcFace + FaceNet + SFace + VGG-Face** (normalized & concatenated)
- Clustering: **DBSCAN** on cosine distance matrix
- Visualization of grouped results
- Colab-ready with GPU support

## 🛠 Requirements
See [`requirements.txt`](requirements.txt). Minimal install:
```bash
pip install deepface opencv-python scikit-learn matplotlib numpy
