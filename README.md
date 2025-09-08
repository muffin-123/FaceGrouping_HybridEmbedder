# 👤 Face Grouping with Hybrid Embeddings + DBSCAN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muffin-123/FaceGrouping_HybridEmbedder/blob/main/HybridEmbedder.ipynb)
![Python](https://img.shields.io/badge/python-3.12.11%2B-blue)


This notebook groups photos by person using an **ensemble of pretrained face models** (ArcFace, FaceNet, SFace, VGG-Face), **MTCNN alignment**, and **DBSCAN** clustering over a cosine distance matrix. It’s order-independent, robust to pose/lighting, and easy to tune.

## ✨ Features
- Face detection & alignment: **MTCNN**
- Hybrid embeddings: **ArcFace + FaceNet + SFace + VGG-Face** (normalized & concatenated)
- Clustering: **DBSCAN** on cosine distance matrix
- Visualization of grouped results
- Colab-ready with GPU support

## 🛠 Requirements
See [`requirements.txt`](requirements.txt). 


## 📂 Dataset

This project was tested on the **Labeled Faces in the Wild (LFW)** dataset.

- 📥 Download (aligned/funneled version):  
  [LFW Deepfunneled (UMass)](http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz)



- 📌 To run experiments:
1. Pick a few subfolders (different people).
2. Copy their images into a flat folder (`/content/images_concat_aligned`).
3. Run the notebook → embeddings are generated and grouped automatically.


### 🔧 Custom Datasets
You can also use **your own dataset**:
- Place images in subfolders (one folder per person), or in a single folder with mixed people.
- Update the dataset path in the notebook:
```python
input_root = "/content/drive/MyDrive/Datasets/testDataset"
```

## ⚠️ Note on Face Grouping

In real-world datasets, the **same person may appear under different lighting, poses, or facial expressions**, which can lead clustering algorithms to create multiple groups for the same individual.  

My approach already reduces this issue significantly by:
- Using **MTCNN alignment** for consistent face crops  
- Combining embeddings from multiple models (**ArcFace, FaceNet, SFace, VGG-Face**)  
- Applying **DBSCAN clustering**, which is more robust than simple thresholding  

Still, this behavior reflects the complexity of face recognition in diverse conditions and is also why many production systems (e.g., Google Photos) include an option for **manual group merging** as part of their workflow.

