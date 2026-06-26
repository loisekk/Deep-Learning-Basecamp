# 🧠 Deep Learning Basecamp

A complete, beginner-friendly Deep Learning repository — from neural network fundamentals to advanced architectures. Every concept is backed by theory, real-world intuition, and hands-on projects.

---

## 📚 Series Roadmap

| Part | Topic | Status |
|------|-------|--------|
| Part 1 — Foundations | Perceptron, ANN, Activation Functions, Loss & Optimizers | ✅ Done |
| Part 2 — Computer Vision | CNNs, Image Augmentation, Transfer Learning | 🔜 Coming Soon |
| Part 3 — Sequence Models | RNNs, LSTMs, GRUs, Time Series Forecasting | 🔜 Coming Soon |
| Part 4 — Advanced Deep Learning | Transformers, GANs, Autoencoders, Reinforcement Learning | 🔜 Coming Soon |

---

## Part 1 — Neural Network Foundations

> *"Before diving into complex architectures, you must master the building blocks of every neural network."*

### What You'll Learn

- What a Perceptron is and how it works
- Artificial Neural Networks (ANN) — from single layer to deep
- Activation Functions — ReLU, Sigmoid, Softmax, and why they matter
- Loss Functions — Categorical Crossentropy, Sparse variants
- Optimizers — SGD, Adam, and learning rate intuition
- Forward Propagation vs Backpropagation
- Epochs, Batch Size, and Training Dynamics
- Model Evaluation for Multi-Class Classification

### 📌 Project 1 — Hybrid Iris Classifier (Perceptron & Keras)

A side-by-side comparison of a classical **Perceptron** vs a **Keras Sequential ANN** on the Iris dataset — with a full interactive Streamlit dashboard.

**Notebook Highlights:**

- Loaded Iris dataset via Seaborn with EDA (pairplot, describe, value_counts)
- Trained a scikit-learn **Perceptron** as a baseline model
- Built a **Keras Sequential ANN**: `Input(4) → Dense(16, ReLU) → Dense(8, ReLU) → Dense(3, Softmax)`
- Trained with `sparse_categorical_crossentropy` and Adam optimizer
- Evaluated with confusion matrix, accuracy score, and classification report

**Streamlit Dashboard — "NeuroLab · ANN Studio":**

An 8-page interactive dashboard featuring:

| Page | What It Does |
|------|-------------|
| **Overview** | Metrics cards, violin plots, donut chart, model status pills |
| **Data Explorer** | Scatter matrix, correlation heatmap, box plots, descriptive stats |
| **Model Comparison** | Side-by-side confusion matrices, per-class P/R/F1 charts, ROC-AUC curves |
| **Live Inference** | Interactive feature sliders, real-time prediction, nearest training sample finder |
| **Retrain ANN** | Full hyperparameter tuning (epochs, batch size, LR, dropout, layers, activation) with live curves |
| **Network Visualizer** | Plotly neural network graph, weight heatmap distributions, weight histograms |
| **Training History** | Loss/accuracy curves, convergence summary, smoothed loss with rolling average |
| **Feature Importance** | Permutation importance, ranked summary, feature variance vs importance scatter |

**Notebook:** [`ANN_Percp-Kearas.ipynb`](Hybrid_Iris_Classifier/ANN_Percp-Kearas.ipynb)

**Dashboard:** [`app.py`](Hybrid_Iris_Classifier/app.py)

---

## Part 2 — Computer Vision (Coming Soon)

> *"Teaching machines to see — from edge detection to object recognition."*

### What You'll Learn

- Convolutional Neural Networks (CNNs) — Conv2D, Pooling, Feature Maps
- Image Preprocessing & Data Augmentation
-经典 architectures — LeNet, AlexNet, VGG, ResNet
- Transfer Learning — leveraging pre-trained models
- Image Classification & Object Detection fundamentals

### 🔬 Planned Projects

- 📌 CIFAR-10 Image Classifier — from scratch CNN vs Transfer Learning
- 📌 Skin Cancer Detection — medical imaging with Transfer Learning

---

## Part 3 — Sequence Models (Coming Soon)

> *"Understanding language, time, and order — one sequence at a time."*

### What You'll Learn

- Recurrent Neural Networks (RNN) — the concept of memory
- LSTM & GRU — solving the vanishing gradient problem
- Bidirectional RNNs
- Time Series Forecasting with Deep Learning
- Text Classification & Sentiment Analysis
- Word Embeddings — Word2Vec, GloVe

### 🔬 Planned Projects

- 📌 Stock Price Predictor — LSTM-based time series forecasting
- 📌 Sentiment Analyzer — IMDB reviews with LSTM/GRU

---

## Part 4 — Advanced Deep Learning (Coming Soon)

> *"The frontier of modern AI — transformers, generative models, and beyond."*

### What You'll Learn

- Transformers & Self-Attention mechanism
- Generative Adversarial Networks (GANs)
- Autoencoders & Variational Autoencoders (VAE)
- Reinforcement Learning basics
- Model deployment strategies

### 🔬 Planned Projects

- 📌 Face Generator — DCGAN on CelebA
- 📌 Text Generator — Transformer-based language model

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| **Deep Learning** | TensorFlow, Keras |
| **Machine Learning** | Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Web Dashboard** | Streamlit |
| **Model Serialization** | Joblib, Keras Native (.keras) |

---

## 🚀 Getting Started

```bash
git clone https://github.com/loisekk/Deep-Learning-Basecamp.git
cd Deep-Learning-Basecamp
pip install -r requirements.txt
```

Open any notebook in Jupyter or Google Colab and follow along.

To run the Iris Classifier dashboard:

```bash
cd Hybrid_Iris_Classifier
streamlit run app.py
```

---

## 📂 Repository Structure

```
Deep-Learning-Basecamp/
├── README.md
├── requirements.txt
├── Hybrid_Iris_Classifier/
│   ├── ANN_Percp-Kearas.ipynb       # Training notebook
│   ├── app.py                        # Streamlit dashboard
│   ├── requirements.txt              # Dependencies
│   ├── keras_sequential_model.keras  # Trained Keras model
│   ├── keras_model.joblib            # Serialized Keras model
│   └── perceptron_model.joblib       # Serialized Perceptron
```

---

## ⭐ Support

If this repo helped you, consider giving it a star — it helps others find it too!
