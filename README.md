# ✈️ Aircraft Damage Detection & Auto-Captioning Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FFEedMPyk24no_Zy45MkPpmer7qBtiET?usp=sharing)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white)

## 📌 Project Overview
Aircraft maintenance requires rigorous, error-free inspections. Traditional manual checks are time-consuming and subjective. This project introduces an automated **Vision-Language Pipeline** that not only detects and classifies aircraft surface damage (Dents vs. Cracks) but also generates automated textual descriptions of the damage.

The system integrates a **Convolutional Neural Network (CNN)** for feature extraction with a **Transformer-based Vision-Language Model (VLM)** for automated reporting, representing a robust approach to industrial visual inspection.

## ✨ Key Features & Architectural Decisions
To overcome the challenge of an extremely small dataset (~300 training images) and prevent severe overfitting, several advanced engineering strategies were implemented:

1.  **Two-Phase Fine-Tuning Strategy:** A highly controlled transfer learning approach using **VGG16**.
2.  **Global Average Pooling (GAP):** Replaced the standard `Flatten` layer with `GlobalAveragePooling2D`. This reduced the trainable parameters from over 25 million to just ~164k, aggressively combating overfitting while retaining spatial context.
3.  **Custom Keras Layer for HuggingFace:** Encapsulated the PyTorch-based **BLIP (Bootstrapping Language-Image Pretraining)** model inside a custom `tf.keras.layers.Layer`, enabling seamless interoperability between TensorFlow and PyTorch within a single inference graph.
4.  **Mild Domain-Specific Augmentation:** Applied restricted geometric transformations (10° rotation, slight shifts) to preserve the physical properties and lighting physics of aerospace metallurgy.

## 🧠 Model Architecture & Training Strategy

### Phase 1: Classification Head Warm-up
* **Action:** The entire VGG16 base model was frozen.
* **Objective:** Train the newly initialized custom classification head (`GAP -> Dense(256) -> Dropout(0.3) -> Dense(128) -> Dense(1)`).
* **Result:** Allowed the random weights to align with the extracted ImageNet features without propagating destructive gradients to the base model.

### Phase 2: Deep Fine-Tuning (The Breakthrough)
* **Action:** Unfroze the final convolutional block (`block5_conv1` to `block5_pool`) of VGG16.
* **Hyperparameters:** Drastically reduced the learning rate to $10^{-5}$ utilizing `ReduceLROnPlateau` and `EarlyStopping`.
* **Objective:** Adapt the high-level geometric feature detectors (originally trained on generic objects) to specifically recognize the textures, shadows, and stress lines of aircraft dents and cracks.

## 📊 Results & Performance
Despite the severely limited dataset, the two-phase fine-tuning approach yielded exceptional generalization on strictly unseen test data:

* **Test Accuracy:** **84.00%**
* **Precision (Cracks):** **90%** (Crucial metric: When the model predicts a structural crack, it is correct 90% of the time).
* **Recall (Dents):** **92%** (The model successfully identifies 92% of all actual dents).

*(Note: Replace the links below with the actual paths to your screenshots after uploading them to your GitHub repository)*
> `![Training Curves](link_to_your_training_curve_image.png)`
> `![Confusion Matrix](link_to_your_confusion_matrix_image.png)`

## 💻 Final Pipeline Output Example
The inference pipeline takes a raw image, classifies it, and generates a context-aware summary.

```text
==================================================
🚀 FINAL PIPELINE TEST: Classification + Captioning
==================================================
🎯 VGG16 Prediction: DENT (Confidence: 96.45%)
--------------------------------------------------
📝 Generating AI Caption...
▶ Caption: This is a close-up photo of an aircraft that was damaged in the crash.

📝 Generating AI Summary...
▶ Summary: A detailed summary of an aircraft surface showing the damage.
==================================================
