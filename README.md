# 🌱 Plant Seedlings Classification with Vision Transformer (ViT)

This project fine-tunes a pretrained **Vision Transformer (ViT)** model to classify plant seedlings into 12 different species using the [Plant Seedlings Classification Dataset](https://www.kaggle.com/competitions/plant-seedlings-classification).

---

## 🧠 Overview

We use the `google/vit-base-patch16-224` model from Hugging Face Transformers and fine-tune it on plant images. The goal is to accurately classify each image into its correct species.

---

## 📂 Dataset

- Source: Kaggle Plant Seedlings Classification
- Structure:
train/
├── Black-grass/
├── Charlock/
├── Cleavers/
└── ...


- Total Classes: 12 plant species

---

## 📊 Model Architecture

- **Base Model:** `google/vit-base-patch16-224`
- **Modified Head:** Adjusted for 12-class classification
- **Input Size:** 224x224 pixels
- **Framework:** PyTorch + Hugging Face Transformers

---

## ⚙️ Training Pipeline

1. **Preprocessing:**
 - Resize to 224x224
 - Normalize using ViT's `image_mean` and `image_std`

2. **Dataset & Dataloader:**
 - Custom PyTorch `Dataset` class
 - `DataLoader` for batching and shuffling

3. **Training Loop:**
 - Uses `CrossEntropyLoss` and `AdamW` optimizer
 - Tracks accuracy and loss per epoch

4. **Validation:**
 - Runs after each epoch
 - No weight updates (uses `model.eval()` and `torch.no_grad()`)

---

## 🔍 Inference

After training, the model can predict the plant species of a given image.

```python
predicted_class_id, image = predict_image("path/to/image.jpg")
print("Predicted label:", id2label[predicted_class_id])
