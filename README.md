# FarmDoc — Plant Disease Detection

A deep learning system for automated plant disease detection using Convolutional Neural Networks. Given an image of a plant leaf, FarmDoc classifies it into one of **38 crop-disease pairs** — helping farmers identify diseases accurately without requiring specialist knowledge, and enabling precise pesticide and fertilizer recommendations.

**Author:** Stuti Upadhyay | GSFC University · CSE · Fundamentals of AI & ML (BTCS405) | Academic Year 2020-21

---

## Problem Statement

Farmers traditionally detect crop diseases with the naked eye, often confusing visually similar diseases, leading to incorrect fertilizer application and potential crop damage. FarmDoc addresses this by automating disease identification from leaf images, enabling accurate, timely intervention.

---

## Dataset

**PlantVillage Dataset**

| Metric | Value |
|---|---|
| Total images | 70,295 |
| Disease classes | 38 crop-disease pairs |
| Image resolution | Resized to 256 × 256 pixels |
| Train / Validation split | 80% / 20% |
| Test images | 33 |

The 38 classes span diseases across Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato — including both healthy and diseased variants for each crop.

---

## Model Architecture

An AlexNet-inspired CNN adapted for the 38-class PlantVillage classification task.

```
Input (256 × 256 × 3)
    → Conv2D (32 filters, 5×5, ReLU)
    → MaxPooling2D (3×3)
    → Conv2D (32 filters, 3×3, ReLU)
    → MaxPooling2D (2×2)
    → Conv2D (64 filters, 3×3, ReLU)
    → MaxPooling2D (2×2)
    → Flatten
    → Dense (512, ReLU)
    → Dropout (0.25)
    → Dense (128, ReLU)
    → Dense (38, Softmax)
```

**Total parameters:** 11,930,502

**Training configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Epochs: 5
- Batch size: 32
- Data augmentation: shear, zoom, horizontal flip

---

## Results

| Epoch | Training Accuracy | Validation Accuracy | Validation Loss |
|---|---|---|---|
| 1 | 39.07% | 84.04% | 0.4942 |
| 2 | 81.50% | 90.16% | 0.2978 |
| 3 | 87.12% | 92.26% | 0.2390 |
| 4 | 89.62% | 94.26% | 0.1724 |
| 5 | **91.31%** | **95.05%** | **0.1488** |

---

## Disease Classes Detected

The model classifies 38 crop-disease combinations, including:

`Apple Scab` · `Apple Black Rot` · `Apple Cedar Apple Rust` · `Apple Healthy` · `Blueberry Healthy` · `Cherry Powdery Mildew` · `Corn Gray Leaf Spot` · `Corn Common Rust` · `Corn Northern Leaf Blight` · `Grape Black Rot` · `Grape Black Measles` · `Grape Leaf Blight` · `Orange Haunglongbing (Citrus Greening)` · `Peach Bacterial Spot` · `Pepper Bacterial Spot` · `Potato Early Blight` · `Potato Late Blight` · `Tomato Bacterial Spot` · `Tomato Early Blight` · `Tomato Late Blight` · `Tomato Leaf Mold` · `Tomato Septoria Leaf Spot` · `Tomato Spider Mites` · `Tomato Target Spot` · `Tomato Yellow Leaf Curl Virus` · `Tomato Mosaic Virus` · and more.

---

## Setup

```bash
git clone https://github.com/stutiupadhyay03/FarmDoc.git
cd FarmDoc
```

**Requirements:** Python 3, TensorFlow/Keras, NumPy, Matplotlib, glob

```python
# Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.2,
    horizontal_flip=True
)

# Prediction
result = model.predict_classes([prepare('/path/to/leaf_image.JPG')])
print(Classes[int(result)])
# Output: 'Apple__Apple_scab'
```

---

## Repository Structure

```
FarmDoc/
├── FarmDoc.py               # Model training and prediction script
├── Report_FarmDoc.docx.pdf  # Full project report
└── README.md
```

---

## Future Work

- Convert model to **TensorFlow Lite** for mobile deployment
- Build an **Android application** with real-time camera input
- Expand dataset to include **grayscale images**
- Add **remedy recommendations** — pesticide and fertilizer guidance with day-by-day dosage tracking
- Integrate **crop recommendation** based on soil type
- Include **government scheme information** for farmers

---

## Stack

`Python` · `TensorFlow` · `Keras` · `NumPy` · `Matplotlib` · `PlantVillage Dataset` · `AlexNet CNN`
