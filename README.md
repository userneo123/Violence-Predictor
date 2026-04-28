# Early Violence Detection using CNN + LSTM

## Overview
This project implements a deep learning-based system for early violence prediction in videos.  
It uses a CNN (ResNet18) for spatial feature extraction and an LSTM for temporal modeling.

---

## Key Idea
Video → Frames → CNN → LSTM → Prediction  

The model learns how actions evolve over time to detect violence before it fully occurs.

---

## Model Architecture
- CNN: ResNet18 (feature extractor, frozen)
- LSTM: Temporal sequence learning
- Fully Connected Layer: Final prediction

---

## Dataset Preparation
- Convert videos into frames  
- Sliding window approach:
  - Window size: 16 frames  
  - Stride: 4  
- Labels:
  - `0` → Non-violence  
  - `1` → Violence  

---

## Training Details
- Loss Function: BCEWithLogitsLoss  
- Optimizer: Adam  
- Batch Size: 4  
- Input Shape: (16, 3, 224, 224)

---

## Output
- Model predicts probability of violence  
- Threshold = 0.5 for classification  
