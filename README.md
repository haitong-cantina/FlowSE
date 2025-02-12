# FlowSE: Flow-Matching Model for Speech Enhancement

**FlowSE** is the first flow-matching model for Speech Enhancement (SE), designed to address the key challenges faced by existing generative models in SE tasks. Traditional approaches like language model-based SE often degrade timbre and intelligibility due to **quantization loss**, while diffusion models suffer from **complex training** and **high inference latency**. FlowSE provides an efficient and innovative solution to these issues.

## 🔑 Key Features
- **Flow Matching for Speech Enhancement**: FlowSE is trained on noisy mel spectrograms and optional text sequences, optimizing a condition flow matching loss with ground-truth mel spectrograms as labels.
- **Implicit Learning of Temporal-Spectral Structure and Text Alignment**: FlowSE learns the speech’s temporal-spectral structure and text-to-speech alignment implicitly without explicit alignment procedures.
- **Flexible Inference Modes**:
  - Inference with noisy mel spectrograms only
  - Inference with noisy mel spectrograms and additional transcripts, providing enhanced performance

## 📊 Experimental Results
Extensive experiments demonstrate that FlowSE significantly **outperforms state-of-the-art generative SE methods**, establishing a new standard for generative-based SE and highlighting the potential of flow matching in advancing the field.

## 🗃️ Project Structure
```plaintext
FlowSE/
│
├── data/                  # Data preprocessing and loading utilities
├── models/                # FlowSE model code
├── checkpoints/           # Pre-trained model weights
├── utils/                 # Utility functions
├── inference.py           # Inference script
├── train.py               # Training script
└── README.md              # This documentation
```


## 🚀 Quick Start

- **1️⃣ Download environment requirements**


- **2️⃣ Download pretrained weights**

  We provided pretrained weights and audio samples.

- **3️⃣ Inference example**


## 📁 Resources
  - Audio samples in FlowSE/static/audio
