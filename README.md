# TrOCR Fine-Tuning Pipeline for Handwritten Text Recognition

This repository contains a lightweight and efficient pipeline for fine-tuning the TrOCR model dedicated solely to handwritten text recognition. The pipeline leverages custom data augmentations, a dedicated dataset, and tailored training configurations to improve performance on handwritten document transcription.

---

## Overview

Handwritten text recognition poses unique challenges compared to printed text due to greater variability in writing styles, noise, and degradation. This pipeline is designed to address these challenges by:

- **Data Processing:** Loading images and normalized transcriptions from a dedicated dataset, with samples filtered using partition files for training, validation, and testing.
- **Data Augmentation:** Employing an Albumentations-based pipeline to ensure consistent image resizing (target size: 256x50 pixels), padding, and conversion to tensor format.
- **Custom Dataset & Collator:** Using a custom PyTorch dataset class and dynamic data collator to handle variable-length label sequences.
- **Model Configuration:** Fine-tuning the TrOCR model with strategies such as gradient accumulation and a cosine learning rate scheduler, along with evaluation metrics for Character Error Rate (CER) and Word Error Rate (WER).

---

## Pipeline Details

### Dataset Source

- **Handwritten Dataset:**  
  [Spanish Notary Collection](https://github.com/raopr/SpanishNotaryCollection)

### Data Processing

- **Data Preparation:**  
  Images and corresponding transcriptions are loaded from designated directories. Transcriptions are normalized (e.g., replacing underscores and specific characters) for consistency. Partition files are used to filter samples into training, validation, and testing sets.

### Data Augmentation

- **Augmentation Pipeline:**  
  The pipeline uses Albumentations to resize images to a fixed target size of 256x50 pixels, apply padding when necessary, and convert the images to tensor format. This ensures that the input size is uniform for the model.

### Custom Dataset & Data Collator

- **Custom Dataset:**  
  A dedicated PyTorch dataset class processes each image and tokenizes its corresponding transcription.
  
- **Dynamic Data Collator:**  
  A custom collator is implemented to dynamically pad the label sequences, handling variable lengths in transcriptions efficiently.

### Model Configuration

- **Fine-Tuning Settings:**  
  The pipeline employs a fine-tuning strategy similar to other TrOCR pipelines, which includes:
  - Gradient accumulation to stabilize training.
  - A cosine learning rate scheduler with warmup steps.
  - Evaluation using CER and WER metrics.
  
- **Performance:**  
  Training graphs demonstrate a gradual reduction in losses, with the validation CER reported to stabilize at around **0.46**. (Note: Graphs such as `handwritten_pipeline_graph.png` visually capture these trends.)

---

## Training and Evaluation

### Common Components

- **Frameworks and Libraries:**  
  This pipeline utilizes Hugging Face’s Transformers and Datasets libraries along with evaluation metrics from the Evaluate library.
  
- **Graphical Analysis:**  
  Training graphs show the decay in loss values over time and a steady convergence of CER. These graphs are essential for understanding the training dynamics.

### Reproducibility

- **Reproducibility Measures:**  
  Random seeds are set to ensure consistent training results, and GPU memory is managed via `torch.cuda.empty_cache()` to prevent memory leaks.

---

## Requirements

- Python 3.7+
- Transformers
- Datasets
- Evaluate
- Torchvision
- Albumentations
- Pandas, PIL, NumPy

---

## How to Run

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/moksha-hub/Trocr-model.git
   cd Trocr-model

