
# Mamba-Multimodal: Image + Text Classification with State Space Models

This repository contains a full PyTorch implementation of a multimodal deep learning pipeline that uses [Mamba](https://github.com/state-spaces/mamba) —  to process and fuse image and text data for classification.
---

## Highlight

* **Multimodal Fusion**: Combines image features (ResNet) and text sequences (Mamba).
* **Mamba Integration**: Uses `mamba-ssm` for efficient long-sequence modeling.
* **Synthetic Dataset**: Self-contained dataset generation for easy testing.
* **Fast Training**: Lightweight and extensible for rapid experimentation.
* **Early Fusion**: Simple architecture to highlight core components.

---

## Requirements

* PyTorch ≥ 2.1
* torchvision
* mamba-ssm
* numpy

---

## How to Run

```bash
python main.py
```

Training will run for 10 epochs on synthetic image + text pairs, printing accuracy and loss each epoch.

---

## Model Architecture

```text
Text Input (L x 768) ──▶ Mamba ─▶ Mean Pool ─▶ Linear ─┐
                                                         │
Image Input (3x224x224) ─▶ ResNet18 ─▶ Linear  ─────────┘
                      ▼
            [ Concatenate + ReLU + Dropout ]
                      ▼
                Final Classifier
```

---

## Project Structure

```
mamba_multimodal/
├── main.py         # Training loop
├── model.py        # Mamba + ResNet multimodal model
├── dataset.py      # Synthetic image + text dataset
├── config.py       # Hyperparameters
├── utils.py        # Accuracy function
├── requirements.txt
└── README.md
```

---


## Notes

* The dataset is synthetic for demonstration purposes. You can replace `SyntheticMultimodalDataset` with any real dataset like if you have the computational capacity:

  * VQA
  * MSCOCO (images + captions)
  * MIMIC-CXR (X-ray + report)

* I am extendeing the model to experiment with:
  * Cross-attention
  * Multimodal transformers
  * Late fusion
---

## Citation

The code is based on the SSM repository. If you use this repo in your work, consider citing:

> [State Space Models: Mamba](https://github.com/state-spaces/mamba)
> Gu, Dao, et al. "Mamba: Linear-Time Sequence Modeling with Selective SSMs." (2023)

---
