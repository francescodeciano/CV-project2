# Joint Detection of AI-Generated Images and Post-Processing Alterations in Real-World Scenarios

This notebook contains the implementation of a unified multi-task Deep Learning framework designed to address two critical challenges in image forensics:
- identifying AI-generated images
- classifying post-processing alterations for images

The project is based on the **RRDataset**.

Table of contents
- Project Overview
- Model Architecture
- Dataset
- Project Structure
- Installation and Requirements
- Ablation Study
- Authors and References

## Project Overview

Standard AI-generated images detectors often fail because images circulating online are rarely unaltered, they undergo compression, resizing or re-digitalization.
This framework addresses the above limitation using a multi-task approach:

1. **Task 1**: real vs. fake detection
2. **Task 2**: transformation classification (Original vs. Internet-transmitted vs. Re-digitized)

A cross-class traces analysis is conducted to build a more resilient detection system to identify sophisticated digital manipulations.

## Model Architecture

The model consists of a shared backbone based on a pre-trained Vision Transformer (ViT-B/16), finetuned with a selective parameter unfreezing strategy.

- **Backbone**: `vit_b_16`
- **Classification Heads**: Two independent classification heads, with a dropout of 0.3 and a ReLU activation, map the extracted 768-dimensional feature vector into the respective output space

### Combined Loss Function
The model is trained to minimize a linear combination of two Cross-Entropy losses:

$$
L = \alpha \cdot L_{rf} + (1 - \alpha) \cdot L_{trans}
$$

with $ \alpha \in {0.0, 0.2, 0.5, 0.8, 1.0} $

## Dataset

The `RRDataset` consists of `.jpg`, `.png` and `.jpeg` extension files. A rule-based text to label conversion is realized to dynamically extract the files for all six possible combinations.

### Data Augmentation Pipeline
- Training set: resized to 256, center cropped to 224 with random horizontal flip ( p = 0.5 ), random rotation ( 10^∘ ) and ImageNet normalization
- Validation/Test set: resized to 256, center cropped to 224 and ImageNet normalization

IMPORTANT NOTE: because of random flip and rotation output metrics and classifications are never the same.
## Project Structure

When running the execution pipeline the following directories are automatically generated:

```
├── data/                     
├── checkpoints/                      
├── models/                          
├── plots/                           
├── metrics/                         
├── CV_project_template.ipynb        
└── README.md      
```                  

## Installation and Requirements
The code is optimized for Google Colab which gives access to CUDA accelerated GPUs.

### Requirements
The core dependencies include:

```
huggingface_hub
torch
torchvision
scikit-learn
pandas
seaborn
matplotlib
tqdm
pillow
```

### Configuration

Key hyperparameters can be set in the `Global` section of the notebook:

```
LEARNING_RATE = 1e-3
EPOCHS = 20
BATCH_SIZE = 64
ALPHA = 0.5       #choose among {0.0, 0.2, 0.5, 0.8, 1.0} 
PATIENCE = 3
```

The pipeline also includes mechanisms for **reproducibility**, **early stopping** and **memory management**.

## Ablation Study

Varying the `alpha` value allows analyzing whether the two tasks compete for representational capacity or complement each other.

| ALPHA Value | Model Focus |
| :--- | :--- |
| **0.0** | Exclusive optimization for the Transformation classification task. |
| **0.5** | Perfectly balanced Multi-Task joint training. |
| **1.0** | Exclusive optimization for the Real/Fake detection task. |

All evaluation data are exported in the `metrics` folder in CSV format.

## Authors and References

* **Li, Chunxiao, et al.** *“Bridging the Gap Between Ideal and Real-world Evaluation: Benchmarking AI-Generated Image Detection in Challenging Scenarios.”* Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.
  * **Official Dataset (Zenodo):** [https://zenodo.org/records/14963880](https://zenodo.org/records/14963880)
  * **Official Repository (GitHub):** [https://github.com/ChunXiaostudy/RRDataset](https://github.com/ChunXiaostudy/RRDataset)
  * **Paper Link (arXiv):** [https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Bridging_the_Gap_Between_Ideal_and_Real-world_Evaluation_Benchmarking_AI-Generated_ICCV_2025_paper.pdf](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Bridging_the_Gap_Between_Ideal_and_Real-world_Evaluation_Benchmarking_AI-Generated_ICCV_2025_paper.pdf)

* **Shao, Rui, Tianxing Wu, and Ziwei Liu.** *“Detecting and grounding multi-modal media manipulation.”* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
  * **Paper Link (arXiv):** [https://openaccess.thecvf.com/content/CVPR2023/papers/Shao_Detecting_and_Grounding_Multi-Modal_Media_Manipulation_CVPR_2023_paper.pdf](https://openaccess.thecvf.com/content/CVPR2023/papers/Shao_Detecting_and_Grounding_Multi-Modal_Media_Manipulation_CVPR_2023_paper.pdf)
