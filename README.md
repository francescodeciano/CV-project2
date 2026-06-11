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

with $`\alpha \in \{0.0, 0.5, 1.0\}`$

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
├── CV2026_project.ipynb        
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
ALPHA = 0.5       #choose among {0.0, 0.5, 1.0} 
PATIENCE = 3
```

The pipeline also includes mechanisms for **reproducibility**, **early stopping** and **memory management**.

## Ablation Study

Varying the `ALPHA` value allows analyzing whether the two tasks compete for representational capacity or complement each other.

| ALPHA Value | Model Focus |
| :--- | :--- |
| **0.0** | Exclusive optimization for the Transformation classification task. |
| **0.5** | Perfectly balanced Multi-Task joint training. |
| **1.0** | Exclusive optimization for the Real/Fake detection task. |

To run the ablation study, change the `ALPHA` value in the `GLOBALS` section and re-execute the following sections: `GLOBALS`, `DATA` (only the cell where the DataLoaders are defined, skip the download and the extraction cells), `NETWORK` (only the model initialization cell), `TRAIN` and `TEST` (only the cell where the test function is called). They are however highlighted by the commente `IF YOU ARE TESTING A DIFFERENT VALUE OF ALPHA EXECUTE AGAIN THIS CELL` at the beginning of the cell.

Make sure to test the code with all three values of `ALPHA` and that the models are all saved in the colab local directory to allow the function which shows the results of the ablation study to be executed correctly.

## Results 
After each run, all evaluation outputs are automatically saved in the Colab temporary directory. The output filenames are built dynamically from the global variable `ALPHA` so that different ablation configurations do not overwrite each other.

**Plots** (`plots/`): the file `plot_alpha{ALPHA}.png` contains the two confusion matrices produced at test time. The blue matrix represents the performance on the Real/Fake task, the green matrix reports the results for the Transformation classification task. The plot is exported as a single high resolution image (300 dpi).

**Metrics** (`metrics/`): the file `breakdown_alpha{ALPHA}.csv contains a cross-class accuracy table showing Real/Fake accuracy broken down by transformation type (Original, Internet, Re-digitized).

**Models** (`models/`): the best-performing checkpoint is saved as `final_model_alpha{ALPHA}.pt` and reloaded at test time. Intermediate epoch checkpoints are stored in `checkpoints/` during training and deleted at the end to save disk space.


## Authors and References

* **Li, Chunxiao, et al.** *“Bridging the Gap Between Ideal and Real-world Evaluation: Benchmarking AI-Generated Image Detection in Challenging Scenarios.”* Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.
  * **Official Dataset (Zenodo):** [https://zenodo.org/records/14963880](https://zenodo.org/records/14963880)
  * **Official Repository (GitHub):** [https://github.com/ChunXiaostudy/RRDataset](https://github.com/ChunXiaostudy/RRDataset)
  * **Paper Link (arXiv):** [https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Bridging_the_Gap_Between_Ideal_and_Real-world_Evaluation_Benchmarking_AI-Generated_ICCV_2025_paper.pdf](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Bridging_the_Gap_Between_Ideal_and_Real-world_Evaluation_Benchmarking_AI-Generated_ICCV_2025_paper.pdf)

* **Shao, Rui, Tianxing Wu, and Ziwei Liu.** *“Detecting and grounding multi-modal media manipulation.”* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
  * **Paper Link (arXiv):** [https://openaccess.thecvf.com/content/CVPR2023/papers/Shao_Detecting_and_Grounding_Multi-Modal_Media_Manipulation_CVPR_2023_paper.pdf](https://openaccess.thecvf.com/content/CVPR2023/papers/Shao_Detecting_and_Grounding_Multi-Modal_Media_Manipulation_CVPR_2023_paper.pdf)
