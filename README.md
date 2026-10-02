# Q-Conditioning and Q-Gating: Metadata-Driven Attention Mechanisms for Medical Vision Transformers

![Status](https://img.shields.io/badge/Status-Published-green)
![Journal](https://img.shields.io/badge/Journal-Information%20(MDPI)-blue)
![Special Issue](https://img.shields.io/badge/Special%20Issue-Advanced%20Neural%20Architectures-lightgrey)
![Python](https://img.shields.io/badge/Python-3.10-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red)
![HMB-Helpers](https://img.shields.io/badge/HMB--Helpers-PyPI-orange)

## Publication Details

* **Journal**: Information (MDPI)
* **Volume/Issue**: 2026, 17(10), 973
* **DOI**: [https://doi.org/10.3390/info17100973](https://doi.org/10.3390/info17100973)
* **Special Issue**: Advanced Neural Architectures for Multimodal Understanding and Generation
* **Academic Editors**: Gerasimos Vonitsanos, John Garofalakis
* **Publication History**: Received: 16 July 2026 | Revised: 19 September 2026 | Accepted: 25 September 2026 |
  Published: 1 October 2026

## Abstract

**Background**: Vision Transformers (ViTs) have demonstrated efficacy in visual recognition tasks through global
dependency modeling via self-attention. However, standard ViT architectures lack native mechanisms for integrating
external clinical metadata (e.g., patient demographics, lesion characteristics, image quality indicators) into the
visual processing pipeline, limiting their utility in multi-modal diagnostic scenarios.

**Methods**: We propose two novel attention mechanisms: Q-Conditioning and Q-Gating, designed for efficient metadata
fusion within the transformer framework. In Q-Conditioning, metadata is projected into the query space and added to the
query vector, guiding attention toward contextually relevant regions during early computation. In Q-Gating, a learned
gate modulates computed attention scores, enabling soft, differentiable control over token interactions. We further
introduce a hybrid attention framework wherein standard, Q-Conditioning, and Q-Gating heads coexist within the same
model.

**Results**: We evaluated our approach on four medical imaging benchmarks (PAD-UFES-20, TissueNet, PH², and NDB-UFES).
Specific configurations, particularly Q-Gating and hybrid variants (e.g., QC_QG, QG_S), achieved >95.5% accuracy on
PAD-UFES-20 and >92% average score on PH², significantly outperforming image-only baseline models (p<0.001). On
TissueNet, where auxiliary inputs consist of PhikonV2 foundation-model embeddings rather than raw clinical metadata, all
configurations exceeded 99% accuracy; however, we note that these gains are substantially attributable to the pretrained
encoder and are reported separately from the primary clinical-metadata claims. Attention concentration metrics,
encompassing both theoretical conditional entropy reduction and empirical absolute entropy dispersion, correlated
strongly with diagnostic performance gains (ρ=0.89), indicating improved focus on diagnostically salient regions while
avoiding spurious attention collapse.

**Conclusions**: Metadata-aware attention mechanisms enhance both performance and interpretability in domains where
auxiliary information is critical. This work provides a scalable extension to ViTs that supports multi-input modeling
while preserving architectural integrity and computational efficiency.

**Keywords**: attention mechanisms; medical image analysis; metadata integration; vision transformers (ViTs)

## Overview

Standard Vision Transformers (ViTs) process images as patch sequences. They do not natively integrate external clinical
metadata, which limits their utility in multi-modal diagnostic scenarios. This repository provides the official
implementation of **Q-Conditioning** and **Q-Gating**. These novel attention mechanisms integrate metadata directly into
the transformer framework to guide spatial attention and modulate token dependencies.

* **Q-Conditioning**: Projects metadata into the query space. It adds the projection to the query vector to direct
  attention to contextually relevant regions during early computation.
* **Q-Gating**: Uses a learned gate derived from query-metadata interactions. It modulates computed attention scores via
  a sigmoid activation to allow soft control over token dependencies.
* **Hybrid Attention Framework**: Permits standard, Q-Conditioning, and Q-Gating heads to coexist within the same
  multi-head attention block via explicit head-type allocation.

## Authors

* **Hossam Magdy Balaha** (Bioengineering Department, J.B. Speed School of Engineering, University of Louisville,
  Louisville, KY 40292, USA; Computer Science and Systems Department, Faculty of Engineering, Mansoura University,
  Mansoura 35516, Egypt)
* **Ahmed Sharafeldeen** (Mathematics and Computer Science Department, Louisiana State University of Alexandria,
  Alexandria, LA 71302, USA)
* **Magdy Hassan Balaha** (Department of Obstetrics and Gynecology, Faculty of Medicine, Tanta University, Tanta 31527,
  Egypt)

## Key Contributions

* Metadata-aware attention mechanisms that preserve gradient flow and architectural efficiency.
* A theoretical analysis of attention concentration that formalizes entropy reduction under explicit metadata-region
  correlation assumptions.
* Extensive clinical validation across three medical imaging benchmarks (PAD-UFES-20, PH², NDB-UFES) with patient-wise
  splits and ten independent trials per configuration.
* A dedicated feature-fusion ablation on the TissueNet dataset using PhikonV2 foundation-model embeddings, isolated from
  primary clinical-metadata claims.
* Comprehensive ablation studies covering image perturbations, metadata missingness, computational complexity, and
  interpretability via attention entropy and mask-based IoU quantification.
* Extensive computational complexity benchmarking against emulated State-of-the-Art (SOTA) Vision Transformer variants.

## The HMB-Helpers Package

This repository relies heavily on the **HMB-Helpers** package, a foundational utility library developed by the authors
to streamline PyTorch training pipelines, statistical analysis, and complex visualizations.

The package can be installed via PyPI and its source code and documentation are available at the following links:

* **PyPI**: [https://pypi.org/project/hmb-helpers/](https://pypi.org/project/hmb-helpers/)
* **Documentation**: [https://hmb-helpers-package.readthedocs.io/](https://hmb-helpers-package.readthedocs.io/)
* **GitHub Repository
  **: [https://github.com/HossamBalaha/HMB-Helpers-Package](https://github.com/HossamBalaha/HMB-Helpers-Package)

Key modules utilized from this package include:

* `HMB.Utils`: General utilities, YAML parsing, and numeric formatting.
* `HMB.Initializations`: Random seed management (`SeedEverything`) and CUDA validation.
* `HMB.PyTorchHelper`: Model profiling, latency measurement, FLOPs computation, and perturbation logic.
* `HMB.StatisticalAnalysisHelper`: Statistical tests, distribution analysis, and automated LaTeX table generation.
* `HMB.PlotsHelper`: Advanced visualization, bubble plots, and complexity comparison charts.
* `HMB.PerformanceMetrics`: Confusion matrix derivation and weighted metric calculation.
* `HMB.DatasetsHelper`: Raw image folder traversal and path extraction.

## Datasets

The pipeline evaluates the proposed mechanisms on four distinct medical imaging benchmarks:

* **PAD-UFES-20**: Smartphone-collected clinical skin lesion images paired with patient demographics.
* **PH²**: Dermoscopic images of melanocytic lesions with expert-derived structured clinical assessments.
* **NDB-UFES**: Histopathological images and patient data for oral cancer and leukoplakia.
* **TissueNet (Feature-Fusion Ablation)**: High-resolution microscopic slide images of uterine cervical tissue biopsies,
  utilizing PhikonV2 deep feature embeddings rather than raw clinical metadata.

## Repository Structure

The codebase is organized into sequential execution steps, core utility modules, and configuration assets to ensure full
reproducibility. File names and dictionary keys follow strict CamelCase conventions.

```text
├── Experiments/                    # Output directory for all experimental results.
├── utils/                          # Pipeline-specific utility modules.
│   ├── DatasetHelpers.py           # Data loading, preprocessing, and metadata dictionary mapping.
│   ├── Handler.py                  # Training orchestration, evaluation, and missingness ablation.
│   ├── ProposalHelpers.py          # ViT architecture, custom attention heads, and baseline models.
│   └── Utils.py                    # Argument parsing, missingness logic, and helper functions.
├── assets/                         # Configuration files.
│   └── hparams.json                # Hyperparameter presets with CamelCase keys.
├── Datasets/                       # Root directory for all benchmark datasets.
├── Step1_HViT_Training.py          # Step 1: Model training and metadata missingness ablation.
├── Step2_HViT_Evaluation.py        # Step 2: Performance aggregation and statistical plotting.
├── Step3_HViT_Ablations.py         # Step 3: Robustness against image perturbations.
├── Step4_HViT_Complexity.py        # Step 4: FLOPs, parameters, and latency profiling.
└── Step5_HViT_Interpretability.py  # Step 5: Attention heatmaps, entropy, and mask-based IoU.
```

## Installation

We recommend using a Conda environment with Python 3.10.

```bash
# Create a new conda environment with Python 3.10.
conda create -n hvit python=3.10 -y
# Activate the newly created environment.
conda activate hvit
# Install PyTorch with CUDA 11.8 support.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Install the HMB-Helpers foundational package.
pip install hmb-helpers
# Install the required data science and utility packages.
pip install pandas numpy scikit-learn matplotlib opencv-python pyyaml tqdm pillow
```

## Dataset Preparation and Loading Logic

The pipeline expects images to be organized in standard folder structures, with metadata stored in CSV or Pickle
formats. Create a `Datasets` directory in the root folder and structure your data as follows:

```text
Datasets/
├── PH2Dataset/
│   ├── Images/
│   ├── Masks/                      # Ground-truth lesion masks for Step 5 IoU evaluation.
│   └── Metadata.csv
├── SkinCancer(PAD-UFES-20)/
│   ├── OrganizedImages/
│   └── Metadata.csv
├── NDBUFES/
│   ├── OrganizedImages/
│   └── ClinicalData.csv
└── TissueNet/
    ├── Tiles/
    └── UterineCervixPhikonV2.p
```

### Metadata Formatting Rules

Ensure that the metadata files contain a `Filename` column and a `Class` column, with auxiliary features occupying the
intermediate columns. Target leakage fields (e.g., `clinical diagnosis`, `consensus diagnosis`) must be explicitly
excluded prior to running the pipeline.

### Internal Data Loading Mechanism

The `utils/DatasetHelpers.py` module handles data ingestion via the `DataLoader` function:

1. **Categorical Encoding**: Non-numeric metadata columns are automatically processed using
   `sklearn.preprocessing.LabelEncoder`.
2. **Normalization**: All metadata features are scaled to a `[0, 1]` range using `MinMaxScaler`.
3. **Dictionary Mapping**: Metadata is stored in a dictionary using a composite key format: `"{ClassLabel}_{Filename}"`.
4. **Stratified Splitting**: The dataset is split into training and testing sets using `train_test_split` with
   stratification based on class labels.
5. **Transformations**: Training data undergoes random horizontal/vertical flips and rotations, while testing data only
   undergoes deterministic resizing and tensor conversion.

## Architectural Framework

The core Vision Transformer architecture is defined in `utils/ProposalHelpers.py`. The framework supports dynamic head
allocation within the `MultiHeadAttention` module.

### Embedding Modules

* **`PatchEmbeddings`**: Handles image tokenization via a convolutional projection (`nn.Conv2d`) where kernel size and
  stride equal the patch size.
* **`StandardEmbeddings`**: Combines patch embeddings with a learnable `[CLS]` token and positional encodings, followed
  by dropout.

### Attention Head Variants

All attention heads inherit from `AttentionHeadParent`, which defines the base Query, Key, and Value linear projections.

* **`StandardAttentionHead`**: Baseline scaled dot-product attention without metadata integration.
* **`QConditioningAttentionHead`**: Injects a linear projection of the metadata directly into the Query space.
* **`KeyConditioningAttentionHead`**: Ablation baseline that injects metadata into the Key space.
* **`ValueConditioningAttentionHead`**: Ablation baseline that injects metadata into the Value space.
* **`QGatingAttentionHead`**: Concatenates Query and Metadata projections, passes them through a linear layer, and
  applies a sigmoid gate to modulate the raw attention scores.

### Transformer Blocks and Encoders

* **`MultiHeadAttention`**: Dynamically instantiates a list of attention heads based on the provided `headTypes`
  configuration.
* **`MLP`**: A standard multi-layer perceptron with GELU activation.
* **`Block`**: A single transformer block combining Multi-Head Attention, Layer Normalization, and the MLP with residual
  connections.
* **`Encoder`**: Stacks multiple `Block` modules and optionally returns attention probabilities for interpretability.

### Classification Wrappers and Baselines

* **`ViTForClassification`**: The primary wrapper that combines `StandardEmbeddings`, `Encoder`, and a linear classifier
  head.
* **`LateFusionViT`**: Baseline architecture that concatenates raw metadata to the `[CLS]` token representation at the
  classifier head.
* **`EarlyFusionViT`**: Baseline architecture that projects metadata into the hidden dimension and adds it to the
  `[CLS]` token prior to the encoder.
* **`PhikonV2LinearProbe`**: A specialized linear classifier designed to operate exclusively on frozen PhikonV2
  embeddings for the TissueNet ablation.

## Hyperparameter Configuration

Model configurations are managed via `assets/hparams.json`. The file contains an array of presets ranging from
`Preset_01_Tiny` to `Preset_12_Robust`. You can define multiple presets and toggle them using the `"Active"` boolean
flag. The pipeline will iterate through all active presets during Step 1.

```json
[
  {
    "Name": "Preset_06_Base",
    "Active": true,
    "PatchSize": 16,
    "HiddenSize": 768,
    "NumAttentionHeads": 6,
    "NumHiddenLayers": 12,
    "IntermediateSize": 3072,
    "HiddenDropoutProb": 0.1,
    "AttentionProbsDropoutProb": 0.1,
    "QkvBias": true,
    "InitializerRange": 0.02,
    "Optimizer": "AdamW",
    "OptimizerParams": {
      "LearningRate": 0.0003,
      "WeightDecay": 0.05,
      "Epsilon": 1e-8
    },
    "Scheduler": "CosineAnnealingLR",
    "SchedulerParams": {
      "TMax": "EPOCHS",
      "EtaMin": 1e-6
    }
  }
]
```

Below is the formal Python logic used to parse these configurations:

```python
# Import the JSON module to handle configuration files.
import json


# Define a function to parse the hyperparameter presets.
def ParseHParams(filePath):
  # Open the JSON file in read mode.
  with open(filePath, "r") as fileObject:
    # Load the JSON data into a list of dictionaries.
    presetsList = json.load(fileObject)

  # Iterate through each preset in the list.
  for presetDict in presetsList:
    # Check if the current preset is marked as active.
    if (presetDict["Active"] == True):
      # Extract the hidden size using a CamelCase key.
      hiddenSize = presetDict["HiddenSize"]
      # Extract the optimizer name using a CamelCase key.
      optimizerName = presetDict["Optimizer"]
      # Print the active configuration details to the console.
      print(f"Loaded active preset with hidden size {hiddenSize} and optimizer {optimizerName}.")
      # Return the active preset dictionary.
      return presetDict

  # Return None if no active preset is found.
  return None
```

## Pipeline Execution Flow

The repository follows a strict sequential execution flow to ensure statistical validity and reproducibility:

1. **Step 1 (Training)**: Trains all architectural variants across multiple independent trials. Computes inverse
   frequency class weights (`1.0 / (classCounts + 1e-8)`) to handle class imbalance. Saves model weights (
   `BestModel.pth`, `LastModel.pth`), training histories (`History.csv`), and `Config.yaml` files. Generates ROC, PRC,
   and Confusion Matrix plots.
2. **Step 2 (Evaluation)**: Aggregates predictions from Step 1, computes weighted performance metrics, and generates
   `Evaluation_Summary.csv`. Utilizes `HMB.StatisticalAnalysisHelper` to generate extensive distribution plots (
   Histograms, Density, Box, Violin, QQ, CDF, ECDF, Swarm, etc.) and automated LaTeX tables. Identifies the single
   best-performing model for subsequent ablations.
3. **Step 3 (Perturbations)**: Loads the best model identified in Step 2 and evaluates its robustness against Gaussian,
   JPEG, Speckle, and Salt & Pepper noise at intensity levels `[0.05, 0.1, 0.2, 0.3, 0.4, 0.5]`.
4. **Step 4 (Complexity)**: Instantiates models dynamically using `hparams.json` to measure FLOPs, parameter counts, and
   inference latency. Emulates SOTA architectures (e.g., Cross-Attention, DeiT, CrossViT, CvT, Swin-Transformer) by
   modifying hyperparameters and replacing attention heads.
5. **Step 5 (Interpretability)**: Extracts CLS-to-Patch attention weights from a specific checkpoint to compute Shannon
   entropy and spatial Intersection over Union (IoU@k) against ground-truth masks. Generates heatmap overlays using
   OpenCV colormaps.

## Detailed Usage Guide & Command-Line Options

### Step 1: Model Training & Missingness Options

| Argument                     | Type    | Default                      | Description                                                           |
|:-----------------------------|:--------|:-----------------------------|:----------------------------------------------------------------------|
| `--experimentsFolder`        | String  | `"Results"`                  | Base directory to save all experimental outputs.                      |
| `--projectKeyword`           | String  | `"PH2"`                      | Keyword used for naming the project subdirectories.                   |
| `--hparamsFile`              | String  | `"assets/hparams.json"`      | Path to the JSON file containing hyperparameter presets.              |
| `--datasetPath`              | String  | `"PH2 Dataset/Images"`       | Path to the root directory of the image dataset.                      |
| `--embeddedPath`             | String  | `"PH2 Dataset/Metadata.csv"` | Path to the CSV or Pickle file containing embedded metadata.          |
| `--metadataDim`              | Integer | `6`                          | Dimensionality of the metadata feature vector.                        |
| `--batchSize`                | Integer | `16`                         | Number of samples per training batch.                                 |
| `--imageSize`                | Integer | `128`                        | Spatial dimension to which input images are resized.                  |
| `--numChannels`              | Integer | `3`                          | Number of image channels (1 for grayscale, 3 for RGB).                |
| `--noOfEpochs`               | Integer | `128`                        | Total number of training epochs.                                      |
| `--noOfTrials`               | Integer | `1`                          | Number of independent trials to run per architecture.                 |
| `--mustCuda`                 | Flag    | `False`                      | Enforces execution on a CUDA-enabled GPU.                             |
| `--clean`                    | Flag    | `False`                      | Deletes previous outputs in the project directory before starting.    |
| `--dpi`                      | Integer | `720`                        | Resolution (Dots Per Inch) for saved matplotlib figures.              |
| `--metadataMissingEvalRates` | List    | `[]`                         | List of missingness rates (0.0 to 1.0) to evaluate post-training.     |
| `--metadataMissingMode`      | String  | `"Zero"`                     | Strategy for replacing missing metadata (`Zero`, `Mean`, `Gaussian`). |
| `--metadataMissingScope`     | String  | `"Feature"`                  | Scope of missingness application (`Feature` or `Sample`).             |
| `--metadataMissingNoiseStd`  | Float   | `0.05`                       | Standard deviation for Gaussian noise replacement mode.               |
| `--metadataMissingTrainRate` | Float   | `0.0`                        | Rate of metadata corruption applied dynamically during training.      |

### Step 2: Performance Aggregation & Statistical Analysis

| Argument              | Type    | Default     | Description                                                   |
|:----------------------|:--------|:------------|:--------------------------------------------------------------|
| `--experimentsFolder` | String  | `"Results"` | Base directory containing the experimental outputs.           |
| `--projectKeyword`    | String  | `"PH2"`     | Keyword identifying the specific project to analyze.          |
| `--dpi`               | Integer | `720`       | Resolution for generated statistical plots and bubble charts. |

### Step 3: Robustness Against Image Perturbations

| Argument              | Type    | Default                      | Description                                                    |
|:----------------------|:--------|:-----------------------------|:---------------------------------------------------------------|
| `--experimentsFolder` | String  | `"Results"`                  | Base directory containing the experimental outputs.            |
| `--projectKeyword`    | String  | `"PH2"`                      | Keyword identifying the specific project to analyze.           |
| `--datasetPath`       | String  | `"PH2 Dataset/Images"`       | Path to the root directory of the image dataset.               |
| `--embeddedPath`      | String  | `"PH2 Dataset/Metadata.csv"` | Path to the metadata file.                                     |
| `--maxSamples`        | Integer | `200`                        | Maximum number of test images to evaluate for perturbations.   |
| `--mustCuda`          | Flag    | `False`                      | Enforces execution on a CUDA-enabled GPU.                      |
| `--doBestOnly`        | Flag    | `False`                      | Evaluates only the best-performing model identified in Step 2. |
| `--dpi`               | Integer | `720`                        | Resolution for generated perturbation plots.                   |

### Step 4: Computational Complexity & Latency Profiling

| Argument              | Type    | Default                 | Description                                         |
|:----------------------|:--------|:------------------------|:----------------------------------------------------|
| `--experimentsFolder` | String  | `"Results"`             | Base directory to save complexity outputs.          |
| `--hparamsFile`       | String  | `"assets/hparams.json"` | Path to the hyperparameters configuration file.     |
| `--hparamName`        | String  | `None`                  | Specific named preset to load from the JSON file.   |
| `--projectKeyword`    | String  | `"PH2-v2"`              | Keyword used for naming the project subdirectories. |
| `--imageSize`         | Integer | `128`                   | Spatial dimension of the input images.              |
| `--numChannels`       | Integer | `3`                     | Number of image channels (1 or 3).                  |
| `--metadataDim`       | Integer | `6`                     | Dimensionality of the metadata feature vector.      |
| `--runs`              | Integer | `100`                   | Number of forward passes to average latency.        |
| `--warmup`            | Integer | `10`                    | Number of warmup forward passes before timing.      |
| `--mustCuda`          | Flag    | `False`                 | Enforces execution on a CUDA-enabled GPU.           |
| `--dpi`               | Integer | `720`                   | Resolution for generated complexity plots.          |

### Step 5: Interpretability & Mask IoU Options

| Argument              | Type    | Default                      | Description                                                            |
|:----------------------|:--------|:-----------------------------|:-----------------------------------------------------------------------|
| `--experimentsFolder` | String  | `"Results"`                  | Base directory containing the experimental outputs.                    |
| `--projectKeyword`    | String  | `"PH2"`                      | Keyword identifying the specific project to analyze.                   |
| `--datasetPath`       | String  | `"PH2 Dataset/Images"`       | Path to the root directory of the image dataset.                       |
| `--embeddedPath`      | String  | `"PH2 Dataset/Metadata.csv"` | Path to the metadata file.                                             |
| `--hparamsFile`       | String  | `"assets/hparams.json"`      | Path to the hyperparameters configuration file.                        |
| `--hparamName`        | String  | `None`                       | Specific named preset to load from the JSON file.                      |
| `--checkpoint`        | String  | `None`                       | Path to the specific model checkpoint (`.pth`) to load.                |
| `--imageSize`         | Integer | `128`                        | Spatial dimension to which input images are resized.                   |
| `--patchSize`         | Integer | `None`                       | Patch size override; if None, reads from hyperparameters.              |
| `--maskDir`           | String  | `None`                       | Directory containing ground-truth lesion masks for quantitative IoU.   |
| `--topK`              | List    | `[1, 5, 10, 20]`             | List of $k$ values for Intersection over Union (IoU) calculations.     |
| `--maxSamples`        | Integer | `200`                        | Maximum number of images to process for heatmap generation.            |
| `--requireMask`       | Flag    | `False`                      | Skips images that lack a corresponding ground-truth segmentation mask. |
| `--mustCuda`          | Flag    | `False`                      | Enforces execution on a CUDA-enabled GPU.                              |
| `--dpi`               | Integer | `720`                        | Resolution for generated heatmap and entropy plots.                    |

## Dataset-Specific Execution Examples

### Step 1: Model Training

```bash
# Execute the training pipeline for the PH2 dataset.
python Step1_HViT_Training.py \
  --projectKeyword "PH2" \
  --datasetPath "Datasets/PH2Dataset/Images" \
  --embeddedPath "Datasets/PH2Dataset/Metadata.csv" \
  --metadataDim 6 \
  --experimentsFolder "Experiments" \
  --noOfEpochs 128 \
  --noOfTrials 10 \
  --mustCuda \
  --dpi 720

# Execute the training pipeline for the PAD-UFES-20 dataset.
python Step1_HViT_Training.py \
  --projectKeyword "PAD-UFES-20" \
  --datasetPath "Datasets/SkinCancer(PAD-UFES-20)/OrganizedImages" \
  --embeddedPath "Datasets/SkinCancer(PAD-UFES-20)/Metadata.csv" \
  --metadataDim 22 \
  --experimentsFolder "Experiments" \
  --noOfEpochs 128 \
  --noOfTrials 10 \
  --mustCuda \
  --dpi 720

# Execute the training pipeline for the NDB-UFES dataset.
python Step1_HViT_Training.py \
  --projectKeyword "NDB-UFES" \
  --datasetPath "Datasets/NDB-UFES/OrganizedImages" \
  --embeddedPath "Datasets/NDB-UFES/ClinicalData.csv" \
  --metadataDim 23 \
  --experimentsFolder "Experiments" \
  --noOfEpochs 128 \
  --noOfTrials 10 \
  --mustCuda \
  --dpi 720

# Execute the training pipeline for the TissueNet dataset (Feature-Fusion).
python Step1_HViT_Training.py \
  --projectKeyword "UterineCervixPhikonV2" \
  --datasetPath "Datasets/TissueNet/Tiles" \
  --embeddedPath "Datasets/TissueNet/UterineCervixPhikonV2.p" \
  --metadataDim 1024 \
  --experimentsFolder "Experiments" \
  --noOfEpochs 128 \
  --noOfTrials 10 \
  --mustCuda \
  --dpi 720
```

### Step 2: Statistical Evaluation and LaTeX Generation

```bash
# Aggregate results and generate statistical plots for the PH2 project.
python Step2_HViT_Evaluation.py \
  --projectKeyword "PH2" \
  --experimentsFolder "Experiments" \
  --dpi 720
```

### Step 3: Robustness Against Image Perturbations

```bash
# Perform ablation studies on the best-performing PH2 model.
python Step3_HViT_Ablations.py \
  --projectKeyword "PH2" \
  --datasetPath "Datasets/PH2Dataset/Images" \
  --embeddedPath "Datasets/PH2Dataset/Metadata.csv" \
  --experimentsFolder "Experiments" \
  --doBestOnly \
  --mustCuda \
  --dpi 720
```

### Step 4: Complexity Profiling against SOTA Emulations

```bash
# Measure FLOPs, parameters, and latency for all architectural variants.
python Step4_HViT_Complexity.py \
  --hparamsFile "assets/hparams.json" \
  --hparamName "Preset_06_Base" \
  --projectKeyword "PH2" \
  --imageSize 256 \
  --metadataDim 6 \
  --runs 100 \
  --warmup 25 \
  --mustCuda
```

### Step 5: Interpretability (with Mask IoU)

```bash
# Execute the interpretability pipeline for the PH2 dataset with ground-truth masks.
python Step5_HViT_Interpretability.py \
  --projectKeyword "PH2" \
  --datasetPath "Datasets/PH2Dataset/Images" \
  --embeddedPath "Datasets/PH2Dataset/Metadata.csv" \
  --maskDir "Datasets/PH2Dataset/Masks" \
  --checkpoint "Experiments/PH2/Step1-Training/QC_QG_Preset_06_Base/Trial_1/BestModel.pth" \
  --hparamsFile "assets/hparams.json" \
  --hparamName "Preset_06_Base" \
  --imageSize 128 \
  --topK 1 5 10 20 \
  --requireMask \
  --maxSamples 200 \
  --mustCuda \
  --dpi 720
```

## Advanced Logic Implementations

### Metadata Missingness Implementation

The pipeline natively supports simulating missing clinical metadata during both training and evaluation. Below is the
formal implementation logic utilized within the handler:

```python
# Import the PyTorch library for tensor operations.
import torch


# Define the function to apply metadata missingness corruption.
def ApplyMetadataMissingness(metadataTensor, missingnessRate, missingnessMode):
  # Clone the metadata tensor to avoid modifying the original data.
  corruptedTensor = metadataTensor.clone()
  # Check if the missingness rate is greater than zero.
  if (missingnessRate > 0.0):
    # Generate a random boolean mask based on the missingness rate.
    missingnessMask = torch.rand(corruptedTensor.size()) < missingnessRate
    # Check if the replacement mode is set to zero.
    if (missingnessMode == "Zero"):
      # Fill the masked entries with zero values.
      corruptedTensor = corruptedTensor.masked_fill(missingnessMask, 0.0)
    # Check if the replacement mode is set to mean.
    elif (missingnessMode == "Mean"):
      # Compute the mean value across the batch dimension.
      batchMean = corruptedTensor.mean(dim=0, keepdim=True)
      # Replace the masked entries with the computed batch mean.
      corruptedTensor = torch.where(missingnessMask, batchMean, corruptedTensor)
  # Return the corrupted metadata tensor.
  return corruptedTensor
```

### Attention Extraction and Entropy Calculation

Interpretability relies on extracting the attention weights from the `[CLS]` token to the spatial patches. Below is the
extraction logic:

```python
# Import the NumPy and PyTorch libraries for tensor operations.
import numpy as np
import torch


# Define the function to extract CLS-to-patches attention vectors.
def ExtractClsToPatchesAttention(allAttentions, layerIndex, aggregateHeads):
  # Select the requested layer attentions from the provided list.
  attentionTensor = allAttentions[layerIndex]
  # Check if the attention object is a NumPy array.
  if (isinstance(attentionTensor, np.ndarray)):
    # Convert the NumPy array to a PyTorch tensor.
    attentionTensor = torch.from_numpy(attentionTensor)
  # Extract the CLS token row to patch tokens.
  clsToPatches = attentionTensor[:, :, 0, 1:]
  # Check if the aggregation method is set to mean.
  if (aggregateHeads == "Mean"):
    # Compute the mean across the head dimension.
    attentionVector = clsToPatches.mean(dim=1)
  # Check if the aggregation method is set to sum.
  elif (aggregateHeads == "Sum"):
    # Compute the sum across the head dimension.
    attentionVector = clsToPatches.sum(dim=1)
  # Convert the tensor to a NumPy array on the CPU.
  attentionVector = attentionVector.cpu().numpy()
  # Normalize each sample attention vector to sum to one.
  attentionVector = attentionVector / (attentionVector.sum(axis=1, keepdims=True) + 1e-12)
  # Return the normalized attention vectors.
  return attentionVector
```

## Citation

If you use this code or our findings in your research, please cite our published manuscript.

```bibtex
@article{balaha2026qconditioning,
    title = {Q-Conditioning and Q-Gating: Metadata-Driven Attention Mechanisms for Medical Vision Transformers},
    author = {Balaha, Hossam Magdy and Sharafeldeen, Ahmed and Balaha, Magdy Hassan},
    journal = {Information},
    volume = {17},
    number = {10},
    pages = {973},
    year = {2026},
    publisher = {MDPI},
    doi = {10.3390/info17100973},
    note = {Special Issue: Advanced Neural Architectures for Multimodal Understanding and Generation}
}
```

## Contact and Links

* **Hossam Magdy Balaha**: [Personal Website and CV](https://hossambalaha.github.io/)
* **Correspondence**: hmbala01@louisville.edu

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
