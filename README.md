# MSGSNet

MSGSNet is a sequence-graph dual-branch collaborative modeling framework for transcription factor binding site (TFBS) prediction that models DNA sequences from complementary semantic and structural viewpoints. 

![image](https://github.com/yamadaanna111/MSGSNet/blob/main/model.jpg)

---

# Key Features

- Sequence Semantic Branch
  - Multi-scale convolutional feature extraction with kernel sizes of 3, 7, and 11.
  - Bidirectional LSTM (Bi-LSTM) for long-range dependency modeling.
  - Hierarchical Bidirectional Attention (HBA) to jointly capture local contextual patterns and global sequence dependencies.
- Multi-scale Graph Branch
  - Construction of De Bruijn graphs from DNA sequences at multiple k-mer scales.
  - Graph representation learning using Chebyshev Graph Convolution Networks (ChebConv).
  - Graph Channel Attention (GCA) for adaptive channel-wise feature refinement.
  - Pairwise Cross-Attention (PCA) for cross-scale interaction among graph representations.
- Sequence–Graph Fusion
  - Bidirectional cross-attention mechanism for interactive fusion between sequence semantic features and graph structural features.
  - Multi-layer perceptron (MLP) classifier for final TFBS prediction.

# Repository Structure

```
MSGSNet
│
├── Datasets/
│   └── 165 ChIP-seq/
│       ├── wgEncodeAwgTfbsSydhK562CfosUniPk/
│       │   ├── train.data
│       │   └── test.data
│       └── FILESLIST.txt
│
├── experiments/
│   └── dataset_name/
│       ├── model/
│       │   └── best_model.pt
│       └── graph_sce_evaluation_results.csv
│
├── model/
│   ├── MSGSNet.py
│   ├── create_graph_dataset.py
│   ├── create_sce_dataset.py
│   ├── seq2graph.py
│   └── train.py
│
├── environment.yml
├── Supplementary Material.pdf
├── model.jpg
└── README.md
```

------

# File Description

## MSGSNet.py

Implementation of the complete MSGSNet framework, including:

### Sequence Semantic Branch

- Multi-scale CNN
- Bi-LSTM
- Hierarchical Bidirectional Attention (HBA)

### Multi-scale Graph Branch

- ChebConv-based graph representation learning
- Graph Channel Attention (GCA)
- Pairwise Cross-Attention (PCA)

### Sequence–Graph Fusion

- Bidirectional Cross-Attention
- MLP classifier

------

## seq2graph.py

Converts DNA sequences into De Bruijn graph representations at different k-mer scales.

------

## create_graph_dataset.py

Constructs graph datasets for the graph branch and generates graph objects used during training.

------

## create_sce_dataset.py

Generates stacked codon-based semantic feature representations from DNA sequences for the sequence branch.

------

## train.py

Training and evaluation entry point for MSGSNet.

------

# Dataset

The benchmark datasets used in this study are derived from 165 ENCODE ChIP-seq datasets curated by Zeng et al., covering:

- 29 transcription factors
- 32 cell lines

Dataset characteristics:

- Sequence length: 101 bp
- Positive samples: DNA fragments centered on ChIP-seq peaks containing experimentally validated TFBSs
- Negative samples: nucleotide-preserving shuffled sequences generated from positive samples
- Class distribution: balanced positive and negative samples

Dataset organization:

```
Datasets/
└── 165 ChIP-seq/
    └── dataset_name/
        ├── train.data
        └── test.data
```

Due to data usage policies, users should prepare datasets following the protocol described in the manuscript or obtain the original datasets from the corresponding ENCODE resources.

------

# Environment Setup

The implementation was developed and tested using:

```
Python 3.10.13
PyTorch 2.1.2
PyTorch Geometric
NumPy 1.26.3
Pandas 2.1.4
Scikit-learn 1.3.2
```

Create the environment using:

```
conda env create -f environment.yml
conda activate msgsnet
```

# Reproducing Experimental Results

To reproduce the reported results:

### Step 1

Create the environment.

```
conda env create -f environment.yml
conda activate msgsnet
```

### Step 2

Prepare the benchmark datasets under:

```
Datasets/165 ChIP-seq/
```

### Step 3

Run model training.

```
python train.py -e 300
```

### Step 4

Check saved checkpoints and evaluation results in:

```
experiments/<dataset_name>/
```

------

# Intended Use

This repository is intended for:

- Reproducible research in TFBS prediction
- Comparative evaluation of TFBS prediction methods
- Development of sequence–graph learning models for regulatory genomics

The current implementation focuses on research reproducibility and experimental evaluation and is not optimized for large-scale production deployment.