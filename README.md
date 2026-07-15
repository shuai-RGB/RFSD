# RFSD: Robust Graph Recommendation via Relation Filtering and Semantic Disentanglement

<p align="center">
  <b>Official PyTorch Implementation of RFSD</b>
</p>

<p align="center">
  <a href="#requirements">Requirements</a> |
  <a href="#data-preparation">Data</a> |
  <a href="#training">Training</a> |
  <a href="#evaluation">Evaluation</a> |
  <a href="#citation">Citation</a>
</p>

## Introduction

This repository provides the official implementation of:

> **Robust Graph Recommendation via Relation Filtering and Semantic Disentanglement**

RFSD is a robust graph representation learning framework for recommendation.
It is designed to address two major problems in graph-based recommender
systems:

1. Unreliable user-user and item-item homogeneous relations.
2. Irrelevant and noisy high-order information propagation over the
   user-item interaction graph.

RFSD consists of three main components:

- **Robust Homogeneous Graph Learning**
- **Semantically Disentangled Message Passing**
- **Semantic Refinement and Optimization**

## Framework

<p align="center">
  <img src="assets/framework.png" width="95%" alt="RFSD Framework">
</p>

<p align="center">
  Overview of the proposed RFSD framework.
</p>

## Highlights

- We construct robust user-user and item-item homogeneous graphs through
  reciprocal neighbor filtering.
- We induce multiple semantic subspaces from the global structures of
  user-side and item-side semantic graphs.
- We perform subspace-specific message propagation to suppress mismatched
  high-order neighborhood information.
- We introduce cross-modal alignment and adaptive semantic gating to
  extract recommendation-relevant semantic information.
- RFSD achieves consistent improvements on Amazon-book, Yelp, and Steam.

## Repository Structure

```text
RFSD/
├── assets/                 # Figures used in README
├── configs/                # Dataset-specific configurations
├── data/                   # Dataset directory
├── preprocessing/          # Data and graph preprocessing
├── scripts/                # Training and evaluation scripts
├── src/                    # Source code
├── main.py                 # Main entry
├── requirements.txt
└── README.md
```

## Requirements

The code has been tested with the following environment:

- Python 3.10
- PyTorch 2.x
- CUDA 11.8 or later
- NumPy
- SciPy
- scikit-learn
- PyYAML
- tqdm

### Option 1: Conda

```bash
conda create -n rfsd python=3.10 -y
conda activate rfsd
pip install -r requirements.txt
```

### Option 2: Environment File

```bash
conda env create -f environment.yml
conda activate rfsd
```

To verify the installation:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

## Data Preparation

We conduct experiments on three public recommendation datasets:

- Amazon-book
- Yelp
- Steam

Place the processed datasets in the following directory:

```text
data/
├── amazon_book/
│   ├── train.txt
│   ├── valid.txt
│   ├── test.txt
│   ├── user_semantic_embeddings.npy
│   └── item_semantic_embeddings.npy
├── yelp/
│   ├── train.txt
│   ├── valid.txt
│   ├── test.txt
│   ├── user_semantic_embeddings.npy
│   └── item_semantic_embeddings.npy
└── steam/
    ├── train.txt
    ├── valid.txt
    ├── test.txt
    ├── user_semantic_embeddings.npy
    └── item_semantic_embeddings.npy
```

More details about dataset preprocessing are provided in
[`data/README.md`](data/README.md).

## Semantic Profile Preparation

RFSD uses textual side information to construct semantic representations
for users and items.

The semantic preprocessing pipeline contains the following steps:

1. Construct user and item textual inputs.
2. Generate recommendation-oriented textual profiles.
3. Encode textual profiles into dense semantic embeddings.
4. Construct user-side and item-side semantic graphs.
5. Perform reciprocal-neighbor relation filtering.
6. Compute spectral representations of semantic graphs.

Example preprocessing command:

```bash
python preprocessing/build_profiles.py \
    --dataset amazon_book \
    --data_path data/amazon_book
```

Construct semantic and co-occurrence graphs:

```bash
python preprocessing/build_semantic_graph.py \
    --dataset amazon_book \
    --config configs/amazon_book.yaml
```

```bash
python preprocessing/build_cooccurrence_graph.py \
    --dataset amazon_book \
    --config configs/amazon_book.yaml
```

Perform spectral decomposition:

```bash
python preprocessing/spectral_decomposition.py \
    --dataset amazon_book \
    --config configs/amazon_book.yaml
```

> Note: Do not upload API keys or private credentials to GitHub. Use
> environment variables or a local `.env` file, and add `.env` to
> `.gitignore`.

## Training

### Train on Amazon-book

```bash
python main.py \
    --config configs/amazon_book.yaml \
    --mode train
```

Alternatively:

```bash
bash scripts/run_amazon_book.sh
```

### Train on Yelp

```bash
python main.py \
    --config configs/yelp.yaml \
    --mode train
```

Alternatively:

```bash
bash scripts/run_yelp.sh
```

### Train on Steam

```bash
python main.py \
    --config configs/steam.yaml \
    --mode train
```

Alternatively:

```bash
bash scripts/run_steam.sh
```

## Evaluation

Evaluate a trained checkpoint:

```bash
python main.py \
    --config configs/amazon_book.yaml \
    --mode test \
    --checkpoint checkpoints/amazon_book/best_model.pt
```

The evaluation reports the following full-ranking metrics:

- Hit Rate: HR@5, HR@10, HR@20
- Recall: Recall@5, Recall@10, Recall@20
- NDCG: NDCG@5, NDCG@10, NDCG@20
- MAP: MAP@5, MAP@10, MAP@20

## Configuration

An example configuration is shown below:

```yaml
dataset: amazon_book
data_path: data/amazon_book

model:
  name: RFSD
  embedding_dim: 256
  num_subspaces: 2
  user_neighbors: 80
  item_neighbors: 10
  user_homogeneous_layers: 3
  item_homogeneous_layers: 1

training:
  optimizer: adam
  learning_rate: 0.0002
  epochs: 2000
  early_stop_patience: 50
  batch_size: 2048
  seed: 2026

loss:
  regularization_weight: 0.0001
  contrastive_weight: 0.1
  clustering_temperature: 1.0
  alignment_temperature: 0.1
```

Dataset-specific configurations are stored in the `configs/` directory.

## Main Results

RFSD is evaluated on Amazon-book, Yelp, and Steam under the full-ranking
evaluation protocol.

| Dataset | HR@20 | Recall@20 | NDCG@20 | MAP@20 |
|---|---:|---:|---:|---:|
| Amazon-book | 0.4543 | 0.2042 | 0.1394 | 0.0798 |
| Yelp | 0.4413 | 0.1445 | 0.0933 | 0.0414 |
| Steam | 0.4408 | 0.1551 | 0.1004 | 0.0475 |

The reported results are averaged over five independent runs with
different random seeds.

## Reproducibility

For reproducible experiments, we recommend explicitly specifying the
random seed:

```bash
python main.py \
    --config configs/amazon_book.yaml \
    --mode train \
    --seed 2026
```

Run experiments using multiple seeds:

```bash
for seed in 2022 2023 2024 2025 2026
do
    python main.py \
        --config configs/amazon_book.yaml \
        --mode train \
        --seed ${seed}
done
```

Please note that small numerical differences may occur due to hardware,
CUDA, and PyTorch versions.

## Pretrained Models

Pretrained checkpoints will be released after the paper is accepted.

Expected checkpoint organization:

```text
checkpoints/
├── amazon_book/
│   └── best_model.pt
├── yelp/
│   └── best_model.pt
└── steam/
    └── best_model.pt
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{dong2026rfsd,
  title     = {Robust Graph Recommendation via Relation Filtering and
               Semantic Disentanglement},
  author    = {Dong, Guoshuai and Zhang, Tingting and Wang, Minghui
               and Li, Yu and Chang, Yi},
  booktitle = {Proceedings of ...},
  year      = {2026}
}
```

The final conference name, page numbers, DOI, and publication information
will be updated after publication.

## Acknowledgements

This implementation benefits from several open-source graph recommendation
projects. We thank the authors and maintainers of these repositories.

## Contact

For questions about the code or paper, please contact:

- Guoshuai Dong: `donggs24@mails.jlu.edu.cn`
- Yu Li: `liyu90@jlu.edu.cn`

You may also open a GitHub issue for code-related questions.
