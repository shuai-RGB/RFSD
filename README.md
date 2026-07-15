# RFSD

RFSD is a text-attributed graph recommendation model that jointly exploits user/item IDs, profile-derived text representations, homogeneous graphs, and soft cluster-aware interaction propagation. The implementation supports **Amazon-Book**, **Yelp**, and **Steam**, with dataset-specific best hyperparameters loaded automatically from YAML.

## Highlights

- Fuses text-similarity and interaction-based co-occurrence graphs with learnable weights.
- Propagates ID and text features on user-user and item-item homogeneous graphs.
- Learns soft user/item cluster assignments through a projection head and Gumbel-Softmax.
- Performs cluster-aware message passing on the user-item interaction graph.
- Aligns ID and text views with contrastive learning and optimizes an adaptive hard-negative BPR objective.
- Provides one-command dataset presets through `--data amazon|yelp|steam`.

## Framework

```mermaid
flowchart LR
    subgraph INPUT["Text-attributed recommendation data"]
        UP["User profiles"]
        IP["Item profiles"]
        UTE["User text embeddings"]
        ITE["Item text embeddings"]
        UID["User ID embeddings"]
        IID["Item ID embeddings"]
        R["User-item interactions"]
    end

    subgraph GRAPH["Multi-view graph construction"]
        US["User text-similarity graph"]
        IS["Item text-similarity graph"]
        UC["User Jaccard co-occurrence graph"]
        IC["Item Jaccard co-occurrence graph"]
    end

    subgraph HOMO["Homogeneous graph encoding"]
        UH["User graph encoder"]
        IH["Item graph encoder"]
    end

    subgraph CLUSTER["Soft cluster discovery"]
        LF["Learnable cluster features"]
        PH["Projection head"]
        GS["Gumbel-Softmax memberships"]
    end

    subgraph PROP["Cluster-aware interaction propagation"]
        CP["Cluster-conditioned user-item messages"]
        LA["Layer aggregation"]
        GF["Gated ID-text fusion"]
    end

    subgraph LEARN["Prediction and learning"]
        SCORE["User-item scores"]
        BPR["Adaptive hard-negative BPR"]
        CL["ID-text contrastive loss"]
        OBJ["Joint objective"]
    end

    UP --> UTE
    IP --> ITE
    UTE --> US
    ITE --> IS
    R --> UC
    R --> IC
    UTE --> UH
    UID --> UH
    US --> UH
    UC --> UH
    ITE --> IH
    IID --> IH
    IS --> IH
    IC --> IH
    LF --> PH --> GS
    UH --> CP
    IH --> CP
    R --> CP
    GS --> CP
    CP --> LA --> GF --> SCORE
    SCORE --> BPR --> OBJ
    GF --> CL --> OBJ
```

<p align="center"><b>Figure 1. Overview of the RFSD framework.</b></p>

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- SciPy

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Text-attributed recommendation datasets

We evaluate RFSD on three public recommendation datasets: **Amazon-Book**, **Yelp**, and **Steam**. Each user and item is associated with a generated textual profile. Every dataset contains training, validation, and test interactions; the validation split can be used for early stopping.

Download the prepared data from [Google Drive](https://drive.google.com/file/d/1PzePFsBcYofG1MV2FisFLBM2lMytbMdW/view), extract it, and place the dataset directories under `data/`.

Dataset statistics used by the current repository are:

| Dataset     |  Users |  Items | Train interactions | Validation interactions | Test interactions |
| ----------- | -----: | -----: | -----------------: | ----------------------: | ----------------: |
| Amazon-Book | 11,000 |  9,332 |            120,464 |                  40,290 |            40,106 |
| Yelp        | 11,091 | 11,010 |            166,620 |                  55,479 |            55,436 |
| Steam       | 23,310 |  5,237 |            316,190 |                 104,897 |           104,835 |

The downloaded/raw data and generated embeddings follow this layout:

```text
data/
├── amazon/
│   ├── trn_mat.pkl       # training interactions (SciPy sparse matrix)
│   ├── val_mat.pkl       # validation interactions (SciPy sparse matrix)
│   ├── tst_mat.pkl       # test interactions (SciPy sparse matrix)
│   ├── usr_prf.pkl       # user text profiles (raw/preprocessing input)
│   ├── itm_prf.pkl       # item text profiles (raw/preprocessing input)
│   ├── usr_emb_np.pkl    # user text embeddings used by RFSD
│   └── itm_emb_np.pkl    # item text embeddings used by RFSD
├── yelp/
│   └── ...
└── steam/
    └── ...
```

The current training pipeline reads `usr_emb_np.pkl` and `itm_emb_np.pkl` directly. A forthcoming `read_profile.py` will read `usr_prf.pkl` and `itm_prf.pkl` and prepare the text embeddings. Until that script is added, make sure the two embedding files are present in each dataset directory.

## Training

Run RFSD from the repository root:

```bash
# Amazon-Book
python main.py --data amazon

# Yelp
python main.py --data yelp

# Steam
python main.py --data steam
```

Amazon-Book is the default dataset, so the following is equivalent to `--data amazon`:

```bash
python main.py
```

The entry point can also be executed from inside `rfsd/`, or as a Python module:

```bash
python -m rfsd --data yelp
```

To select a GPU or override a preset value:

```bash
python main.py --data yelp --device cuda:0
python main.py --data steam --lr 1e-3 --early-stop
```

Explicit command-line arguments take precedence over the values loaded from YAML.

## Best hyperparameters

The best parameters for all datasets are stored in [`rfsd/best_params.yaml`](rfsd/best_params.yaml). Selecting `--data` automatically loads the corresponding section. New datasets can be added by inserting another top-level section containing at least `data_dir`.

```yaml
yelp:
  data_dir: data/yelp
  learning_rate: 1e-4
  num_negatives: 20
  user_top_k: 40
  item_top_k: 60
```

View every available command-line override with:

```bash
python main.py --help
```

## Evaluation

RFSD reports the following top-*k* ranking metrics after each epoch:

- NDCG@*k*
- Recall@*k*
- Hit Rate@*k*
- MAP@*k*

Training interactions and validation interactions are masked before test ranking.

## Project structure

```text
.
├── main.py                    # compatibility command-line entry
├── requirements.txt
├── rfsd/
│   ├── main.py                # configuration, YAML loading, and CLI
│   ├── best_params.yaml       # dataset-specific best hyperparameters
│   ├── data.py                # data loading and graph construction
│   ├── model.py               # RFSD model and neural components
│   ├── evaluation.py          # objectives and ranking metrics
│   ├── trainer.py             # training, validation, and early stopping
│   └── utils.py               # sparse-matrix and graph utilities
└── data/                      # local datasets (not committed)
```

The root-level `model.py`, `train.py`, `dataset.py`, `evaluation.py`, and `utilis.py` are compatibility modules for earlier imports. New code should import from `rfsd` directly.

## Reproducibility

- The default random seed is `42`; override it with `--seed`.
- Dataset-specific settings are versioned in `rfsd/best_params.yaml`.
- Use the same interaction splits and text embeddings when comparing results.
- Enable validation-based early stopping with `--early-stop` when required.

## Citation

Citation information will be added when the RFSD paper is publicly available.
