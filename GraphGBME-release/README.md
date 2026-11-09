# GraphGBME


## Installation

Use a Python 3.11 environment. The reference setup uses PyTorch 2.2.1, DGL 2.2.1, and torchdata 0.7.1. Select matching PyTorch and DGL builds for your operating system and CPU or CUDA runtime. See the [PyTorch installation commands](https://pytorch.org/get-started/previous-versions/) and the [DGL 2.2.1 release notes](https://github.com/dmlc/dgl/releases/tag/v2.2.1).

CPU example:

```bash
python -m pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -f https://data.dgl.ai/wheels/torch-2.2/repo.html
```

CUDA 12.1 example for Linux x86-64:

```bash
python -m pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install dgl==2.2.1+cu121 -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
python -m pip install -r requirements.txt
```

For Windows x86-64, use `https://data.dgl.ai/wheels/cu121/repo.html` as the DGL wheel index in the CUDA example.

## Datasets

Download datasets from their providers and place them under the paths below. To use a different location, pass `--data PATH`.

| `--dataset` | Default path | Source |
| --- | --- | --- |
| `amazon` | `datasets/Amazon/Amazon.mat` | [CARE-GNN data](https://github.com/YingtongDou/CARE-GNN/tree/master/data), extract `Amazon.zip` |
| `tfinance` | `datasets/T-Finance/tfinance.zip` | [T-Finance release](https://github.com/squareRoot3/Rethinking-Anomaly-Detection#how-to-run) |
| `yelpchi` | `datasets/YelpChi/FraudYelp.zip` | [DGL YelpChi archive](https://data.dgl.ai/dataset/FraudYelp.zip) |
| `elliptic` | `datasets/Elliptic/elliptic-data-set.zip` | [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) |

## Training

Run commands from the repository root. Select `tfinance`, `amazon`, `yelpchi`, or `elliptic` with `--dataset`. The training script runs seeds 42, 43, 44, 45, and 46 using all labels in each training split.

```bash
python train.py --dataset yelpchi --device auto
```

## Five-seed robustness experiment

`run_robustness.py` runs the same five seeds at each of the four minority-supervision ratios of 1%, 2%, 3%, and 4%. One invocation trains 20 models on the selected dataset. Each ratio specifies the fraction of labeled training nodes in the minority class. All majority training labels are retained, and minority nodes whose labels are withheld remain available for message passing. The feature scaler and granular balls are fitted using the retained minority training labels.

```bash
python run_robustness.py --dataset yelpchi --device auto
```

| Dataset | Split quality threshold | Minimum size to attempt a split | Class-weight exponent |
| --- | ---: | ---: | ---: |
| Amazon | 0.90 | 2 | 0.75 |
| T-Finance | 0.80 | 2 | 0.25 |
| YelpChi | 0.85 | 6 | 0.15 |
| Elliptic | 0.95 | 10 | 0.25 |

These presets use direct concatenation, shuffled training seeds, hidden size 128, fan-outs 25 and 10, batch size 1024, and up to 50 epochs with patience 15. The script computes means and sample standard deviations directly from the generated per-seed metrics. It writes `experiment_config.json` and `mean_std_metrics.json` under `outputs/robustness/<dataset>/`. Each `ratio_<ratio>/seed_<seed>/` directory contains `metrics.json`, `train.log`, `best_model.pt`, and `granular_balls.pt`. Use a fresh `--output-dir` for each experiment.

## Five-seed ablation experiment

```bash
python run_ablation.py --dataset yelpchi --device auto
```

The ablation script uses all labels in each training split and the dataset presets listed above. It trains each variant from scratch for seeds 42–46, using the same split and shuffled training-node order for each seed across variants.

| Variant | Configuration |
| --- | --- |
| `full` | Graph-constrained granular balls, prototype concatenation, and class-weighted cross-entropy |
| `without_prototype_enhancement` | Feed only the local GraphSAGE representation into the second layer |
| `without_graph_constraint` | Assign granular-ball members by Euclidean feature distance while retaining message passing on the input graph |
| `without_class_weighting` | Set the class-weight exponent to zero and use ordinary cross-entropy |

The script writes `experiment_config.json` and `mean_std_metrics.json` under `outputs/ablation/<dataset>/`. Each `<variant>/seed_<seed>/` directory contains the corresponding metrics, training log, checkpoint, and ball partition.

