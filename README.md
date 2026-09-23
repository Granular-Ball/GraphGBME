# GraphGBME

GraphGBME is a granular ball enhanced model for imbalanced graph node classification. The training entry point of this repository is `train.py`, and Amazon and T-Finance are currently supported.

## 1. Environment Setup

Python 3.10 or later is recommended. Install the appropriate versions of PyTorch and DGL for your local CUDA environment. Other dependencies include NumPy, SciPy, and scikit-learn.

```bash
python -m pip install torch "dgl>=2.2,<3" numpy scipy scikit-learn
```

## 2. Data Preparation

Download the datasets from their original release pages:

- [Amazon](https://github.com/YingtongDou/CARE-GNN/blob/master/data/Amazon.zip) — download and extract `Amazon.zip` to obtain `Amazon.mat`.
- [T-Finance](https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr?usp=sharing) — download `tfinance.zip` from the shared folder.

Place the dataset files in the following locations:

```text
datasets/
├── Amazon/
│   └── Amazon.mat
└── T-Finance/
    └── tfinance.zip
```

You can also use `--data` to specify another dataset file path.

## 3. Running the Model

Run the following commands from the project root directory.

### Amazon

```bash
python train.py \
  --dataset amazon \
  --quality-threshold 1.00 \
  --min-split-size 7 \
  --min-ball-size 2 \
  --hidden-dim 128 \
  --device cuda:0
```

### T-Finance

```bash
python train.py \
  --dataset tfinance \
  --quality-threshold 0.95 \
  --min-split-size 9 \
  --min-ball-size 2 \
  --hidden-dim 128 \
  --device cuda:0
```

If no GPU is available, change `--device cuda:0` to:

```bash
--device cpu
```



## 4. Five-Seed Mean ± Standard Experiment

`run_five_seed_mean_std.py` can be run directly from the project root after the environment and both datasets have been prepared:

```bash
python run_five_seed_mean_std.py --device cuda:0
```

The script launches ten independent training processes from scratch: five seeds for Amazon and five seeds for T-Finance. It does not load an existing checkpoint. The default seeds are `42,43,44,45,46`, and the reported standard deviation is the sample standard deviation (`ddof=1`).

The default experiment configuration is:

| Parameter | Amazon | T-Finance |
|---|---:|---:|
| Hidden dimension | 128 | 128 |
| Quality threshold | 1.00 | 0.95 |
| Minimum split size | 7 | 9 |
| Minimum prototype ball size | 2 | 2 |
| Fan-outs | 25, 10 | 25, 10 |
| Epochs / patience | 50 / 15 | 50 / 15 |

Use a separate output directory when keeping multiple experiment runs:

```bash
python run_five_seed_mean_std.py \
  --device cuda:0 \
  --output-dir outputs/five-seed-reproduction
```

Important options can also be specified explicitly:

```bash
python run_five_seed_mean_std.py \
  --device cuda:0 \
  --seeds 42,43,44,45,46 \
  --hidden-dim 128 \
  --amazon-quality-threshold 1.00 \
  --amazon-min-split-size 7 \
  --tfinance-quality-threshold 0.95 \
  --tfinance-min-split-size 9 \
  --min-ball-size 2 \
  --output-dir outputs/five-seed-reproduction
```

Exactly five distinct seeds are required. If no CUDA-compatible GPU is available, `--device cpu` can be used, although the full ten-run experiment will be considerably slower.

The combined results are written to:

```text
outputs/five-seed-gb-saign/
├── experiment_config.json
├── all_runs.json
├── mean_std_metrics.json
├── mean_std_metrics.csv
├── amazon/
│   ├── mean_std_metrics.json
│   ├── mean_std_metrics.csv
│   └── seed_<seed>/
└── tfinance/
    ├── mean_std_metrics.json
    ├── mean_std_metrics.csv
    └── seed_<seed>/
```

Each `seed_<seed>/` directory contains that run's `metrics.json`, checkpoint files, granular-ball data, and `train.log`. The combined JSON and CSV files contain the five-seed mean and standard deviation for classification and efficiency metrics.

The script always starts all requested runs from scratch. To avoid replacing files from an earlier invocation, pass a new `--output-dir` for each experiment.

## 5. Output Files

After training, the following files are generated under `outputs/<dataset>/` by default:

```text
best_model.pt       Best model parameters and run configuration
granular_balls.pt   Constructed granular-ball information
metrics.json        Validation/test metrics and training statistics
```

Use `--output-dir` to specify a separate output directory for each experiment. For example:

```bash
python train.py \
  --dataset amazon \
  --quality-threshold 1.00 \
  --min-split-size 7 \
  --min-ball-size 2 \
  --hidden-dim 128 \
  --device cuda:0 \
  --output-dir outputs/amazon-run
```
