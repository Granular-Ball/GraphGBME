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
  --quality-threshold 0.85 \
  --min-split-size 7 \
  --min-ball-size 1 \
  --device cuda:0
```

### T-Finance

```bash
python train.py \
  --dataset tfinance \
  --quality-threshold 0.95 \
  --min-split-size 4 \
  --min-ball-size 1 \
  --device cuda:0
```

If no GPU is available, change `--device cuda:0` to:

```bash
--device cpu
```



## 4. Output Files

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
  --quality-threshold 0.85 \
  --min-split-size 7 \
  --min-ball-size 1 \
  --device cuda:0 \
  --output-dir outputs/amazon-run
```
