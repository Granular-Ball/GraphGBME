"""GraphGBME loss exponents selected by validation PR-AUC with random sampling."""

SEEDS = (42, 43, 44, 45, 46)
CLASS_WEIGHT_GAMMAS = {"tfinance": 0.50, "amazon": 0.50, "elliptic": 0.0, "yelpchi": 0.15}

# Split quality, minimum split size, and class-weight exponent.
EXPERIMENT_PRESETS = {
    "tfinance": (0.80, 2, 0.25),
    "amazon": (0.90, 2, 0.75),
    "yelpchi": (0.85, 6, 0.15),
    "elliptic": (0.95, 10, 0.25),
}
