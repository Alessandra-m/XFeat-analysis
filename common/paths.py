import os
import sys

# repo root: this file lives in <root>/common/
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XFEAT_DIR = os.path.join(HERE, "third_party", "accelerated_features")
RESULTS_DIR = os.path.join(HERE, "results")
PRUNE_DIR = os.path.join(HERE, "prune", "results")

sys.path.insert(0, XFEAT_DIR)
