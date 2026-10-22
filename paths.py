import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
XFEAT_DIR = os.path.join(HERE, "third_party", "accelerated_features")
RESULTS_DIR = os.path.join(HERE, "results")

# XFeat code lives in the submodule, make `modules` importable
sys.path.insert(0, XFEAT_DIR)
