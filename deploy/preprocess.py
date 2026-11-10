"""Input preprocessing shared by export (xfeat env) and RKNN conversion (rknn env): only cv2 + numpy."""
import cv2
import numpy as np


def load_gray(path, h, w):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def normalize(img):
    """== mean over channels + InstanceNorm2d(1), eps=1e-5 (the part removed from the ONNX graph)."""
    return ((img - img.mean()) / np.sqrt(img.var() + 1e-5)).astype(np.float32)


def preprocess(path, h, w):
    """Returns (raw, normalized), both float32 of shape (1, 1, h, w) -- NCHW."""
    img = load_gray(path, h, w)
    return img[None, None], normalize(img)[None, None]
