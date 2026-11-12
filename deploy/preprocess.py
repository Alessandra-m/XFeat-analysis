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


def load_raw(path, h, w):
    """Raw BGR frame as the gray model (xfeat_grayscale_*.onnx) expects it: float32 (1, 3, h, w), 0..255.
    Grayscale + normalization happen inside that model's graph."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None])


def model_input(path, h, w, channels):
    """Input for a model with `channels` input channels: 1 -> normalized gray, 3 -> raw BGR."""
    return load_raw(path, h, w) if channels == 3 else preprocess(path, h, w)[1]


def s2d(x, f):
    """Space-to-depth == torch.pixel_unshuffle: (1, C, H, W) -> (1, C*f*f, H/f, W/f)."""
    if f == 1:
        return x
    n, c, h, w = x.shape
    y = x.reshape(n, c, h // f, f, w // f, f).transpose(0, 1, 3, 5, 2, 4)
    return np.ascontiguousarray(y.reshape(n, c * f * f, h // f, w // f))


def s2d_factor(channels):
    """Input channels of a gray model input -> its space-to-depth factor (1 -> 1, 4 -> 2, 16 -> 4, 64 -> 8)."""
    f = int(round(channels ** 0.5))
    assert f * f == channels, f"{channels} channels is not a space-to-depth of a 1-channel image"
    return f


def model_inputs(path, shapes):
    """All inputs of a model from one image. shapes: list of NCHW input shapes, e.g. from onnxruntime.
    3 channels -> raw BGR; c = f*f channels -> normalized gray at (h*f, w*f), space-to-depth by f."""
    out = []
    for _, c, h, w in shapes:
        if c == 3:
            out.append(load_raw(path, h, w))
        else:
            f = s2d_factor(c)
            out.append(s2d(preprocess(path, h * f, w * f)[1], f))
    return out


def preprocess(path, h, w):
    """Returns (raw, normalized), both float32 of shape (1, 1, h, w) -- NCHW."""
    img = load_gray(path, h, w)
    return img[None, None], normalize(img)[None, None]
