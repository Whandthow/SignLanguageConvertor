import numpy as np

def normalize(flat63):
    pts = np.asarray(flat63, dtype=np.float32).reshape(21, 3)
    pts = pts - pts[0]                         # зап'ястя = початок координат
    scale = np.linalg.norm(pts[9, :2])         # відстань зап'ястя → основа середнього пальця
    if scale < 1e-6:
        scale = 1.0
    return (pts / scale).flatten()