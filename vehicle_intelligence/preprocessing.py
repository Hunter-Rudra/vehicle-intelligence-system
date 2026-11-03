import numpy as np

def to_grayscale(frame):
    frame = np.asarray(frame)
    if frame.ndim == 2:
        return frame.astype(np.float32, copy=False)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("Expected HxW grayscale or HxWx3 RGB frame")
    a = frame[..., :3].astype(np.float32)
    return 0.299*a[..., 0] + 0.587*a[..., 1] + 0.114*a[..., 2]

def gaussian_kernel(size=5, sigma=1.0):
    if size < 3 or size % 2 == 0 or sigma <= 0:
        raise ValueError("size must be odd and >=3; sigma must be positive")
    axis = np.arange(size, dtype=np.float32) - size//2
    xx, yy = np.meshgrid(axis, axis)
    k = np.exp(-(xx*xx + yy*yy)/(2*sigma*sigma))
    return k / k.sum()

def convolve2d(image, kernel):
    image = np.asarray(image, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    if image.ndim != 2 or kernel.ndim != 2 or any(s % 2 == 0 for s in kernel.shape):
        raise ValueError("Expected 2D image and odd-sized 2D kernel")
    ph, pw = kernel.shape[0]//2, kernel.shape[1]//2
    padded = np.pad(image, ((ph,ph),(pw,pw)), mode="edge")
    out = np.zeros_like(image)
    k = kernel[::-1, ::-1]
    for y in range(k.shape[0]):
        for x in range(k.shape[1]):
            out += k[y,x] * padded[y:y+image.shape[0], x:x+image.shape[1]]
    return out

def gaussian_blur(image, size=5, sigma=1.0):
    return convolve2d(image, gaussian_kernel(size, sigma))
