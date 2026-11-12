import numpy as np

image = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 255, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
], dtype=np.float32)


def dynamic_kernel(size, sigma=1.0):
    
    coordinates = np.arange((-(size // 2)), (size // 2 + 1))
    
    x, y = np.meshgrid(coordinates, coordinates)

    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    
    return kernel
    
    
def convolution(image, kernel):
    height, width = image.shape
    kh, kw = kernel.shape
    
    pad_h = kh // 2
    pad_w = kw // 2
    
    padded = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant"
    )
    
    output = np.zeros((height, width), dtype=np.float32)
    
    for x in range(height):
        for y in range(width):
            region = padded[x : x+kh, y : y+kw]
            output[x, y] = np.sum(region * kernel)
            
    return output


kernel = dynamic_kernel(3)

print(convolution(image, kernel))