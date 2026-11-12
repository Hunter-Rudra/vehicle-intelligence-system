import numpy as np

image = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 255, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=np.float32)

kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32) / 16

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
    
    for y in range(height):
        for x in range(width):
            region = padded[y : y+kh, x : x+kw]
            output[y, x] = np.sum(region * kernel)
            
    return output

result = convolution(image, kernel)

print("Original image:")
print(image)

print("\nGaussian kernel:")
print(kernel)

print("\nGaussian blurred image:")
print(result)
    