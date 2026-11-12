import numpy as np

# step 1: Create a small grayscale image
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 255, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
], dtype=np.float32)

# Step 2: Create a 3x3 averaging Kernel
kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.float32)/ 9

# Step 3: Implement convolution
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
    
    # Slide kernel across image
    for y in range(height):
        for x in range(width):
            
            region = padded[y:y+kh, x:x+kw]
            
            # Multiply corresponding values and sum them
            output[y, x] = np.sum(region * kernel)
            
    return output

# Step 4: Apply convolution
result = convolution(image, kernel)

# Step 5: Display results
print("ORIGINAL IMAGE:")
print(image)

print("\nKERNEL:")
print(kernel)

print("\nCONVOLVED IMAGE:")
print(result)


