import numpy as np

# 1. Create a simpe grayscale image
image = np.array([
    [0, 0, 0, 0],
    [0, 255, 255, 0],
    [0, 255, 255, 0],
    [0, 0, 0, 0]
], dtype=np.uint8)

# 2. Display image information 
print("""#-------------------------------------------------------------------------
""", " IMAGE INFORMATION", """\n#-------------------------------------------------------------------------
""")

print("Image:\n", image)
print("Shape:", image.shape)
print("Dimensions:", image.ndim)
print("Data type:", image.dtype)
print("Minimum pixel:", image.min())
print("Maximum pixel", image.max())

# 3. Access an individual pixel
print("""\n#-------------------------------------------------------------------------
""", "PIXEL ACCESS", """\n#-------------------------------------------------------------------------
""")
print("Pixel at row 1, column 1:", image[1, 1])

# 4. Extract a region of the image
region = image[1:3, 1:3]

print("""\n#-------------------------------------------------------------------------
""", "IMAGE REGION", """\n#-------------------------------------------------------------------------
""")

# 5. Convert image to floating-point values
float_image = image.astype(np.float32)

# Normalize pixel values between 0 and 1
normalized_image = float_image / 255

print("""\n#-------------------------------------------------------------------------
""", "NORMALIZED_IMAGE", """#-------------------------------------------------------------------------
""")
print(normalized_image)

print("Normalized minimum:", normalized_image.min())
print("Normalized maximum:", normalized_image.max())

# 6. Create a blank RGB image
rgb_image = np.zeros((4, 4, 3), dtype=np.uint8)
print("rgb_image:", rgb_image)

print("""\n#-------------------------------------------------------------------------
""", "RGB IMAGE", """\n#-------------------------------------------------------------------------
""")
print("RGB shape:", rgb_image.shape)
print("RGB dimensions:", rgb_image.ndim)

print("\nPROGRAM COMPLETED")