import cv2
import matplotlib.pyplot as plt
import os

img_path = r"D:\FYP\Processed_MRI\0001\T2_TSE_TRA__0001_001.png"

if not os.path.exists(img_path):
    print(f"File does not exist: {img_path}")
else:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to read the image. Check the file integrity.")
    else:
        plt.imshow(img, cmap='gray')
        plt.title("Axial MRI Slice")
        plt.axis('off')
        plt.show()
