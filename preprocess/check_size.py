from PIL import Image
import os

# Path to a sample image in the preprocessed-wiki directory
image_path = r"D:\temporal_CV\1. AIPI 590 - Computer Vision\assignment\3. third_project\wiki_preprocess\wiki\00\67000_1947-05-13_2007.jpg"

# Open the image and print its size
with Image.open(image_path) as img:
    print(f"Image size: {img.size}")  # Outputs (width, height)


# Path to a sample image in the preprocessed-wiki directory
image_path = r"D:\temporal_CV\1. AIPI 590 - Computer Vision\assignment\3. third_project\wiki_preprocess\UTKFace\unlabeled\1_0_0_20161219140623097.jpg.chip.jpg"

# Open the image and print its size
with Image.open(image_path) as img:
    print(f"Image size: {img.size}")  # Outputs (width, height)
