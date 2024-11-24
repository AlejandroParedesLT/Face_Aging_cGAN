from PIL import Image
import os

def resize_image(input_path, output_path, target_width=400, target_height=400):
    """
    Resize an image to the target dimensions.
    
    Parameters:
    - input_path (str): Path to the original image.
    - output_path (str): Path to save the resized image.
    - target_width (int): Desired width of the resized image.
    - target_height (int): Desired height of the resized image.
    """
    with Image.open(input_path) as img:
        # Resize with upscale to target dimensions
        resized_img = img.resize((target_width, target_height), Image.LANCZOS)
        resized_img.save(output_path)

# Define directories
input_dir = "D:/temporal_CV/1. AIPI 590 - Computer Vision/assignment/3. third_project/wiki_preprocess/UTKFace/unlabeled"
resized_dir = "D:/temporal_CV/1. AIPI 590 - Computer Vision/assignment/3. third_project/wiki_preprocess/UTKFace/unlabeled"
os.makedirs(resized_dir, exist_ok=True)

# Resize all images in the UTKFace dataset
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(resized_dir, filename)
        print("Resizing:", input_path)
        # Resize image to 400x400 pixels
        resize_image(input_path, output_path, target_width=400, target_height=400)
        print("Resized:", output_path)


print("Resizing complete. Resized images saved to:", resized_dir)

# Update script to use the resized directory for processing
utk_dir = resized_dir
