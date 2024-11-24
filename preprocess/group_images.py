import os
from PIL import Image

def group_images(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the input directory
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])
    
    # Group images based on the prefix ending with '_0.jpg'
    groups = {}
    for file in image_files:
        # Find the prefix by splitting on "_0.jpg" and taking the first part
        prefix = file.rsplit("_0.jpg", 1)[0]
        groups.setdefault(prefix, []).append(file)
    
    # Process each group
    for prefix, files in groups.items():
        if len(files) < 5:
            print(f"Skipping group {prefix} as it has less than 5 images.")
            continue
        
        # Create a new blank image for the 1x5 layout
        first_image = Image.open(os.path.join(input_dir, files[0]))
        width, height = first_image.size
        combined_image = Image.new('RGB', (width * 5, height))  # 1x5 grid
        
        # Paste images into the combined image
        for i, file in enumerate(files[:5]):  # Only use the first 5 images
            img = Image.open(os.path.join(input_dir, file))
            combined_image.paste(img, (i * width, 0))
        
        # Save the combined image
        output_path = os.path.join(output_dir, f"{prefix}_1x5.jpg")
        combined_image.save(output_path)
        print(f"Saved combined image: {output_path}")

# Example usage
input_directory = r"D:\temporal_CV\1. AIPI 590 - Computer Vision\assignment\3. third_project\Face-Aging-with-Identity-Preserved-Conditional-Generative-Adversarial-Networks\age\0_conv5_lsgan_transfer_g75_0.5f-4_a30\test_ouput_finished"
output_directory = "grouped_generated_images"
group_images(input_directory, output_directory)
