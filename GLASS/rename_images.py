import os

# Path to the directory containing the images
directory = '/shared/ssd_30T/NO_WZ/DroneModel/GLASS/datasets/mvtec/thermal_image/test/good'

# Get a sorted list of image files (only PNG, JPG, JPEG)
image_extensions = ('.png', '.jpg', '.jpeg')
image_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(image_extensions)])

# Rename each file
for idx, filename in enumerate(image_files):
    new_name = f"{idx:03}.png"  # Zero-padded to 3 digits
    src = os.path.join(directory, filename)
    dst = os.path.join(directory, new_name)
    os.rename(src, dst)
    print(f"Renamed {filename} -> {new_name}")
