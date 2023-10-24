import os
import shutil

# Directory paths
wider_dir = "/Users/yky/Downloads/WIDER"
images_dir = os.path.join(wider_dir, "Images")
train_txt = os.path.join(wider_dir, "train.txt")
val_txt = os.path.join(wider_dir, "val.txt")

# Create directories
os.makedirs(os.path.join(wider_dir, "images/train"), exist_ok=True)
os.makedirs(os.path.join(wider_dir, "images/valid"), exist_ok=True)

# Copy train images
with open(train_txt, "r") as file:
    for line in file:
        src_img_path = os.path.join(images_dir, line.strip() + ".jpg")
        dest_img_path = os.path.join(wider_dir, "images/train", os.path.basename(line.strip()) + ".jpg")
        shutil.copy(src_img_path, dest_img_path)

# Copy validation images
with open(val_txt, "r") as file:
    for line in file:
        src_img_path = os.path.join(images_dir, line.strip() + ".jpg")
        dest_img_path = os.path.join(wider_dir, "images/valid", os.path.basename(line.strip()) + ".jpg")
        shutil.copy(src_img_path, dest_img_path)
