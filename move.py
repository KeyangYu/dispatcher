import os

# Paths to directories
train_images_path = "/Users/yky/PycharmProjects/dispatcher/YOLO/yolov5/WIDER/Images/train"
valid_images_path = "/Users/yky/PycharmProjects/dispatcher/YOLO/yolov5/WIDER/Images/valid"

# Function to rename the label files
def rename_labels(img_directory):
    for label_file in os.listdir(img_directory):
        if label_file.endswith('.jpg.txt'):
            # Get the new name for the label file
            new_name = label_file.replace('.jpg.txt', '.txt')
            old_path = os.path.join(img_directory, label_file)
            new_path = os.path.join(img_directory, new_name)
            os.rename(old_path, new_path)

# Rename the label files
rename_labels(train_images_path)
rename_labels(valid_images_path)

print("Label files renamed successfully!")
