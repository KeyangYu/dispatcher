import os


def normalize_and_reformat_labels(img_directory, img_width, img_height):
    for label_file_name in os.listdir(img_directory):
        if label_file_name.endswith('.txt'):
            file_path = os.path.join(img_directory, label_file_name)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Remove the first line
            lines = lines[1:]

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                class_id = parts[0]
                x_center = (float(parts[1]) + float(parts[3]) / 2) / img_width
                y_center = (float(parts[2]) + float(parts[4]) / 2) / img_height
                width = float(parts[3]) / img_width
                height = float(parts[4]) / img_height
                new_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")

            # Overwrite the file with the reformatted labels
            with open(file_path, 'w') as f:
                f.write("\n".join(new_lines))


# Assuming the images have a consistent size (for example, 640x640).
# Adjust the values accordingly if this is not the case.
img_width = 640
img_height = 640

train_images_path = "/Users/yky/PycharmProjects/dispatcher/YOLO/yolov5/WIDER/Images/train"
valid_images_path = "/Users/yky/PycharmProjects/dispatcher/YOLO/yolov5/WIDER/Images/valid"

normalize_and_reformat_labels(train_images_path, img_width, img_height)
normalize_and_reformat_labels(valid_images_path, img_width, img_height)

print("Label files reformatted successfully!")
