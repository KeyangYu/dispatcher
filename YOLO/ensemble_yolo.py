import cv2
import numpy as np
import torch
from pathlib import Path
from ensemble_boxes import nms



def load_yolo_for_human_detection():
    """
    Load a pre-defined YOLO model or initialize a new one for human detection.
    Returns:
        model: The YOLO model initialized/loaded.
    """
    # You can either load a pretrained model or initialize a new one here
    model = torch.hub.load('ultralytics/yolov5',
                           'yolov5s')  # 'yolov5s' is the small version. There's also 'yolov5m', 'yolov5l', and 'yolov5x'
    return model


def train_yolo(model, train_images, train_annotations):
    # In YOLOv5, the dataset structure is usually set up in a specific way,
    # but for simplicity, let's assume you've set it up correctly or are using
    # the standard COCO dataset directory structure.

    # Setting up training arguments
    cfg = "yolov5s.yaml"  # This is the configuration file for the small YOLOv5 model. You can use 'yolov5m', 'yolov5l', or 'yolov5x' for bigger models.
    epochs = 50  # Adjust as needed
    batch_size = 16  # Adjust based on your GPU memory

    # Path to the COCO dataset
    data = "coco.yaml"

    # Train the model
    model = model.train()  # Set the model to training mode
    results = model.fit(train_images, train_annotations, cfg=cfg, data=data, epochs=epochs, batch_size=batch_size)

    return model


def yolo_predict(model, image):
    # The YOLOv5 repository provides a `detect` function that can be used.
    results = model(image)
    detections = results.pred[0]  # Bounding boxes

    return detections



def ensemble_detections(detections_from_all_models):
    boxes_list = []
    scores_list = []
    labels_list = []

    # Extract boxes, scores, and labels from detections
    for detections in detections_from_all_models:
        boxes = detections[..., :4]
        scores = detections[..., 4:5]
        labels = torch.ones((scores.shape[0], ), dtype=torch.int)

        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    # Apply NMS
    boxes, scores, labels = nms(boxes_list, scores_list, labels_list, iou_thr=0.5)
    final_detections = list(zip(boxes, scores, labels))

    return final_detections
