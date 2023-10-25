import cv2
import numpy as np
import torch
from pathlib import Path
from ensemble_boxes import nms


def load_yolo_for_human_detection():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    return model


def train_yolo(model, train_images_path, train_annotations_path):
    # Since we're using YOLOv5's training mechanism, the input paths won't be used directly.
    # They're kept for reference.

    # Setting up training arguments
    cfg = "yolov5s.yaml"
    epochs = 50
    batch_size = 16

    # Path to the WIDER dataset config
    data = "wider.yaml"

    # Train the model
    model = model.train()
    results = model.fit(data=data, cfg=cfg, epochs=epochs, batch_size=batch_size)

    return model


def yolo_predict(model, image):
    results = model(image)
    detections = results.pred[0]
    return detections


def ensemble_detections(detections_from_all_models):
    boxes_list = []
    scores_list = []
    labels_list = []

    for detections in detections_from_all_models:
        boxes = detections[..., :4]
        scores = detections[..., 4:5]
        labels = torch.ones((scores.shape[0],), dtype=torch.int)

        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    boxes, scores, labels = nms(boxes_list, scores_list, labels_list, iou_thr=0.5)
    final_detections = list(zip(boxes, scores, labels))

    return final_detections
