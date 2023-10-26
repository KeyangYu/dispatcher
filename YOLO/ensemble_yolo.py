import cv2
import numpy as np
import torch
from pathlib import Path
from ensemble_boxes import nms
import subprocess


def load_yolo_for_human_detection():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    return model


def train_yolo(model, train_images_path, train_annotations_path):
    # Setting up training arguments
    cfg = "yolov5s.yaml"
    epochs = "50"
    batch_size = "16"

    # Path to the WIDER dataset config
    data = "./yolov5/WIDER/wider.yaml"

    # Train the model
    cmd = [
        "python", "yolov5/train.py",
        "--data", data,
        "--cfg", cfg,
        "--weights", "",  # Start training from scratch
        "--batch-size", batch_size,
        "--epochs", epochs
    ]
    subprocess.check_call(cmd)

    # Load the trained weights and return
    trained_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', path_or_model="runs/train/exp/weights/best.pt")
    return trained_model



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
