from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from threading import Thread
import joblib
import os
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
from flask import Flask
import pandas as pd
from threading import Thread
import numpy as np
from ensemble_yolo import ensemble_detections
from ensemble_yolo import load_yolo_for_human_detection, train_yolo, yolo_predict
import cv2
import glob

# Base directory
WIDER_DIR = "WIDER/"

# Load paths to images
train_images = glob.glob(os.path.join(WIDER_DIR, "Images", "train", "*.jpg"))
valid_images = glob.glob(os.path.join(WIDER_DIR, "Images", "valid", "*.jpg"))

# For the sake of your existing code, I'll assume we're working with the 'train' set.
# You can change this as per your needs.
wider_images = train_images

# Load corresponding annotations
wider_annotations = [os.path.join(WIDER_DIR, "Annotations", os.path.basename(img_path) + ".txt") for img_path in wider_images]

class CustomXMLRPCRequestHandler(SimpleXMLRPCRequestHandler):
    def do_POST(self):
        self.server.client_address = self.client_address
        super().do_POST()


app = Flask(__name__)
nodes = {}
server = SimpleXMLRPCServer(('0.0.0.0', 9090), requestHandler=CustomXMLRPCRequestHandler)
server.register_introspection_functions()

training_started = False


def load_client_ranks_from_csv(file_name):
    df = pd.read_csv(file_name)
    ranks = dict(zip(df['client_id'], df['rank']))
    return ranks


client_ranks = load_client_ranks_from_csv('rank.csv')

def read_wider_annotation(anno_path):
    with open(anno_path, 'r') as f:
        lines = f.readlines()
        boxes = []
        for line in lines[1:]:
            box = [int(coord) for coord in line.strip().split()]
            # Assuming format: [x, y, width, height]
            x, y, w, h = box[:4]
            # Convert to format: [x_min, y_min, x_max, y_max]
            boxes.append([x, y, x+w, y+h])
    return boxes


def get_human_data(client_id):
    node_id = server.client_address[0]
    nodes[node_id] = {'status': 'working'}

    proportion = client_ranks.get(client_id, 0.2)

    total_data_size = len(wider_images)

    start_index = int((client_ranks[client_id] - 0.2) * total_data_size)
    end_index = start_index + int(proportion * total_data_size)

    # Get a subset of images
    selected_images = wider_images[start_index:end_index]

    # Get corresponding annotations
    selected_annotations = [read_wider_annotation(anno_path) for anno_path in wider_annotations[start_index:end_index]]

    return selected_images, selected_annotations

# Register the function with the server
server.register_function(get_human_data, 'get_human_data')


def mark_training_complete(node_id):
    nodes[node_id]['status'] = 'finished'
    return True


server.register_function(mark_training_complete, 'mark_training_complete')


def preprocess_for_yolo(frame, size=416):
    """
    Preprocess the image frame for YOLO model input.

    :param frame: Input frame.
    :param size: Size to which the frame should be resized.
    :return: Preprocessed frame.
    """
    # Resize the frame
    resized_frame = cv2.resize(frame, (size, size))

    # Normalize the pixel values
    normalized_frame = resized_frame.astype('float32') / 255.0

    # Add batch dimension
    input_frame = np.expand_dims(normalized_frame, axis=0)

    return input_frame


def draw_bboxes(frame, detections):
    """
    Draw bounding boxes on a given frame.

    :param frame: The frame on which to draw.
    :param detections: The detections for this frame.
    :return: The frame with bounding boxes drawn.
    """
    for detection in detections:
        # Assuming each detection is a tuple (x_min, y_min, x_max, y_max)
        cv2.rectangle(frame, (detection[0], detection[1]), (detection[2], detection[3]), (0, 255, 0), 2)
    return frame


def yolo_predict_and_draw(model, video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    all_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for YOLO model input
        processed_frame = preprocess_for_yolo(frame)

        # Get detections for the current frame
        detections = model.predict(processed_frame)  # Modify based on your YOLO model's API
        all_detections.append(detections)

        # Draw bounding boxes on the frame
        frame_with_bboxes = draw_bboxes(frame, detections)

        # Write the frame with bounding boxes to the output video
        out.write(frame_with_bboxes)

    cap.release()
    out.release()
    return all_detections


models_received = 0

def upload_model(model_binary, client_id):
    global models_received
    filename = f"{client_id}_model.h5"
    with open(filename, 'wb') as f:
        f.write(model_binary.data)
    print(f"Model from {client_id} dumped in dispatcher as {filename}.")

    models_received += 1
    if models_received == 3:
        ensemble_and_evaluate_yolo()

    return True

server.register_function(upload_model, 'upload_model')

def ensemble_and_evaluate_yolo():
    model_files = [f"client{i}_model.h5" for i in range(1, 4)]
    test_video = "test_1.mp4"

    detections_from_all_models = []

    for i, model_file in enumerate(model_files):
        model = load_yolo_for_human_detection(model_file)  # Placeholder to load your YOLO model
        output_video_path = f"output_{i}.mp4"  # Name of the video with bounding boxes for this model
        detections = yolo_predict_and_draw(model, test_video, output_video_path)
        detections_from_all_models.append(detections)

    final_detections = ensemble_detections(detections_from_all_models)
    # Evaluate the ensemble against the ground truth (like mAP, IoU, etc.)


def should_start_training():
    return training_started


server.register_function(should_start_training, 'should_start_training')


@app.route('/')
def index():
    return "Coordinator is running"


def run_rpc_server():
    global training_started
    print("Dispatcher RPC Server is running...")

    while True:
        start = input("Enter 'start' to begin training on all clients: ")
        if start.strip().lower() == 'start':
            training_started = True
            break

    server.serve_forever()


if __name__ == '__main__':
    rpc_thread = Thread(target=run_rpc_server)
    rpc_thread.start()
    app.run(host='0.0.0.0', port=5050)
