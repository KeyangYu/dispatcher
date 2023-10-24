import xmlrpc.client
import time
import threading
from xmlrpc.server import SimpleXMLRPCServer
import os
import numpy as np
from ensemble_yolo import load_yolo_for_human_detection, train_yolo, yolo_predict

CLIENT_PORT = 9091

proxy = xmlrpc.client.ServerProxy("http://localhost:9090/RPC2")


def wait_for_start_signal():
    print("Waiting for coordinator to start training...")
    while True:
        try:
            if proxy.should_start_training():
                return True
            time.sleep(5)
        except Exception as e:
            print(f"Error while waiting for start signal: {e}")
            time.sleep(10)


client_server = SimpleXMLRPCServer(('localhost', CLIENT_PORT))
client_server.register_introspection_functions()


def run_client_server():
    client_server.serve_forever()


if __name__ == "__main__":
    thread = threading.Thread(target=run_client_server)
    thread.start()

    while True:
        if wait_for_start_signal():
            print("Start signal received. Starting training...")
            try:
                # Load human data
                print("Fetching data for training...")
                train_images, train_annotations = proxy.get_human_data(f"client1")

                # Load YOLO model for human detection
                print("Loading YOLO model...")
                model = load_yolo_for_human_detection()

                # Train YOLO on human data
                print("Training YOLO model...")
                train_yolo(model, train_images, train_annotations)

                # Save the trained model
                print("Saving trained model...")
                model_name = f"client1_model.h5"
                model.save(model_name)

                # Send the model to the coordinator
                print("Sending model to coordinator...")
                with open(model_name, "rb") as model_file:
                    proxy.upload_model(xmlrpc.client.Binary(model_file.read()), f"client1")

                print("Model training completed and sent to the coordinator.")
                break

            except Exception as e:
                print(f"Error during training or sending the model: {e}")
                time.sleep(10)
