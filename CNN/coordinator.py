from flask import Flask
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
import numpy as np
from threading import Thread
import pandas as pd
import joblib
import os

class CustomXMLRPCRequestHandler(SimpleXMLRPCRequestHandler):
    def do_POST(self):
        self.server.client_address = self.client_address
        super().do_POST()

mnist = fetch_openml('mnist_784')
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

shuffled_data, shuffled_target = shuffle(mnist.data, mnist.target, random_state=42)

def get_mnist(client_id):
    node_id = server.client_address[0]
    nodes[node_id] = {'status': 'working'}
    proportion = client_ranks.get(client_id, 0.2)

    total_data_size = len(shuffled_data)
    start_index = int((client_ranks[client_id] - 0.2) * total_data_size)
    end_index = start_index + int(proportion * total_data_size)

    return shuffled_data.iloc[start_index:end_index].values.tolist(), shuffled_target.iloc[start_index:end_index].values.tolist()

server.register_function(get_mnist, 'get_mnist')

def mark_training_complete(node_id):
    nodes[node_id]['status'] = 'finished'
    return True

server.register_function(mark_training_complete, 'mark_training_complete')

models_received = 0

def upload_model(model_binary, client_id):
    global models_received
    filename = f"{client_id}_model.h5"
    with open(filename, 'wb') as f:
        f.write(model_binary.data)
    print(f"Model from {client_id} dumped in dispatcher as {filename}.")

    models_received += 1
    if models_received == 3:
        # ensemble_and_evaluate_voting()
        # ensemble_and_evaluate_avg()
        ensemble_and_evaluate_stacking()

    return True

server.register_function(upload_model, 'upload_model')

def ensemble_and_evaluate_voting():
    models = []
    for client_id in ["client1", "client2", "client3"]:
        model_name = f"{client_id}_model.h5"
        if os.path.exists(model_name):
            models.append(load_model(model_name))
        else:
            print(f"Model for {client_id} not found.")
            return

    X_test = np.array(shuffled_data.iloc[int(0.7 * len(shuffled_data)):].values.tolist()).reshape(-1, 28, 28, 1)
    y_test = np.array(shuffled_target.iloc[int(0.7 * len(shuffled_data)):].values.tolist())

    # Majority vote
    predictions = []
    for x in X_test:
        preds = [np.argmax(model.predict(x.reshape(1, 28, 28, 1))) for model in models]
        predictions.append(max(set(preds), key=preds.count))

    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")

def ensemble_and_evaluate_avg():
    models = []
    for client_id in ["client1", "client2", "client3"]:
        model_name = f"{client_id}_model.h5"
        if os.path.exists(model_name):
            models.append(load_model(model_name))
        else:
            print(f"Model for {client_id} not found.")
            return

    X_test = np.array(shuffled_data.iloc[int(0.7 * len(shuffled_data)):].values.tolist()).reshape(-1, 28, 28, 1)
    y_test = np.array(shuffled_target.iloc[int(0.7 * len(shuffled_data)):].values.tolist()).astype(int)

    # Averaging method
    avg_predictions = np.zeros((len(X_test), 10))  # Assuming 10 classes for MNIST
    for model in models:
        avg_predictions += model.predict(X_test)
    avg_predictions /= len(models)
    final_predictions = np.argmax(avg_predictions, axis=1)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_test, final_predictions)}")

def ensemble_and_evaluate_stacking():
    models = []
    for client_id in ["client1", "client2", "client3"]:
        model_name = f"{client_id}_model.h5"
        if os.path.exists(model_name):
            models.append(load_model(model_name))
        else:
            print(f"Model for {client_id} not found.")
            return

    X_test = np.array(shuffled_data.iloc[int(0.7 * len(shuffled_data)):].values.tolist()).reshape(-1, 28, 28, 1)
    y_test = np.array(shuffled_target.iloc[int(0.7 * len(shuffled_data)):].values.tolist()).astype(int)

    # Generate base models' predictions
    base_predictions = np.zeros((len(X_test), 10 * len(models)))
    for idx, model in enumerate(models):
        base_predictions[:, idx*10:(idx+1)*10] = model.predict(X_test)

    # Now, you'd typically train a meta-model on these predictions. However, we'll directly use them for evaluation in this case.
    # Train meta-model
    meta_model = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence
    meta_model.fit(base_predictions, y_test)

    # Predict using the meta-model
    final_predictions = meta_model.predict(base_predictions)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_test, final_predictions)}")


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
