from flask import Flask
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
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

    return shuffled_data.iloc[start_index:end_index].values.tolist(), shuffled_target.iloc[
                                                                      start_index:end_index].values.tolist()


server.register_function(get_mnist, 'get_mnist')


def mark_training_complete(node_id):
    nodes[node_id]['status'] = 'finished'
    return True


server.register_function(mark_training_complete, 'mark_training_complete')

models_received = 0


def upload_model(model_binary, client_id):
    global models_received
    filename = f"{client_id}_model.pkl"
    with open(filename, 'wb') as f:
        f.write(model_binary.data)
    print(f"Model from {client_id} dumped in dispatcher as {filename}.")

    models_received += 1
    if models_received == 3:
        ensemble_and_evaluate()

    return True


server.register_function(upload_model, 'upload_model')


def ensemble_and_evaluate():
    # Load the SVM models
    models = []
    for client_id in ["client1", "client2", "client3"]:
        model_name = f"{client_id}_model.pkl"   # Assuming the coordinator knows the IP of each client
        if os.path.exists(model_name):         # Check if the model exists before loading
            models.append(joblib.load(model_name))
        else:
            print(f"Model for {client_id} not found.")
            return

    # Assuming you use the latter 30% of the MNIST dataset for evaluation
    X_test, y_test = shuffled_data.iloc[int(0.7 * len(shuffled_data)):].values.tolist(), shuffled_target.iloc[
                                                                                         int(0.7 * len(
                                                                                             shuffled_data)):].values.tolist()

    # Voting based prediction
    predictions = []
    for x in X_test:
        votes = [model.predict([x])[0] for model in models]
        predictions.append(max(set(votes), key=votes.count))

    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    print(f"F1 Score: {f1_score(y_test, predictions, average='macro')}")
    print(f"Matthews Correlation Coefficient: {matthews_corrcoef(y_test, predictions)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, predictions)}")


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
