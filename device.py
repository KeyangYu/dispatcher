import xmlrpc.client
import time
import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import joblib
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
from flask import Flask, render_template
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
from sklearn.datasets import fetch_openml
from threading import Thread
import pandas as pd

class CustomXMLRPCRequestHandler(SimpleXMLRPCRequestHandler):
    def do_POST(self):
        self.server.client_address = self.client_address
        super().do_POST()

def load_client_ranks_from_csv(file_path):
    df = pd.read_csv(file_path)
    ranks = dict(zip(df.client_id, df.rank))
    return ranks

client_ranks = load_client_ranks_from_csv('path_to_your_csv.csv')

mnist = fetch_openml('mnist_784')

app = Flask(__name__)
nodes = {}

server = SimpleXMLRPCServer(('0.0.0.0', 9090), requestHandler=CustomXMLRPCRequestHandler)
server.register_introspection_functions()

def get_mnist():
    node_id = server.client_address[0]
    rank_percentage = client_ranks.get(node_id, 100) / 100.0
    slice_size = int(rank_percentage * len(mnist.data))

    nodes[node_id] = {'status': 'working', 'rank': client_ranks.get(node_id, 100)}
    return mnist.data.values[:slice_size].tolist(), mnist.target.values[:slice_size].tolist()

def wait_for_start_signal():
    while True:
        try:
            # Attempt to get the start signal from the coordinator
            should_start = proxy.get_start_signal()
            if should_start:
                return True
        except:
            # If connection fails or the method isn't available, just retry
            time.sleep(5)

# Connect to the RPC server (dispatcher)
if __name__ == "__main__":
    while True:
        if wait_for_start_signal():
            with xmlrpc.client.ServerProxy("http://192.168.1.234:9090/RPC2") as proxy:
                print("Fetching MNIST data from Dispatcher...")
                data, target = proxy.get_mnist()

                # Convert data back to numpy arrays
                data = np.array(data)[:int(0.4*len(data))]  # Only take 40% of the dataset
                target = np.array(target)[:int(0.4*len(target))]

                # Split data into train and test subsets
                X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, shuffle=False)

                # Standardize data
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Train a Linear SVM model
                classifier = LinearSVC(max_iter=10000)
                classifier.fit(X_train, y_train)

                # Dump the trained model to a file
                joblib.dump(classifier, 'trained_model.pkl')

                # Send the model back to the dispatcher (This is a simplified approach; in practice, you might need more intricate handling)
                with open('trained_model.pkl', 'rb') as model_file:
                    proxy.upload_model(xmlrpc.client.Binary(model_file.read()))

                # Make predictions on the test set
                predicted = classifier.predict(X_test)

                # Print evaluation results
                print(f"Accuracy: {accuracy_score(y_test, predicted)}")
                print(f"F1 Score: {f1_score(y_test, predicted, average='macro')}")
                print(f"Matthews Correlation Coefficient: {matthews_corrcoef(y_test, predicted)}")
                print(f"Confusion Matrix:\n{confusion_matrix(y_test, predicted)}")


