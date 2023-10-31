from flask import Flask
from xmlrpc.server import SimpleXMLRPCRequestHandler
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle, resample
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from keras.utils import to_categorical
from threading import Thread
import pandas as pd
import joblib
import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
import time
from xmlrpc.server import SimpleXMLRPCServer
from threading import Thread
from scipy import stats

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
        # ensemble_and_evaluate_stacking()
        # ensemble_and_evaluate_bagging()
        # ensemble_and_evaluate_model_mixture()
        ensemble_and_evaluate_neural_ensemble()
    return True

server.register_function(upload_model, 'upload_model')

def build_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

def train_centralized_model():
    # Define input shape and number of classes
    input_shape = (28, 28, 1)
    num_classes = 10

    # Convert labels to one-hot encoded form (categorical)
    X_train = np.array(shuffled_data.iloc[:int(0.7 * len(shuffled_data))].values.tolist()).reshape(-1, 28, 28, 1)
    y_train = to_categorical(np.array(shuffled_target.iloc[:int(0.7 * len(shuffled_data))].values.tolist()).astype(int), num_classes=num_classes)

    # Use the build_cnn_model function
    model = build_cnn_model(input_shape, num_classes)
    model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=1)

    return model


def test_centralized_model(model):
    X_test = np.array(shuffled_data.iloc[int(0.7 * len(shuffled_data)):].values.tolist()).reshape(-1, 28, 28, 1)
    y_test = np.array(shuffled_target.iloc[int(0.7 * len(shuffled_data)):].values.tolist()).astype(int)

    y_test_one_hot = to_categorical(y_test, 10)  # Convert to one-hot vectors for categorical crossentropy
    loss, accuracy = model.evaluate(X_test, y_test_one_hot, verbose=0)

    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class labels

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # F1 Score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1 Score:", f1)

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_test, y_pred)
    print("Matthews Correlation Coefficient:", mcc)

    return accuracy


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

    meta_model = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence
    meta_model.fit(base_predictions, y_test)

    # Predict using the meta-model
    final_predictions = meta_model.predict(base_predictions)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_test, final_predictions)}")


def ensemble_and_evaluate_bagging(num_bags=3):
    # Load the client models
    all_models = []
    for client_id in ["client1", "client2", "client3"]:
        model_name = f"{client_id}_model.h5"
        if os.path.exists(model_name):
            all_models.append(load_model(model_name))
        else:
            print(f"Model for {client_id} not found.")
            return

    # Sample models (with replacement) from the client models
    bagged_models = resample(all_models, n_samples=num_bags, replace=True)

    X_test = np.array(shuffled_data.iloc[int(0.7 * len(shuffled_data)):].values.tolist()).reshape(-1, 28, 28, 1)
    y_test = np.array(shuffled_target.iloc[int(0.7 * len(shuffled_data)):].values.tolist()).astype(int)

    # Generate predictions for each model
    predictions = [model.predict(X_test) for model in bagged_models]

    # Ensemble predictions using averaging
    average_predictions = np.mean(predictions, axis=0)
    final_predictions = np.argmax(average_predictions, axis=1)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_test, final_predictions)}")

def ensemble_and_evaluate_model_mixture():
    # Load the client models
    all_models = []
    for client_id in ["client1", "client2", "client3"]:
        model_name = f"{client_id}_model.h5"
        if os.path.exists(model_name):
            all_models.append(load_model(model_name))
        else:
            print(f"Model for {client_id} not found.")
            return

    X_test = np.array(shuffled_data.iloc[int(0.7 * len(shuffled_data)):].values.tolist()).reshape(-1, 28, 28, 1)
    y_test = np.array(shuffled_target.iloc[int(0.7 * len(shuffled_data)):].values.tolist()).astype(int)

    # Generate predictions for each model
    predictions = [model.predict(X_test) for model in all_models]

    # Combine predictions using learned weights
    # (weights could be learned using a validation set and optimization techniques)
    weights = np.array([0.4, 0.3, 0.3])  # These weights should ideally be learned and not set arbitrarily
    final_prediction = np.sum(np.array(predictions) * weights[:, None, None], axis=0)

    # Convert predictions to label indices
    final_label_predictions = np.argmax(final_prediction, axis=1)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_test, final_label_predictions)}")


def ensemble_and_evaluate_neural_ensemble():
    # Load the client models
    all_models = []
    for client_id in ["client1", "client2", "client3"]:
        model_name = f"{client_id}_model.h5"
        if os.path.exists(model_name):
            all_models.append(load_model(model_name))
        else:
            print(f"Model for {client_id} not found.")
            return

    # Load your test dataset
    X_test = np.array(shuffled_data.iloc[int(0.7 * len(shuffled_data)):].values.tolist()).reshape(-1, 28, 28, 1)
    y_test = np.array(shuffled_target.iloc[int(0.7 * len(shuffled_data)):].values.tolist()).astype(int)

    # Generate predictions for each model
    predictions = [model.predict(X_test) for model in all_models]

    # Stack the predictions together
    stacked_predictions = np.hstack(predictions)

    # Build the meta-model
    meta_model = Sequential()
    meta_model.add(Dense(128, activation='relu', input_shape=(stacked_predictions.shape[1],)))
    meta_model.add(Dense(64, activation='relu'))
    meta_model.add(Dense(10, activation='softmax'))
    meta_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    y_test_one_hot = to_categorical(y_test, 10)  # Convert to one-hot vectors for categorical crossentropy

    # Train the meta-model
    meta_model.fit(stacked_predictions, y_test_one_hot, epochs=10, batch_size=32, verbose=1)

    # Make predictions
    meta_predictions = meta_model.predict(stacked_predictions)
    final_label_predictions = np.argmax(meta_predictions, axis=1)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_test, final_label_predictions)}")


def centralized_training_flow():
    # Centralized Training
    print("Starting centralized training...")
    centralized_model = train_centralized_model()
    print("Centralized training completed!")

    # Testing the centralized model
    centralized_accuracy = test_centralized_model(centralized_model)
    print(f"Centralized model accuracy: {centralized_accuracy * 100:.2f}%")


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

            # Start centralized training in parallel
            # centralized_training_thread = Thread(target=centralized_training_flow)
            # centralized_training_thread.start()

            # Wait for centralized training to complete (Optional)
            # centralized_training_thread.join()

            break

    server.serve_forever()

if __name__ == '__main__':
    rpc_thread = Thread(target=run_rpc_server)
    rpc_thread.start()
    app.run(host='0.0.0.0', port=5050)
