import xmlrpc.client
import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import joblib
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
import time
from xmlrpc.server import SimpleXMLRPCServer

CLIENT_PORT = 9094  # Unique port for each client

proxy = xmlrpc.client.ServerProxy("http://localhost:9090/RPC2")


def wait_for_start_signal():
    print("Waiting for coordinator to start training...")
    while True:
        if proxy.should_start_training():
            return True
        time.sleep(5)  # Poll every 5 seconds


client_server = SimpleXMLRPCServer(('localhost', CLIENT_PORT))
client_server.register_introspection_functions()


# Register any additional functions for client_server if needed

def run_client_server():
    client_server.serve_forever()


if __name__ == "__main__":
    # Starting the XML-RPC server in the background
    import threading

    thread = threading.Thread(target=run_client_server)
    thread.start()

    while True:
        if wait_for_start_signal():
            # Now, the rest of your training code follows
            with xmlrpc.client.ServerProxy("http://localhost:9090/RPC2") as proxy:
                print("Fetching MNIST data from Dispatcher...")
                data, target = proxy.get_mnist("client4")  # Passing client ID

                # Convert data back to numpy arrays
                data = np.array(data)[:int(0.4 * len(data))]  # Only take 40% of the dataset
                target = np.array(target)[:int(0.4 * len(target))]

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

                # Send the model back to the dispatcher
                with open('trained_model.pkl', 'rb') as model_file:
                    proxy.upload_model(xmlrpc.client.Binary(model_file.read()), "client4")

                # Make predictions on the test set
                predicted = classifier.predict(X_test)

                # Print evaluation results
                print(f"Accuracy: {accuracy_score(y_test, predicted)}")
                print(f"F1 Score: {f1_score(y_test, predicted, average='macro')}")
                print(f"Matthews Correlation Coefficient: {matthews_corrcoef(y_test, predicted)}")
                print(f"Confusion Matrix:\n{confusion_matrix(y_test, predicted)}")

            break
