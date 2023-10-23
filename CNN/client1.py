import xmlrpc.client
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import time
from xmlrpc.server import SimpleXMLRPCServer

CLIENT_PORT = 9091  # Unique port for each client. Replace X with the client number.

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

def build_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Starting the XML-RPC server in the background
    import threading

    thread = threading.Thread(target=run_client_server)
    thread.start()

    while True:
        if wait_for_start_signal():
            # Fetch the MNIST data
            print("Fetching MNIST data from Dispatcher...")
            data, target = proxy.get_mnist(f"client1")  # Replace X with the client number

            # Preprocess the data for CNN
            data = np.array(data).reshape(-1, 28, 28, 1)
            target = to_categorical(np.array(target))

            model = build_cnn_model((28, 28, 1), 10)
            model.fit(data, target, epochs=10, batch_size=128, verbose=1)

            # Save the trained model to a file
            model_name = f"client1_model.h5"  # Replace X with the client number
            model.save(model_name)

            # Send the model back to the coordinator
            with open(model_name, "rb") as model_file:
                proxy.upload_model(xmlrpc.client.Binary(model_file.read()), f"client1")  # Replace X with the client number

            print("Model training completed and sent to the coordinator.")
            break