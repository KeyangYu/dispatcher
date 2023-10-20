from flask import Flask
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from threading import Thread
import pandas as pd

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
    proportion = client_ranks.get(client_id, 0.2)  # Default to 20% if rank not found

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

def upload_model(model_binary, client_id):
    filename = f"{server.client_address[0]}_{client_id}_model.pkl"
    with open(filename, 'wb') as f:
        f.write(model_binary.data)
    print(f"Model from {client_id} dumped in dispatcher as {filename}.")
    return True


server.register_function(upload_model, 'upload_model')

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
