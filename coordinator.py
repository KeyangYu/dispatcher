from flask import Flask, render_template
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
from sklearn.datasets import fetch_openml
from threading import Thread

# Subclass the RequestHandler to capture client address in the server
class CustomXMLRPCRequestHandler(SimpleXMLRPCRequestHandler):
    def do_POST(self):
        self.server.client_address = self.client_address
        super().do_POST()

# Load MNIST data
mnist = fetch_openml('mnist_784')

app = Flask(__name__)

# State information
nodes = {}

# Create server using the custom request handler
server = SimpleXMLRPCServer(('0.0.0.0', 9090), requestHandler=CustomXMLRPCRequestHandler)
server.register_introspection_functions()

# Define a function to serve MNIST data
def get_mnist():
    # Get client IP address from the server
    node_id = server.client_address[0]
    nodes[node_id] = {'status': 'working'}
    return mnist.data.values.tolist(), mnist.target.values.tolist()


server.register_function(get_mnist, 'get_mnist')

# Define a function to mark the training as finished for a node
def mark_training_complete(node_id):
    nodes[node_id]['status'] = 'finished'
    return True

server.register_function(mark_training_complete, 'mark_training_complete')

# Function to upload the trained model
def upload_model(model_binary):
    # Save the received model to a file
    with open('received_model.pkl', 'wb') as f:
        f.write(model_binary.data)
    print("Model dumped in dispatcher.")
    return True

server.register_function(upload_model, 'upload_model')


@app.route('/')
def index():
    return render_template('status.html', nodes=nodes)

# Background thread for XMLRPC server
def run_rpc_server():
    print("Dispatcher RPC Server is running...")
    server.serve_forever()

if __name__ == '__main__':
    rpc_thread = Thread(target=run_rpc_server)
    rpc_thread.start()
    app.run(host='0.0.0.0', port=5050)
