from flask import Flask, render_template
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
from sklearn.datasets import fetch_openml
from threading import Thread

# Load MNIST data
mnist = fetch_openml('mnist_784')

app = Flask(__name__)

# State information
nodes = {}  # Format: {node_id: {'status': 'working/finished'}}


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


# Create server
server = SimpleXMLRPCServer(('0.0.0.0', 9000), requestHandler=RequestHandler)
server.register_introspection_functions()


# Define a function to serve MNIST data
def get_mnist():
    # Register the node and set its status to 'working'
    node_id = str(request.remote_addr)
    nodes[node_id] = {'status': 'working'}

    return mnist.data.tolist(), mnist.target.tolist()


server.register_function(get_mnist, 'get_mnist')


# Define a function to mark the training as finished for a node
def mark_training_complete(node_id):
    nodes[node_id]['status'] = 'finished'
    return True


server.register_function(mark_training_complete, 'mark_training_complete')


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
